# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import pandas as pd
import jax
import jax.numpy as jnp
from sklearn import metrics

import spu
import spu.utils.distributed as ppd

def predict(x, w, b):
    # 泊松回归使用指数链接函数
    return jnp.exp(jnp.matmul(x, w) + b)

def poisson_loss(x, y, w, b, use_cache):
    if use_cache:
        w = spu.experimental.make_cached_var(w)
        b = spu.experimental.make_cached_var(b)
    
    pred = predict(x, w, b)
    # 泊松回归的负对数似然损失
    loss = jnp.mean(pred - y * jnp.log(pred + 1e-10))  # 添加小epsilon防止log(0)
    
    if use_cache:
        w = spu.experimental.drop_cached_var(w, loss)
        b = spu.experimental.drop_cached_var(b, loss)
    
    return loss

class PoissonRegression:
    def __init__(self, n_epochs=20, n_iters=5, step_size=0.05):
        self.n_epochs = n_epochs
        self.n_iters = n_iters
        self.step_size = step_size

    def fit_auto_grad(self, feature, label, use_cache=False):
        w = jnp.zeros(feature.shape[1])
        b = 0.0

        if use_cache:
            feature = spu.experimental.make_cached_var(feature)

        xs = jnp.array_split(feature, self.n_iters, axis=0)
        ys = jnp.array_split(label, self.n_iters, axis=0)

        def body_fun(_, loop_carry):
            w_, b_ = loop_carry
            for x, y in zip(xs, ys):
                grad = jax.grad(poisson_loss, argnums=(2, 3))(x, y, w_, b_, use_cache)
                w_ -= grad[0] * self.step_size
                b_ -= grad[1] * self.step_size
            return w_, b_

        ret = jax.lax.fori_loop(0, self.n_epochs, body_fun, (w, b))

        if use_cache:
            feature = spu.experimental.drop_cached_var(feature, *ret)

        return ret

def run_on_cpu(x_train, y_train):
    pr = PoissonRegression()
    w, b = jax.jit(pr.fit_auto_grad)(x_train, y_train)
    return w, b

SPU_OBJECT_META_PATH = "/tmp/driver_spu_jax_pr_object.txt"

import cloudpickle as pickle

def save_and_load_model(x_test, y_test, W, b):
    meta = ppd.save((W, b))
    with open(SPU_OBJECT_META_PATH, "wb") as f:
        pickle.dump(meta, f)

    with open(SPU_OBJECT_META_PATH, "rb") as f:
        meta_ = pickle.load(f)
    W_, b_ = ppd.load(meta_)
    W_r, b_r = ppd.get(W_), ppd.get(b_)
    print("Weights:", W_r, "Bias:", b_r)
    
    pred = predict(x_test, W_r, b_r)
    mse = jnp.mean((y_test - pred) ** 2)
    print(f"MSE(save_and_load_model)={mse}")
    return mse

def compute_score(x_test, y_test, W_r, b_r, type):
    pred = predict(x_test, W_r, b_r)
    mse = jnp.mean((y_test - pred) ** 2)
    print(f"MSE({type})={mse}")
    # 对于二值数据，额外计算准确率
    pred_binary = (pred > 0.5).astype(int)
    accuracy = jnp.mean(pred_binary == y_test)
    print(f"Accuracy({type})={accuracy}")
    return mse

def run_on_spu(x, y, use_cache=False):
    @ppd.device("SPU")
    def train(x1, x2, y):
        x = jnp.concatenate((x1, x2), axis=1)
        pr = PoissonRegression()
        return pr.fit_auto_grad(x, y, use_cache)

    split_idx = x.shape[1] // 2
    x1 = ppd.device("P1")(lambda x: x[:, :split_idx])(x)
    x2 = ppd.device("P2")(lambda x: x[:, split_idx:])(x)
    y = ppd.device("P1")(lambda x: x)(y)
    W, b = train(x1, x2, y)
    return W, b

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='distributed driver.')
    parser.add_argument(
        "-c", "--config", default="examples/python/conf/2pc_semi2k.json"
    )
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        conf = json.load(file)

    ppd.init(conf["nodes"], conf["devices"])

    # 读取本地数据集
    x = pd.read_csv("examples/python/ml/jax_pr/balloonX.csv", header=None).values
    y = pd.read_csv("examples/python/ml/jax_pr/balloony.csv", header=None).values.ravel()
    x = jnp.array(x, dtype=jnp.float32)
    y = jnp.array(y, dtype=jnp.float32)

    print('Run on CPU\n------\n')
    w, b = run_on_cpu(x, y)
    compute_score(x, y, w, b, 'cpu, auto_grad')

    print('Run on SPU\n------\n')
    w, b = run_on_spu(x, y)
    w_r, b_r = ppd.get(w), ppd.get(b)
    compute_score(x, y, w_r, b_r, 'spu')
    save_and_load_model(x, y, w, b)

    print('Run on SPU with cache\n------\n')
    w, b = run_on_spu(x, y, True)
    w_r, b_r = ppd.get(w), ppd.get(b)
    compute_score(x, y, w_r, b_r, 'spu_cached')