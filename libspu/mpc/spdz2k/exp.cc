// Copyright 2024 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "exp.h"

#include "type.h"
#include <cmath>
#include <random>


#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/core/encoding.h"
#include "libspu/core/trace.h"

#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/spdz2k/commitment.h"
#include "libspu/mpc/spdz2k/state.h"
#include "libspu/mpc/spdz2k/type.h"
#include "libspu/mpc/spdz2k/value.h"
#include "libspu/mpc/spdz2k/arithmetic.h"
#include "libspu/kernel/hal/fxp_cleartext.h"
// #include "libspu/mpc/spdz2k/protocol.h"



namespace spu::mpc::spdz2k {

NdArrayRef encodeToRing(const NdArrayRef& src, FieldType field, size_t fxp_bits,
                        DataType* out_type) {
  SPU_ENFORCE(src.eltype().isa<PtTy>(), "expect PtType, got={}", src.eltype());
  const PtType pt_type = src.eltype().as<PtTy>()->pt_type();
  PtBufferView pv(static_cast<const void*>(src.data()), pt_type, src.shape(),
                  src.strides());
  return encodeToRing(pv, field, fxp_bits, out_type);
}

NdArrayRef decodeFromRing(const NdArrayRef& src, DataType in_dtype,
                          size_t fxp_bits) {
  const PtType pt_type = getDecodeType(in_dtype);
  NdArrayRef dst(makePtType(pt_type), src.shape());
  PtBufferView pv(static_cast<void*>(dst.data()), pt_type, dst.shape(),
                  dst.strides());
  decodeFromRing(src, in_dtype, fxp_bits, &pv, nullptr);
  return dst;
}

template <typename FN>
NdArrayRef applyFloatingPointFn(SPUContext* ctx, const NdArrayRef& in, FN&& fn) {
  SPU_TRACE_HAL_DISP(ctx, in);
  
  // SPU_ENFORCE(in.isFxp(), "expected fxp, got={}", in.eltype());

  const size_t fxp_bits = ctx->getFxpBits();
  const auto field = in.eltype().as<Ring2k>()->field();
  const Type ring_ty = makeType<RingTy>(field);

  // 解码为浮点数
  // DataType in_dtype = getEncodeType(in.eltype().as<PtTy>()->pt_type());
  DataType in_dtype = DT_F64; 
  auto fp_arr = decodeFromRing(in.as(ring_ty), in_dtype, fxp_bits);
  auto pt_type = getDecodeType(in_dtype);


  for (auto iter = fp_arr.begin(); iter != fp_arr.end(); ++iter) {
    DISPATCH_FLOAT_PT_TYPES(pt_type, [&]() {
      auto* ptr = reinterpret_cast<ScalarT*>(&*iter);
      *ptr = fn(*ptr);
    });
  }

  DataType dtype;
  const auto out = encodeToRing(fp_arr, field, fxp_bits, &dtype);
  SPU_ENFORCE(dtype == DT_F16 || dtype == DT_F32 || dtype == DT_F64,
              "sanity failed");
  return out.as(in.eltype());
}

NdArrayRef f_exp_a(SPUContext* ctx, const NdArrayRef& in) {
  SPU_TRACE_HAL_DISP(ctx, in);
  return applyFloatingPointFn(ctx, in, [](float x) { return std::exp(x); });
}

// NdArrayRef GetMacShare(KernelEvalContext* ctx, const NdArrayRef& in) {
//   const auto field = in.eltype().as<Ring2k>()->field();
//   auto* beaver = ctx->getState<Spdz2kState>()->beaver();
//   const size_t k = ctx->getState<Spdz2kState>()->k();
//   const size_t s = ctx->getState<Spdz2kState>()->s();

//   const auto& x = getValueShare(in);
//   NdArrayRef x_mac;
//   if (in.eltype().as<AShrTy>()->hasMac()) {
//     x_mac = getMacShare(in);
//   } else {
//     SPDLOG_DEBUG("generate mac share");
//     x_mac = beaver->AuthArrayRef(x, field, k, s);
//   }
//   return x_mac;
// }
/*
NdArrayRef exp_ring(SPUContext* ctx, const NdArrayRef& in){
  // 获取字段类型和输入形状
  auto field = in.eltype().as<Ring2k>()->field();
  int64_t numel = in.numel();
  int64_t fxp_bits = ctx->getFxpBits();
  double scale = 1L << fxp_bits;

  // 创建输出数组
  NdArrayRef out(in.eltype(), in.shape());

  // 分发到所有字段类型
  DISPATCH_ALL_FIELDS(field, [&]() {
    using ring2k_t = std::make_signed<ring2k_t>::type;

    // 获取输入和输出的视图
    NdArrayView<ring2k_t> _in(in);
    NdArrayView<ring2k_t> _out(out);

    // 并行处理每个元素
    pforeach(0, numel, [&](int64_t idx) {
      // 将环元素解码为浮点数
      double x = static_cast<double>(_in[idx]) / scale;

      // 计算指数
      x = std::exp(x);

      // 将浮点数重新编码为环元素
      _out[idx] = static_cast<ring2k_t>(std::round(x * scale));

      // 确保结果在环的范围内（模 2^k）
      // ring_bitmask_(_out[idx], 0, k);
    });
  });

  return out;
}
*/


NdArrayRef ExpA::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {

  const auto field = in.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* beaver = ctx->getState<Spdz2kState>()->beaver();
  const auto key = ctx->getState<Spdz2kState>()->key();
  const auto k = ctx->getState<Spdz2kState>()->k();
  const auto s = ctx->getState<Spdz2kState>()->s();
  // size_t WorldSize = comm->getWorldSize();
  size_t rank = comm->getRank();
   

  // in
  const auto& x = getValueShare(in);
  const auto& x_mac = GetMacShare(ctx, in);

  auto start_time = std::chrono::high_resolution_clock::now();
  auto prev_comm = comm->getStats();


  // c1 = a1 * b1
  auto [vec1, mac_vec1] = beaver->AuthMul(field, in.shape(), k, s);

  auto [a1, b1, c1] = vec1;
  auto [a1_mac, b1_mac, c1_mac] = mac_vec1;

  auto [vec2, mac_vec2] = beaver->AuthMul(field, in.shape(), k, s);

  auto [a2, b2, c2] = vec2;
  auto [a2_mac, b2_mac, c2_mac] = mac_vec2;

  auto a1_ = makeAShare(a1, a1_mac, field);
  auto a2_ = makeAShare(a2, a2_mac, field);
  auto b1_ = makeAShare(b1, b1_mac, field);
  auto b2_ = makeAShare(b2, b2_mac, field);
  
  
  MulAA mulAA;
  auto c12 = mulAA.proc(ctx, a1_, b2_);
  auto c21 = mulAA.proc(ctx, a2_, b1_);


  // p0 have r=a,r_prime=a_prime
  // p1 have r=b,r_prime=b_prime
  NdArrayRef r(in.eltype(), in.shape());
  NdArrayRef r_prime(in.eltype(), in.shape());
  if(rank == 0){
    comm->sendAsync(comm->nextRank(), b1, "b1");
    comm->sendAsync(comm->nextRank(), b2, "b2");
    auto recv_a1 = comm->recv(comm->nextRank(), makeType<RingTy>(field), "a1");
    auto recv_a2 = comm->recv(comm->nextRank(), makeType<RingTy>(field), "a2");
    r = ring_add(recv_a1, a1);
    r_prime = ring_add(recv_a2, a2);
  }else{
    auto recv_b1 = comm->recv(comm->nextRank(), makeType<RingTy>(field), "b1");
    auto recv_b2 = comm->recv(comm->nextRank(), makeType<RingTy>(field), "b2");
    comm->sendAsync(comm->nextRank(), a1, "a1");
    comm->sendAsync(comm->nextRank(), a2, "a2");
    r = ring_add(recv_b1, b1);
    r_prime = ring_add(recv_b2, b2);
  }
  
  // 计算耗时和通信量
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  auto comm_cost = comm->getStats() - prev_comm;

  // 输出结果
  SPDLOG_INFO("Exp offline time: {} seconds", duration.count() / 1e6);
  SPDLOG_INFO("Exp offline communication: {} bytes", comm_cost.comm / (1024.0 * 1024.0));
  SPDLOG_INFO("Exp offline rounds: {}", comm_cost.latency);

  /*
    这里不删先,纯是测试e^x的
  */
/*
  // y_prime = e^x
  // DataType dtype = DataType::DT_F64;

  // auto y_prime = f_exp_a(ctx->sctx(), x);
  // auto y_prime = exp_ring(ctx->sctx(), x);
  // auto y = exp_ring(ctx->sctx(), x);
  // Value x_value(x, DT_F64); 
  // auto y = spu::kernel::hal::f_exp_p(ctx->sctx(), x_value).data();
  auto y = f_exp_a(ctx->sctx(), x);
  NdArrayRef x_prime;
  NdArrayRef z = ring_zeros(field, in.shape());
  NdArrayRef z_mac = ring_zeros(field, in.shape());
  if(rank == 1){
    auto key_prime = ring_mul( ring_ones(field, {x.shape()}),key);
    comm->sendAsync(comm->nextRank(), x, "x");
    comm->sendAsync(comm->nextRank(), key_prime, "key");
  }else{
    x_prime = comm->recv(comm->nextRank(), makeType<RingTy>(field), "x");
    auto key_prime = comm->recv(comm->nextRank(), makeType<RingTy>(field), "key");
    auto key_r = ring_mul( ring_ones(field, {x.shape()}), key);
    // auto y_prime = exp_ring(ctx->sctx(), x_prime);
    // Value x_prime_value(x_prime, DT_F64); 
    // auto y_prime= spu::kernel::hal::f_exp_p(ctx->sctx(), x_prime_value).data();
    auto y_prime = f_exp_a(ctx->sctx(), x_prime);
    // z0 = e^(x_1) * e^(x_2) z1 = 0
    z = ring_mul(y, y_prime);
    // z0_mac = e^(x_1) * e^(x_2)*global_key  z1_mac = 0
    z_mac = ring_mul(z, ring_add(key_r , key_prime));
  }
  return makeAShare(z, z_mac, field);
*/

  auto y_prime = f_exp_a(ctx->sctx(), x);

  // d =  y_prime - r, d_prime = spdz_key * y_prime - r_prime
  auto d = ring_sub(y_prime, r);
  auto d_prime = ring_sub(ring_mul(y_prime, key), r_prime);


  comm->sendAsync(comm->nextRank(), d, "d");
  comm->sendAsync(comm->nextRank(), d_prime, "d_prime");
  auto recv_d = comm->recv(comm->nextRank(), makeType<RingTy>(field), "d");
  auto recv_d_prime = comm->recv(comm->nextRank(), makeType<RingTy>(field), "d_prime");


  // p0 y = r * recv_d + c1  (r=a1, r_prime=a2)
  // p1 y = recv_d * d + r * recv_d + c1  (r=b1, r_prime=b2)
  auto y = ring_add(ring_mul(r, recv_d), c1);
  // auto y = ring_add(ring_mul(a, recv_d), c1);
  if(rank == 1){
    auto tmp_d = ring_mul(d, recv_d);
    ring_add_(y, tmp_d);
  }

  // y_mac = d_prime * recv_d + r_prime * recv_d + recv_d_prime * r + c12 + c21
  auto c12_ = getValueShare(c12);
  auto c21_ = getValueShare(c21);
  auto y_mac = ring_add(ring_mul(d_prime, recv_d), ring_mul(r_prime, recv_d));
  ring_add_(y_mac, ring_mul(recv_d_prime, r));
  ring_add_(y_mac, c12_);
  ring_add_(y_mac, c21_);

  return makeAShare(y, y_mac, field);
}

}  // namespace spu::mpc::spdz2k
