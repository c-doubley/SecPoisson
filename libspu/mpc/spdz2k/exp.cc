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
  DataType in_dtype = getEncodeType(in.eltype().as<PtTy>()->pt_type());
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

NdArrayRef GetMacShare(KernelEvalContext* ctx, const NdArrayRef& in) {
  const auto field = in.eltype().as<Ring2k>()->field();
  auto* beaver = ctx->getState<Spdz2kState>()->beaver();
  const size_t k = ctx->getState<Spdz2kState>()->k();
  const size_t s = ctx->getState<Spdz2kState>()->s();

  const auto& x = getValueShare(in);
  NdArrayRef x_mac;
  if (in.eltype().as<AShrTy>()->hasMac()) {
    x_mac = getMacShare(in);
  } else {
    SPDLOG_DEBUG("generate mac share");
    x_mac = beaver->AuthArrayRef(x, field, k, s);
  }
  return x_mac;
}

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


  // c1 = a1 * b1
  auto [vec1, mac_vec1] = beaver->AuthMul(field, in.shape(), k, s);

  auto [a1, b1, c1] = vec1;
  auto [a_mac1, b_mac1, c_mac1] = mac_vec1;

  // c12 = a1* b2  c21 = a2 * b1
  auto a2 = ring_rand(field, in.shape());
  auto b2 = ring_rand(field, in.shape());

  MulAA mulAA;
  auto c12 = mulAA.proc(ctx, a1, b2);
  auto c21 = mulAA.proc(ctx, a2, b1);

  // p0 have r=a,r_prime=a_prime
  // p1 have r=b,r_prime=b_prime
  NdArrayRef r(field, in.shape());
  NdArrayRef r_prime(field, in.shape());
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


  // // w = v + a * b
  // // Sender: input a, receive v
  // // Receiver: input b, receive w
  // std::vector<NdArrayRef> w1, v1;
  // std::vector<NdArrayRef> w2, v2;
  // for (size_t i = 0; i < WorldSize; ++i) {
  //   for (size_t j = 0; j < WorldSize; ++j) {
  //     if (i == j) {
  //       continue;
  //     }

  //     if (i == rank) {
  //       auto tmp1 = beaver->voleRecv(field, b2);
  //       w1.emplace_back(tmp1);
  //       auto tmp2 = voleRecv(field, b1);
  //       w2.emplace_back(tmp2);
  //     }
  //     if (j == rank) {
  //       auto tmp1 = voleSend(field, a1);
  //       v1.emplace_back(tmp1);
  //       auto tmp2 = voleSend(field, a2);
  //       v2.emplace_back(tmp2);
  //     }
  //   }
  // }

  // a * b = w - v
  // auto a1_b2 = ring_zeros(field, in.shape());
  // auto a2_b1 = ring_zeros(field, in.shape());
  // for (size_t i = 0; i < WorldSize - 1; ++i) {
  //   ring_add_(a1_b2, ring_sub(w1[i], v1[i]));
  //   ring_add_(a2_b1, ring_sub(w2[i], v2[i]));
  // }
  
  // c12 = a1[0]b2[0] + a1[1]b2[0] + a1[0]b2[1] + a1[1]b2[1] 
  // a1*b2 = a1[0]b2[0] a1_b2 = a1[1]b2[0]
  // auto c12 = ring_add(ring_mul(a1, b2), a1_b2);
  // auto c21 = ring_add(ring_mul(a2, b1), a2_b1);
  // NdArrayView<T> _c12(c12);
  // NdArrayView<T> _c21(c21);



  // y_prime = e^x
  auto y_prime = f_exp_a(ctx, x);

  // d =  y_prime - r, d_prime = spdz_key * y_prime - r_prime
  auto d = ring_sub(y_prime, r);
  auto d_prime = ring_sub(ring_mul(key, y_prime), r_prime);


  comm->sendAsync(comm->nextRank(), d, "d");
  comm->sendAsync(comm->nextRank(), d_prime, "d_prime");
  auto recv_d = comm->recv(comm->nextRank(), makeType<RingTy>(field), "d");
  auto recv_d_prime = comm->recv(comm->nextRank(), makeType<RingTy>(field), "d_prime");


  // p0 y = r * recv_d + c1  (r=a1, r_prime=a2)
  // p1 y = recv_d * d + r * recv_d + c1  (r=b1, r_prime=b2)
  auto y = ring_add(ring_mul(r, recv_d), c1);
  if(rank == 1){
    ring_mul(d,recv_d);
    ring_add_(y, d);
  }

  // y_mac = d_prime * recv_d + r_prime * recv_d + recv_d_prime * r + c12 + c21
  auto y_mac = ring_add(ring_mul(d_prime, recv_d), ring_mul(r_prime, recv_d));
  ring_add_(y_mac, ring_mul(recv_d_prime, r));
  ring_add_(y_mac, c12);
  ring_add_(y_mac, c21);



  return makeAShare(y, y_mac, field);
}

}  // namespace spu::mpc::spdz2k
