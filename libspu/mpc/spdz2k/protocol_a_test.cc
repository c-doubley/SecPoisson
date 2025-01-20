#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/ab_api_test.h"
#include "libspu/mpc/api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/kernel.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"
#include "libspu/mpc/spdz2k/protocol.h"


#include <mutex>
#include <random>

#include "gtest/gtest.h"
// #include "yacl/crypto/key_utils.h"


#include "libspu/mpc/spdz2k/exp.h"
#include "libspu/mpc/spdz2k/type.h"
#include "libspu/mpc/spdz2k/value.h"
#include "libspu/mpc/spdz2k/state.h"
#include "libspu/mpc/spdz2k/arithmetic.h"


namespace spu::mpc::test {
namespace {


RuntimeConfig makeConfig(FieldType field) {
  RuntimeConfig conf;
  conf.set_protocol(ProtocolKind::SPDZ2K);  // FIXME:
  conf.set_field(field);
  if (field == FieldType::FM64) {
    conf.set_fxp_fraction_bits(17);
  } 
  conf.set_experimental_enable_exp_prime(true);
  return conf;
}

// std::unique_ptr<SPUContext> makeMpcSpdz2kProtocol(
//     const RuntimeConfig& rt, const std::shared_ptr<yacl::link::Context>& lctx) {
//   RuntimeConfig mpc_rt = rt;
//   mpc_rt.set_beaver_type(RuntimeConfig_BeaverType_MultiParty);

//   return makeSpdz2kProtocol(mpc_rt, lctx);
// }
}  // namespace

// INSTANTIATE_TEST_SUITE_P(
//     Spdz2k, ArithmeticTest,
//     testing::Values(testing::Values(CreateObjectFn(makeSpdz2kProtocol,
//                                                     "tfp")),        //
//                      testing::Values(makeConfig(FieldType::FM64)),  //
//                      testing::Values(2)),                           //
//     [](const testing::TestParamInfo<ArithmeticTest::ParamType>& p) {
//       return fmt::format("{}x{}x{}", std::get<0>(p.param).name(),
//                          std::get<1>(p.param).field(), std::get<2>(p.param));
//     });


class BeaverCacheTest : public ::testing::TestWithParam<OpTestParams> {};

// INSTANTIATE_TEST_SUITE_P(
//     Spdz2k, BeaverCacheTest,
//     testing::Combine(testing::Values(CreateObjectFn(makeSpdz2kProtocol, "tfp")),
//                                      CreateObjectFn(makeMpcSpdz2kProtocol, "mpc"),         //
//                      testing::Values(makeConfig(FieldType::FM32),    //
//                                      makeConfig(FieldType::FM64),    //
//                                      makeConfig(FieldType::FM128)),  //
//                      testing::Values(2, 3, 5)),                      //
//     [](const testing::TestParamInfo<BeaverCacheTest::ParamType>& p) {
//       return fmt::format("{}x{}x{}", std::get<0>(p.param).name(),
//                          std::get<1>(p.param).field(), std::get<2>(p.param));
//       ;
//     });

INSTANTIATE_TEST_SUITE_P(
    Spdz2k, ArithmeticTest,
    testing::Values(
        std::tuple{CreateObjectFn(makeSpdz2kProtocol, "tfp"), makeConfig(FieldType::FM64), 2}
    ),
    [](const testing::TestParamInfo<ArithmeticTest::ParamType>& p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param).name(),
                         std::get<1>(p.param).field(), std::get<2>(p.param));
    }
);

INSTANTIATE_TEST_SUITE_P(
    Spdz2k, BeaverCacheTest,
    testing::Combine(testing::Values(CreateObjectFn(makeSpdz2kProtocol,
                                                    "tfp")),        //
                     testing::Values(makeConfig(FieldType::FM64)),  //
                     testing::Values(2)),   
    [](const testing::TestParamInfo<BeaverCacheTest::ParamType>& p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param).name(),
                         std::get<1>(p.param).field(), std::get<2>(p.param));
    }
);

Shape kShape = {30, 40};
const std::vector<size_t> kShiftBits = {0, 1, 2, 31, 32, 33, 64, 1000};

#define EXPECT_VALUE_EQ(X, Y)                            \
  {                                                      \
    EXPECT_EQ((X).dtype(), (Y).dtype());                 \
    EXPECT_EQ((X).shape(), (Y).shape());                 \
    EXPECT_TRUE(ring_all_equal((X).data(), (Y).data())); \
  }

#define EXPECT_VALUE_ALMOST_EQ(X, Y, ERR)                \
  {                                                      \
    EXPECT_EQ((X).dtype(), (Y).dtype());                 \
    EXPECT_EQ((X).shape(), (Y).shape());                 \
    EXPECT_TRUE(ring_all_equal((X).data(), (Y).data())); \
  }

bool verifyCost(Kernel* kernel, std::string_view name, FieldType field,
                const Shape& shape, size_t npc,
                const Communicator::Stats& cost) {
  if (kernel->kind() == Kernel::Kind::Dynamic) {
    return true;
  }

  auto comm = kernel->comm();
  auto latency = kernel->latency();

  size_t numel = shape.numel();

  bool succeed = true;
  constexpr size_t kBitsPerBytes = 8;
  ce::Params params = {{"K", SizeOf(field) * 8}, {"N", npc}};
  if (comm->eval(params) * numel != cost.comm * kBitsPerBytes) {
    fmt::print("Failed: {} comm mismatch, expected={}, got={}\n", name,
               comm->eval(params) * numel, cost.comm * kBitsPerBytes);
    succeed = false;
  }
  if (latency->eval(params) != cost.latency) {
    fmt::print("Failed: {} latency mismatch, expected={}, got={}\n", name,
               latency->eval(params), cost.latency);
    succeed = false;
  }

  return succeed;
}



TEST_P(ArithmeticTest, HadamSS) {
  printf("\n=== Starting test case HadamSS ===\n");
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  // 定义矩阵维度
  const int64_t M = 70;  // 行数
  const int64_t N = 60;  // 列数
  const Shape shape_A = {M, N};
  const Shape shape_B = {M, N};  // Hadamard积要求两个矩阵维度相同

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    // // 检查协议是否支持 hadam_ss 操作
    // if (!obj->prot()->hasKernel("hadam_ss")) {
    //   return;
    // }

    /* GIVEN */
    // 生成两个随机矩阵
    auto p0 = rand_p(obj.get(), shape_A);
    auto p1 = rand_p(obj.get(), shape_B);

    // SPDLOG_INFO("Input matrices:");
    // SPDLOG_INFO("p0: \n{}", p0.toString());  // 打印输入矩阵p0
    // SPDLOG_INFO("p1: \n{}", p1.toString());  // 打印输入矩阵p1

    // 转换为secret sharing形式
    auto s0 = p2s(obj.get(), p0);
    auto s1 = p2s(obj.get(), p1);

    /* WHEN */

    // 记录通信开销
    auto prev = obj->prot()->getState<Communicator>()->getStats();
    printf("=== Starting Hadamard multiplication test ===\n");

    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 执行Hadamard积运算
    auto tmp = hadam_ss(obj.get(), s0, s1).value();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;
        


    /* THEN */
    printf("=== Performance Results ===\n");
    printf("Communication: %zu bytes\n", cost.comm);
    printf("Latency: %zu rounds\n", cost.latency);
    printf("Runtime: %ld microseconds\n", duration.count());
    printf("=========================\n");

    
    // 转换回明文进行验证
    auto r_ss = s2p(obj.get(), tmp);
    // 计算明文的Hadamard积作为对照
    auto r_pp = mul_pp(obj.get(), p0, p1);

    /* THEN */
    EXPECT_VALUE_EQ(r_ss, r_pp);
    EXPECT_TRUE(verifyCost(obj->getKernel("hadam_aa"), "hadam_aa", conf.field(),
                        kShape, npc, cost));

  });

  printf("=== Test case HadamSS completed ===\n\n");
}


FieldType getRuntimeField(FieldType data_field) {
  switch (data_field) {
    case FM32:
      return FM64;
    case FM64:
      return FM128;
    default:
      SPU_THROW("unsupported data field {} for spdz2k", data_field);
  }
  return FT_INVALID;
}

NdArrayRef CastRing(const NdArrayRef& in, FieldType out_field) {
  const auto* in_ty = in.eltype().as<Ring2k>();
  const auto in_field = in_ty->field();
  auto out = ring_zeros(out_field, in.shape());

  return DISPATCH_ALL_FIELDS(in_field, [&]() {
    NdArrayView<ring2k_t> _in(in);
    return DISPATCH_ALL_FIELDS(out_field, [&]() {
      NdArrayView<ring2k_t> _out(out);
      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx] = static_cast<ring2k_t>(_in[idx]);
      });

      return out;
    });
  });
}

TEST_P(BeaverCacheTest, ExpA) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  // spdz2k exp only supports 2 party
  if (npc != 2 ) {
    return;
  }
  auto fxp = conf.fxp_fraction_bits();

  NdArrayRef ring2k_shr[2];

  int64_t numel = 10;
  FieldType field = conf.field();
  FieldType runtime_field = getRuntimeField(field);

  std::uniform_real_distribution<double> dist(-18.0, 15.0);
  std::default_random_engine rd;
  std::vector<double> real_vec(numel);
  for (int64_t i = 0; i < numel; ++i) {
    real_vec[i] = static_cast<double>(std::round((dist(rd) * (1L << fxp)))) / (1L << fxp);
  }

  auto rnd_msg = ring_zeros(field, {numel});


  DISPATCH_ALL_FIELDS(field, [&]() {
    using sT = std::make_signed<ring2k_t>::type;
    NdArrayView<sT> xmsg(rnd_msg);
    pforeach(0, numel, [&](int64_t i) {
      xmsg[i] = std::round(real_vec[i] * (1L << fxp));
    });
  });

  NdArrayRef outp[2];
  NdArrayRef outp_shr[2];
  NdArrayRef got;

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    KernelEvalContext kcontext(obj.get());

    int rank = lctx->Rank();

    size_t bytes = lctx->GetStats()->sent_bytes;
    size_t action = lctx->GetStats()->sent_actions;
    // auto* beaver = kcontext.getState<Spdz2kState>()->beaver();
    // const auto k = kcontext.getState<Spdz2kState>()->k();
    // const auto s = kcontext.getState<Spdz2kState>()->s();
    // auto* comm = kcontext.getState<Communicator>();

    spu::mpc::spdz2k::P2A p2a;
    ring2k_shr[rank] = p2a.proc(&kcontext, rnd_msg);

    spu::mpc::spdz2k::ExpA exp;
    // spu::mpc::spdz2k::NegateA exp;

    outp[rank] = exp.proc(&kcontext, ring2k_shr[rank]);

    bytes = lctx->GetStats()->sent_bytes - bytes;
    action = lctx->GetStats()->sent_actions - action;
    SPDLOG_INFO("Spdz2kExpA ({}) for n = {}, sent {} MiB ({} B per), actions {}",
                runtime_field, numel, bytes * 1. / 1024. / 1024., bytes * 1. / numel,
                action);


    spu::mpc::spdz2k::A2P a2p;
    got = a2p.proc(&kcontext, outp[rank]);
    
  });

/*
  NdArrayRef outp_pub;
  // NdArrayRef outp_shr[2];
  NdArrayRef got;

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    KernelEvalContext kcontext(obj.get());

    // int rank = lctx->Rank();

    size_t bytes = lctx->GetStats()->sent_bytes;
    size_t action = lctx->GetStats()->sent_actions;
    // auto* beaver = kcontext.getState<Spdz2kState>()->beaver();
    // const auto k = kcontext.getState<Spdz2kState>()->k();
    // const auto s = kcontext.getState<Spdz2kState>()->s();

    spu::mpc::spdz2k::P2A p2a;
    auto rnd_msg_shr = p2a.proc(&kcontext, rnd_msg);
    // std::cout << "Element type of rnd_msg_shr: " << rnd_msg_shr.eltype().toString() << std::endl;
    // auto tmp_shr = ring_rand(runtime_field, rnd_msg_shr.shape());
    // auto tmp_shr_mac = beaver->AuthArrayRef(tmp_shr, runtime_field, k, s);
    // auto tmp_shr1 = ring_sub(rnd_msg_shr, tmp_shr);
    // std::cout << "ring2k_shr[0] type after assignment: " << ring2k_shr[0].eltype() << std::endl;
    // auto tmp_shr1_mac = beaver->AuthArrayRef(tmp_shr1, runtime_field, k, s);

    // ring2k_shr[0] = spu::mpc::spdz2k::makeAShare(tmp_shr, tmp_shr_mac, runtime_field);
    // std::cout << "ring2k_shr[0] type after assignment: " << ring2k_shr[0].eltype() << std::endl;
    // ring2k_shr[1] = spu::mpc::spdz2k::makeAShare(tmp_shr1, tmp_shr1_mac, runtime_field);
    
    // std::cout << "ring2k_shr[1] type after assignment: " << ring2k_shr[1].eltype() << std::endl;

    spu::mpc::spdz2k::ExpA exp;
    // outp_shr[rank] = exp.proc(&kcontext, ring2k_shr[rank]);
    auto outp_shr = exp.proc(&kcontext, rnd_msg_shr);

    bytes = lctx->GetStats()->sent_bytes - bytes;
    action = lctx->GetStats()->sent_actions - action;
    SPDLOG_INFO("Spdz2kExpA ({}) for n = {}, sent {} MiB ({} B per), actions {}",
                runtime_field, numel, bytes * 1. / 1024. / 1024., bytes * 1. / numel,
                action);

    // assert(outp_shr[0].eltype() == ring2k_shr[0].eltype());

    spu::mpc::spdz2k::A2P a2p;
    // NdArrayRef outp[2];
    // std::cout << "expa here2" << std::endl;
    // outp[0] = a2p.proc(&kcontext, outp_shr[0]);
    // outp[1] = a2p.proc(&kcontext, outp_shr[1]);
    // got = ring_add(outp[0], outp[1]);
    got = a2p.proc(&kcontext, outp_shr);
  });
*/

  ring_print(got, "exp result");
  DISPATCH_ALL_FIELDS(field, [&]() {
    using sT = std::make_signed<ring2k_t>::type;
    NdArrayView<sT> got_view(got);

    double max_err = 0.0;
    for (int64_t i = 0; i < numel; ++i) {
      double expected = std::exp(real_vec[i]);
      expected = static_cast<double>(std::round((expected * (1L << fxp)))) / (1L << fxp);
      double got = static_cast<double>(got_view[i]) / (1L << fxp);
      std::cout << "real_vec[i]: " << fmt::format("{0:f}", real_vec[i]) << std::endl;
      std::cout << "expected: " << fmt::format("{0:f}", expected)
                << ", got: " << fmt::format("{0:f}", got) << std::endl;
      std::cout << "expected: "
                << fmt::format("{0:b}",
                               static_cast<ring2k_t>(expected * (1L << fxp)))
                << ", got: " << fmt::format("{0:b}", got_view[i]) << std::endl;
      max_err = std::max(max_err, std::abs(expected - got));
    }
    ASSERT_LE(max_err, 1e-0);
  });

}



}  // namespace spu::mpc::test
