#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/ab_api_test.h"
#include "libspu/mpc/api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/kernel.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::test {
namespace {

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

// bool verifyCost(Kernel* kernel, std::string_view name, FieldType field,
//                 const Shape& shape, size_t npc,
//                 const Communicator::Stats& cost) {
//   if (kernel->kind() == Kernel::Kind::Dynamic) {
//     return true;
//   }

//   auto comm = kernel->comm();
//   auto latency = kernel->latency();

//   size_t numel = shape.numel();

//   bool succeed = true;
//   constexpr size_t kBitsPerBytes = 8;
//   ce::Params params = {{"K", SizeOf(field) * 8}, {"N", npc}};
//   if (comm->eval(params) * numel != cost.comm * kBitsPerBytes) {
//     fmt::print("Failed: {} comm mismatch, expected={}, got={}\n", name,
//                comm->eval(params) * numel, cost.comm * kBitsPerBytes);
//     succeed = false;
//   }
//   if (latency->eval(params) != cost.latency) {
//     fmt::print("Failed: {} latency mismatch, expected={}, got={}\n", name,
//                latency->eval(params), cost.latency);
//     succeed = false;
//   }

//   return succeed;
// }


}  // namespace

TEST_P(ArithmeticTest, HadamSS) {
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

    // 检查协议是否支持 hadam_ss 操作
    if (!obj->prot()->hasKernel("hadam_ss")) {
      return;
    }

    /* GIVEN */
    // 生成两个随机矩阵
    auto p0 = rand_p(obj.get(), shape_A);
    auto p1 = rand_p(obj.get(), shape_B);

    SPDLOG_INFO("{} {}", p0, p1);  // 打印矩阵信息便于调试

    // 转换为secret sharing形式
    auto s0 = p2s(obj.get(), p0);
    auto s1 = p2s(obj.get(), p1);

    /* WHEN */
    // 记录通信开销
    // auto prev = obj->prot()->getState<Communicator>()->getStats();
    // 执行Hadamard积运算
    auto tmp = hadam_ss(obj.get(), s0, s1).value();
    // auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;

    // 转换回明文进行验证
    auto r_ss = s2p(obj.get(), tmp);
    // 计算明文的Hadamard积作为对照
    auto r_pp = hadam_pp(obj.get(), p0, p1);

    /* THEN */
    // 验证结果
    EXPECT_VALUE_EQ(r_ss, r_pp);

  });
}

}  // namespace spu::mpc::test
