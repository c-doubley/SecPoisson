#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "libspu/mpc/spdz2k/arithmetic.h"
#include "libspu/mpc/spdz2k/state.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::spdz2k {

class ArithmeticTest : public testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(ArithmeticTest, HadamAA) {
  // 定义测试参数
  size_t npc = 2;  // 2方计算
  const int64_t M = 70;  // 行数
  const int64_t N = 60;  // 列数
  const Shape shape_A = {M, N};
  const Shape shape_B = {M, N};  // Hadamard积要求两个矩阵维度相同

  RuntimeConfig conf;
  conf.set_protocol(ProtocolKind::SPDZ2K);
  conf.set_field(FieldType::FM64);  // 或其他适合的field类型

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    // 创建SPU上下文
    KernelEvalContext ctx;
    ctx.setRuntimeConfig(conf);
    
    // 设置必要的状态
    ctx.setState<Communicator>(std::make_shared<Communicator>(lctx));
    auto spdz2k_state = std::make_shared<Spdz2kState>();
    // 设置必要的SPDZ2k参数
    spdz2k_state->k_ = 64;  // for FM64
    spdz2k_state->s_ = 8;   // security parameter
    ctx.setState(spdz2k_state);

    // 创建HadamAA kernel
    HadamAA kernel;

    /* GIVEN */
    // 生成随机输入
    auto x = ring_rand(conf.field(), shape_A);
    auto y = ring_rand(conf.field(), shape_B);

    // 创建认证分享
    auto x_mac = ring_rand(conf.field(), shape_A);
    auto y_mac = ring_rand(conf.field(), shape_B);

    auto x_share = makeAShare(x, x_mac, conf.field());
    auto y_share = makeAShare(y, y_mac, conf.field());

    /* WHEN */
    // 执行Hadamard乘法
    auto z = kernel.proc(&ctx, x_share, y_share);

    /* THEN */
    // 验证结果
    // 1. 验证形状
    EXPECT_EQ(z.shape(), shape_A);
    
    // 2. 验证类型
    EXPECT_EQ(z.eltype(), makeType<AShrTy>(conf.field()));

    // 3. 如果需要，可以验证具体的计算结果
    // 注意：在实际的secret sharing中，单个分享的值并不能反映最终结果
    // 需要各方的分享组合才能得到真实结果
  });
}

}  // namespace spu::mpc::spdz2k