#include <gtest/gtest.h>

#include <kiwitorch/kiwitorch.hpp>
using namespace kiwitorch;

// Helper function for floating point comparison
bool assert_close(float a, float b, float rtol = 1e-5) { return std::abs(a - b) < rtol; }

// Basic arithmetic tests
TEST(ScalarTest, BasicOperations) {
  EXPECT_EQ(Scalar::add(2, 3), 5);
  EXPECT_EQ(Scalar::mul(2, 3), 6);
  EXPECT_EQ(Scalar::neg(5), -5);
  EXPECT_EQ(Scalar::id(42), 42);
}

// Basic operator tests
TEST(OperatorsTest, BasicOperations) {
  float x = 1.5f;
  float y = 2.0f;

  EXPECT_TRUE(assert_close(Scalar::mul(x, y), x * y));
  EXPECT_TRUE(assert_close(Scalar::add(x, y), x + y));
  EXPECT_TRUE(assert_close(Scalar::neg(x), -x));
  EXPECT_TRUE(assert_close(Scalar::max(x, y), std::max(x, y)));
  if (std::abs(x) > 1e-5) { EXPECT_TRUE(assert_close(Scalar::inv(x), 1.0f / x)); }
}

TEST(OperatorsTest, ReluTest) {
  float a = 1.5f;
  EXPECT_TRUE(assert_close(Scalar::relu(a), a));

  a = -1.5f;
  EXPECT_TRUE(assert_close(Scalar::relu(a), 0.0f));
}

TEST(OperatorsTest, ReluBackTest) {
  float a = 1.5f;
  float b = 2.0f;
  EXPECT_TRUE(assert_close(Scalar::reluBack(a, b), b));

  a = -1.5f;
  EXPECT_TRUE(assert_close(Scalar::reluBack(a, b), 0.0f));
}

// Property tests
TEST(OperatorsTest, SigmoidProperties) {
  float a = 1.5f;
  float sig_a = Scalar::sigmoid(a);

  // Between 0 and 1
  EXPECT_TRUE(sig_a >= 0.0f && sig_a <= 1.0f);

  // 1 - sigmoid(x) = sigmoid(-x)
  EXPECT_TRUE(assert_close(1.0f - Scalar::sigmoid(a), Scalar::sigmoid(-a)));

  // Crosses 0.5 at x=0
  EXPECT_TRUE(assert_close(Scalar::sigmoid(0.0f), 0.5f));

  // Strictly increasing
  float b = a + 0.1f;
  EXPECT_TRUE(Scalar::sigmoid(b) > Scalar::sigmoid(a));
}

// List operation tests
TEST(OperatorsTest, ListOperations) {
  std::vector<double> v1 = {1, 2, 3};
  std::vector<double> v2 = {4, 5, 6};

  // Test map
  auto neg_result = Scalar::negList(v1);
  EXPECT_EQ(neg_result.size(), 3);
  EXPECT_EQ(neg_result[0], -1);
  EXPECT_EQ(neg_result[1], -2);
  EXPECT_EQ(neg_result[2], -3);

  // Test zipWith
  auto sum_result = Scalar::addLists(v1, v2);
  EXPECT_EQ(sum_result.size(), 3);
  EXPECT_EQ(sum_result[0], 5);
  EXPECT_EQ(sum_result[1], 7);
  EXPECT_EQ(sum_result[2], 9);

  // Test reduce
  EXPECT_EQ(Scalar::sum(v1), 6);
  EXPECT_EQ(Scalar::prod(v1), 6);
}

TEST(OperatorsTest, NegListTest) {
  std::vector<double> ls = {1.0f, -2.0f, 3.0f};
  auto result = Scalar::negList(ls);

  for (size_t i = 0; i < ls.size(); i++) { EXPECT_TRUE(assert_close(ls[i], -result[i])); }
}
