#include <gtest/gtest.h>

#include "../include/operators.hpp"

using namespace kiwitorch;

TEST(ScalarTest, BasicOperations) {
  // Test basic arithmetic operations
  EXPECT_EQ(Scalar::add(2, 3), 5);
  EXPECT_EQ(Scalar::mul(2, 3), 6);
  EXPECT_EQ(Scalar::neg(5), -5);
  EXPECT_EQ(Scalar::id(42), 42);
}

TEST(ScalarTest, ComparisonOperations) {
  EXPECT_TRUE(Scalar::lt(2, 3));
  EXPECT_FALSE(Scalar::lt(3, 2));
  EXPECT_TRUE(Scalar::eq(2, 2));
  EXPECT_FALSE(Scalar::eq(2, 3));
  EXPECT_EQ(Scalar::max(2, 3), 3);
  EXPECT_EQ(Scalar::max(3, 2), 3);
}

TEST(ScalarTest, AdvancedMathOperations) {
  // Test sigmoid
  EXPECT_TRUE(Scalar::is_close(Scalar::sigmoid(0), 0.5));
  EXPECT_TRUE(Scalar::sigmoid(100) < 1.0 && Scalar::sigmoid(100) > 0.99);
  EXPECT_TRUE(Scalar::sigmoid(-100) > 0.0 && Scalar::sigmoid(-100) < 0.01);

  // Test ReLU
  EXPECT_EQ(Scalar::relu(3), 3);
  EXPECT_EQ(Scalar::relu(-3), 0);
  EXPECT_EQ(Scalar::relu(0), 0);

  // Test exp and log
  EXPECT_TRUE(Scalar::is_close(Scalar::log(Scalar::exp(2)), 2));
  EXPECT_TRUE(Scalar::is_close(Scalar::exp(Scalar::log(2)), 2));
}

TEST(ScalarTest, VectorOperations) {
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