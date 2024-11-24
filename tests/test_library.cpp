#include <gtest/gtest.h>

#include <kiwitorch/kiwitorch.hpp>
using namespace kiwitorch;

TEST(LibraryTest, Add) {
  // Consider adding more test cases
  EXPECT_EQ(Scalar::add(2, 3), 5);
  EXPECT_EQ(Scalar::add(-1, 1), 0);
  EXPECT_EQ(Scalar::add(0, 0), 0);
  EXPECT_EQ(Scalar::add(-2, -3), -5);
}
