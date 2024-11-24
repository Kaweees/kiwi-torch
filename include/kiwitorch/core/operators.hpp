#pragma once

#include <vector>

namespace kiwitorch {
class Scalar {
  public:
  // Basic operators
  // Multiplies two numbers
  static double mul(double x, double y) { return x * y; }
  // Returns the input unchanged
  static double id(double x) { return x; }
  // Adds two numbers
  static double add(double x, double y) { return x + y; }
  // Negates a number
  static double neg(double x) { return -x; }
  // Checks if one number is less than another
  static bool lt(double x, double y) { return x < y; }
  // Checks if two numbers are equal
  static bool eq(double x, double y) { return x == y; }
  // Returns the larger of two numbers
  static double max(double x, double y) { return x > y ? x : y; }
  // Checks if two numbers are close in value
  static bool is_close(double x, double y) { return std::abs(x - y) < 1e-6; }

  // More complex operators
  // Calculates the sigmoid function
  static double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
  // Applies the ReLU activation function
  static double relu(double x) { return x > 0 ? x : 0; }
  // Calculates the natural logarithm
  static double log(double x) { return std::log(x); }
  // Calculates the exponential function
  static double exp(double x) { return std::exp(x); }
  // Calculates the reciprocal
  static double inv(double x) { return 1 / x; }

  // Derivative functions
  // Computes the derivative of log times a second arg
  static double logBack(double x, double y) { return y / x; }
  // Computes the derivative of reciprocal times a second arg
  static double invBack(double x, double y) { return -y / (x * x); }
  // Computes the derivative of ReLU times a second arg
  static double reluBack(double x, double y) { return x > 0 ? y : 0; }

  // Higher-order functions
  template <typename F>
  static std::vector<double> map(F fn, const std::vector<double>& ls) {
    std::vector<double> result;
    result.reserve(ls.size());
    for (const auto& x : ls) {
      result.push_back(fn(x));
    }
    return result;
  }

  static std::vector<double> negList(const std::vector<double>& ls) {
    return map(neg, ls);
  }

  template <typename F>
  static std::vector<double> zipWith(
      F fn, const std::vector<double>& ls1, const std::vector<double>& ls2) {
    std::vector<double> result;
    size_t size = std::min(ls1.size(), ls2.size());
    result.reserve(size);
    for (size_t i = 0; i < size; ++i) {
      result.push_back(fn(ls1[i], ls2[i]));
    }
    return result;
  }

  static std::vector<double> addLists(
      const std::vector<double>& ls1, const std::vector<double>& ls2) {
    return zipWith(add, ls1, ls2);
  }

  template <typename F>
  static double reduce(F fn, double start, const std::vector<double>& ls) {
    double result = start;
    for (const auto& x : ls) {
      result = fn(result, x);
    }
    return result;
  }

  static double sum(const std::vector<double>& ls) {
    return reduce(add, 0.0, ls);
  }

  static double prod(const std::vector<double>& ls) {
    return reduce(mul, 1.0, ls);
  }
};
}  // namespace kiwitorch
