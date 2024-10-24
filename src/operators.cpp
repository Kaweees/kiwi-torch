#include "../include/operators.hpp"

#include <functional>
#include <vector>

namespace kiwitorch {
float Scalar::add(float x, float y) { return x + y; }

float Scalar::mul(float x, float y) { return x * y; }

float Scalar::id(float x) { return x; }

float Scalar::neg(float x) { return -x; }

bool Scalar::lt(float x, float y) { return x < y; }

bool Scalar::eq(float x, float y) { return x == y; }

float Scalar::max(float x, float y) { return x > y ? x : y; }

bool Scalar::is_close(float x, float y) { return std::abs(x - y) < 1e-6; }

float Scalar::relu(float x) { return x > 0 ? x : 0; }

float Scalar::sigmoid(float x) { return 1 / (1 + std::exp(-x)); }

float Scalar::combine3(
    std::function<float(float, float)> fn, float a, float b, float c) {
  return fn(fn(a, b), c);
}

std::function<float(float, float, float)> Scalar::combine3(
    std::function<float(float, float)> fn) {
  return [fn](float a, float b, float c) -> float { return fn(fn(a, b), c); };
}

std::function<std::vector<float>(const std::vector<float>&)> Scalar::filter(
    std::function<bool(float)> fn) {
  return [fn](const std::vector<float>& ls) -> std::vector<float> {
    std::vector<float> ret;
    for (float x : ls) {
      if (fn(x)) {
        ret.push_back(x);
      }
    }
    return ret;
  };
}
}  // namespace kiwitorch
