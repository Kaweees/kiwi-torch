#include "../include/operators.hpp"

float add(float a, float b) { return a + b; }

float mul(float a, float b) { return a * b; }

float combine3(
    std::function<float(float, float)> fn, float a, float b, float c) {
  return fn(fn(a, b), c);
}

std::function<float(float, float, float)> combine3(
    std::function<float(float, float)> fn) {
  return [fn](float a, float b, float c) -> float { return fn(fn(a, b), c); };
}

std::function<std::vector<float>(const std::vector<float>&)> filter(
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
