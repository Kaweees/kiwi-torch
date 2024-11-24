#pragma once
#include <cmath>

namespace kiwitorch {
namespace ops {

// Basic arithmetic operators
template <typename T>
struct Add {
  static T forward(const T& a, const T& b) { return a + b; }
  static std::pair<T, T> backward(const T& grad) { return {grad, grad}; }
};

template <typename T>
struct Mul {
  static T forward(const T& a, const T& b) { return a * b; }
  static std::pair<T, T> backward(const T& grad, const T& a, const T& b) {
    return {grad * b, grad * a};
  }
};

// Neural network operators
template <typename T>
struct ReLU {
  static T forward(const T& x) { return x > 0 ? x : 0; }
  static T backward(const T& grad, const T& x) { return x > 0 ? grad : 0; }
};

template <typename T>
struct Sigmoid {
  static T forward(const T& x) {
    return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
  }
  static T backward(const T& grad, const T& y) {
    return grad * y * (static_cast<T>(1) - y);
  }
};

template <typename T>
struct Tanh {
  static T forward(const T& x) { return std::tanh(x); }
  static T backward(const T& grad, const T& y) {
    return grad * (static_cast<T>(1) - y * y);
  }
};

// Loss functions
template <typename T>
struct MSELoss {
  static T forward(const T& pred, const T& target) {
    auto diff = pred - target;
    return diff * diff;
  }
  static T backward(const T& pred, const T& target) {
    return static_cast<T>(2) * (pred - target);
  }
};

template <typename T>
struct CrossEntropyLoss {
  static T forward(const T& pred, const T& target) {
    return -target * std::log(pred) -
           (static_cast<T>(1) - target) * std::log(static_cast<T>(1) - pred);
  }
  static T backward(const T& pred, const T& target) {
    return (pred - target) / (pred * (static_cast<T>(1) - pred));
  }
};

}  // namespace ops
}  // namespace kiwitorch
