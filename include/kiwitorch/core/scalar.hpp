#pragma once
#include <functional>
#include <memory>

namespace kiwitorch {

class Scalar {
  public:
  // Constructors
  Scalar(float data = 0.0f, bool requires_grad = false);

  // Arithmetic operations
  Scalar operator+(const Scalar& other) const;
  Scalar operator-(const Scalar& other) const;
  Scalar operator*(const Scalar& other) const;
  Scalar operator/(const Scalar& other) const;

  // Neural network operations
  Scalar relu() const;
  Scalar sigmoid() const;
  Scalar tanh() const;

  // Gradient operations
  void backward(float gradient = 1.0f);
  bool requires_grad() const { return requires_grad_; }
  float grad() const { return grad_; }

  // Accessors
  float data() const { return data_; }
  void set_data(float data) { data_ = data; }

  private:
  struct AutogradHistory {
    std::function<void(float)> backward_fn;
    std::vector<Scalar*> dependencies;
  };

  float data_;
  float grad_ = 0.0f;
  bool requires_grad_ = false;
  std::shared_ptr<AutogradHistory> history_;

  // Helper for creating new Scalars with autograd history
  static Scalar binary_op(const Scalar& a, const Scalar& b,
      std::function<float(float, float)> forward,
      std::function<std::pair<float, float>(float)> backward);
};

}  // namespace kiwitorch
