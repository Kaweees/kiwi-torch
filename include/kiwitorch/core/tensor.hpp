#pragma once
#include <functional>
#include <memory>
#include <vector>

#include "operators.hpp"

namespace kiwitorch {

template <typename T = float> class Tensor {
  public:
    // Constructors
    Tensor() = default;
    explicit Tensor(const std::vector<size_t>& shape);
    Tensor(const std::vector<size_t>& shape, const std::vector<T>& data);

    // Factory methods
    static Tensor<T> zeros(const std::vector<size_t>& shape);
    static Tensor<T> ones(const std::vector<size_t>& shape);
    static Tensor<T> rand(const std::vector<size_t>& shape);

    // Basic operations
    Tensor<T> operator+(const Tensor<T>& other) const;
    Tensor<T> operator-(const Tensor<T>& other) const;
    Tensor<T> operator*(const Tensor<T>& other) const;
    Tensor<T> operator/(const Tensor<T>& other) const;

    // Neural network operations
    Tensor<T> matmul(const Tensor<T>& other) const;
    Tensor<T> conv2d(const Tensor<T>& kernel) const;
    Tensor<T> relu() const;
    Tensor<T> sigmoid() const;
    Tensor<T> tanh() const;

    // Shape operations
    Tensor<T> view(const std::vector<size_t>& new_shape) const;
    Tensor<T> transpose(size_t dim1, size_t dim2) const;
    Tensor<T> sum(int64_t dim = -1, bool keepdim = false) const;

    // Gradient operations
    void backward();

    bool requires_grad() const { return requires_grad_; }

    void set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }

    const Tensor<T>& grad() const { return *grad_; }

    // Accessors
    T item() const; // For single-element tensors

    const std::vector<size_t>& shape() const { return shape_; }

    size_t size(size_t dim) const { return shape_[dim]; }

    const std::vector<T>& data() const { return data_; }
  private:
    struct AutogradContext {
        std::function<void(const Tensor<T>&)> backward_fn;
        std::vector<Tensor<T>> saved_tensors;
    };

    std::vector<T> data_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    bool requires_grad_ = false;
    std::shared_ptr<Tensor<T>> grad_;
    std::shared_ptr<AutogradContext> autograd_context_;

    void compute_strides();
    size_t compute_index(const std::vector<size_t>& indices) const;
};

} // namespace kiwitorch
