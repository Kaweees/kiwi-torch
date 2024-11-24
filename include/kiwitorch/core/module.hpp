#pragma once
#include <memory>
#include <string>
#include <unordered_map>

#include "tensor.hpp"

namespace kiwitorch {

class Module {
  public:
  Module() = default;
  virtual ~Module() = default;

  // Prevent copying, allow moving
  Module(const Module&) = delete;
  Module& operator=(const Module&) = delete;
  Module(Module&&) = default;
  Module& operator=(Module&&) = default;

  // Core functionality
  virtual Tensor<float> forward(const Tensor<float>& x) = 0;

  // Parameter management
  void add_parameter(const std::string& name, const Tensor<float>& param);
  void add_parameter(const std::string& name, Tensor<float>&& param);
  std::unordered_map<std::string, Tensor<float>> named_parameters() const;

  // Module management
  void add_module(const std::string& name, std::shared_ptr<Module> module);
  const std::unordered_map<std::string, std::shared_ptr<Module>>& modules()
      const;

  // Training modes
  void train(bool mode = true) { training_ = mode; }
  void eval() { training_ = false; }
  bool is_training() const { return training_; }

  // Zero gradients
  void zero_grad();

  protected:
  std::unordered_map<std::string, Tensor<float>> parameters_;
  std::unordered_map<std::string, std::shared_ptr<Module>> modules_;
  bool training_ = true;
};

// Parameter wrapper class
class Parameter {
  public:
  explicit Parameter(const Tensor<float>& tensor);
  explicit Parameter(Tensor<float>&& tensor);

  Tensor<float>& value() { return tensor_; }
  const Tensor<float>& value() const { return tensor_; }

  private:
  Tensor<float> tensor_;
};

}  // namespace kiwitorch
