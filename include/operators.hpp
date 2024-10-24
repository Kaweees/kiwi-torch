#pragma once

#include <functional>
#include <vector>

namespace kiwitorch {
class Scalar {
  public:
  // Multiplies two numbers
  static float mul(float x, float y);
  // Returns the input unchanged
  static float id(float x);
  // Adds two numbers
  static float add(float x, float y);
  // Negates a number
  static float neg(float x);
  // Checks if one number is less than another
  static bool lt(float x, float y);
  // Checks if two numbers are equal
  static bool eq(float x, float y);
  // Returns the larger of two numbers
  static float max(float x, float y);
  // Checks if two numbers are close in value
  static bool is_close(float x, float y);
  // Calculates the sigmoid function
  static float sigmoid(float x);
  // Applies the ReLU activation function
  static float relu(float x);
  // Calculates the natural logarithm
  static float log(float x);
  // Calculates the exponential function
  static float exp(float x);
  // Calculates the reciprocal
  static float inv(float x);
  // Computes the derivative of log times a second arg
  static float log_back(float x, float y);
  // Computes the derivative of reciprocal times a second arg
  static float inv_back(float x, float y);
  // Computes the derivative of ReLU times a second arg
  static float relu_back(float x, float y);

  static float combine3(
      std::function<float(float, float)> fn, float a, float b, float c);

  static std::function<float(float, float, float)> combine3(
      std::function<float(float, float)> fn);

  static std::function<std::vector<float>(const std::vector<float>&)> filter(
      std::function<bool(float)> fn);
};
}  // namespace kiwitorch
