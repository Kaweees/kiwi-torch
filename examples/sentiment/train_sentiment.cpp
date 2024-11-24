#include <iostream>
#include <kiwitorch/kiwitorch.hpp>
using namespace kiwitorch;

int main() {
  std::cout << "Sum: " << Scalar::add(2, 3) << std::endl;
  return 0;
}
