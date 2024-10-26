#include <stddef.h>
#include <stdio.h>

#include <iostream>

#include "../include/operators.hpp"

//------------------------------------------------------------------------------------
// Program main entry point
//------------------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  std::vector<double> numbers = {1, 3, 5};
  std::cout << "numbers: ";
  for (double num : numbers) {
    std::cout << num << " ";
  }
  std::cout << std::endl;
  std::cout << "sum: " << kiwitorch::Scalar::sum(numbers) << std::endl;
  std::cout << "prod: " << kiwitorch::Scalar::prod(numbers) << std::endl;
  return 0;
}