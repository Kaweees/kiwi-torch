#include <stddef.h>
#include <stdio.h>

#include <iostream>

#include "../include/operators.hpp"

bool more_than_4(float x) { return x > 4; }

//------------------------------------------------------------------------------------
// Program main entry point
//------------------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  std::cout << "combine3 with add: " << combine3(add, 1, 3, 5) << std::endl;
  std::cout << "combine3 with mul: " << combine3(mul, 1, 3, 5) << std::endl;

  auto add3 = combine3(add);
  std::cout << "add3: " << add3(1, 3, 5) << std::endl;

  auto filter_for_more_than_4 = filter(more_than_4);
  std::vector<float> numbers = {1, 10, 3, 5};
  auto filtered = filter_for_more_than_4(numbers);

  std::cout << "Filtered numbers: ";
  for (float num : filtered) {
    std::cout << num << " ";
  }
  std::cout << std::endl;

  return 0;
}