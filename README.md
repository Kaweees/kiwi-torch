<!--
*** This readme was adapted from Best-README-Template.
  https://github.com/othneildrew/Best-README-Template
-->

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<div align="left">

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]

</div>

<a href="https://github.com/Kaweees/kiwitorch">
  <img alt="C++ Logo" src="assets/img/cpp.svg" align="right" width="150">
</a>

<div align="left">
  <h1><em><a href="https://github.com/Kaweees/kiwitorch">~kiwitorch</a></em></h1>
</div>

<!-- ABOUT THE PROJECT -->

A header-only implementation of pytorch written in C++13 and accelerated with CUDA/Apple Metal.

### Built With

[![C++][C++-shield]][C++-url]
[![CUDA][CUDA-shield]][CUDA-url]
[![Apple][Apple-shield]][Apple-url]
[![GitHub Actions][github-actions-shield]][github-actions-url]

<!-- PROJECT PREVIEW -->
## Preview

<p align="center">
  <img src="assets/img/demo.mp4"
  width = "80%"
  alt = "Video demonstration"
  />
</p>

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

Before attempting to build this project, make sure you have [GNU Make](https://www.gnu.org/software/make/), and [CMake](https://cmake.org) installed on your machine.

Additionally, if you wish to utilize the GPU acceleration features, you will need to have [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) or [Apple Metal](https://developer.apple.com/metal/cpp/) installed on your machine.

### Installation

To get a local copy of the project up and running on your machine, follow these simple steps:

1. Clone the project repository

   ```sh
   git clone https://github.com/Kaweees/kiwitorch.git
   cd kiwitorch
   ```

2. Create a fresh build directory and navigate to it

   ```sh
   rm -rf build
   mkdir build
   cd build
   ```

3. Generate build files using CMake

   ```sh
   cmake -S .. -B . -DCMAKE_BUILD_TYPE=Debug
   ```

4. Build the entire project

   ```sh
   cmake --build .
   ```

### Building and Running Tests

1. Build only the tests

   ```sh
   cmake --build . --target tests
   ```

2. Run all tests

   ```sh
   ctest --output-on-failure
   ```

3. Run a specific test suite

   ```sh
   ./tests/test_suite_name
   ```

### Building and Running Examples

1. Build only the examples

   ```sh
   cmake --build . --target examples
   ```

2. Run a specific example

   ```sh
   ./examples/example_name
   ```

<!-- PROJECT FILE STRUCTURE -->
## Project Structure

```sh
graphics/
├── .github/                       - GitHub Actions CI/CD workflows
├── include/                       - project header files
├── src/                           - project source files
│   └── main.c                     - Entry point, main function
├── CMakeLists.txt                 - CMake build script
├── LICENSE                        - project license
└── README.md                      - you are here
```

``` bash
kiwitorch/
include/kiwitorch/
├── core/               # Core functionality
│   ├── tensor.hpp
│   ├── scalar.hpp
│   ├── operators.hpp
│   └── module.hpp
├── nn/                 # Neural network components
│   ├── activations.hpp
│   └── layers/        # Layer implementations
│       ├── linear.hpp
│       └── conv2d.hpp
├── optim/             # Optimization algorithms
│   └── optimizer.hpp
├── data/              # Data handling
│   ├── dataset.hpp
│   └── dataloader.hpp
├── utils/             # Core utilities
│   └── graph_builder.hpp
├── io/                # Input/Output operations
│   └── visualization.hpp
└── kiwitorch.hpp      # Main include file
```

## License

The source code for kiwitorch is distributed under the terms of the GNU General Public License v3.0, as I firmly believe that collaborating on free and open-source software fosters innovations that mutually and equitably beneficial to both collaborators and users alike. See [`LICENSE`](./LICENSE) for details and more information.

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/Kaweees/kiwitorch.svg?style=for-the-badge
[contributors-url]: https://github.com/Kaweees/kiwitorch/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Kaweees/kiwitorch.svg?style=for-the-badge
[forks-url]: https://github.com/Kaweees/kiwitorch/network/members
[stars-shield]: https://img.shields.io/github/stars/Kaweees/kiwitorch.svg?style=for-the-badge
[stars-url]: https://github.com/Kaweees/kiwitorch/stargazers

<!-- MARKDOWN SHIELD BAGDES & LINKS -->
<!-- https://github.com/Ileriayo/markdown-badges -->
[C++-shield]: https://img.shields.io/badge/C++-%23008080.svg?style=for-the-badge&logo=c%2B%2B&logoColor=004482&labelColor=222222&color=004482
[C++-url]: https://isocpp.org/
[CUDA-shield]: https://img.shields.io/badge/cuda-%23008080.svg?style=for-the-badge&logo=nVIDIA&logoColor=76B900&labelColor=222222&color=76B900
[CUDA-url]: https://developer.nvidia.com/cuda-zone
[Apple-shield]: https://img.shields.io/badge/metal-%23008080.svg?style=for-the-badge&logo=apple&logoColor=white&labelColor=222222&color=white
[Apple-url]: https://developer.apple.com/metal/
[github-actions-shield]: https://img.shields.io/badge/github%20actions-%232671E5.svg?style=for-the-badge&logo=githubactions&logoColor=2671E5&labelColor=222222&color=2671E5
[github-actions-url]: https://github.com/features/actions
