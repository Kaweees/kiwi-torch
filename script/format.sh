#!/bin/bash

# Format script called by the CI
# Usage:
#    format.sh format

#
#  Private Impl
#

format() {
  clang-format -style=file -i $(find . -name "*.c" -o -name "*.h" -o -name "*.cpp" -o -name "*.hpp" -o -name "*.cc" -o -name "*.cxx" -o -name "*.cu")
  echo "Formatted all files"
}

# Main script logic
case "$1" in
  format)
    format
    ;;
  *)
    echo "Usage: $0 {format}"
    exit 1
    ;;
esac
