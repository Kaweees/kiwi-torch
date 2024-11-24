add_executable(train_mnist train_mnist.cpp)
target_link_libraries(train_mnist PRIVATE ${PROJECT_NAME})

# # Copy MNIST data files
# file(COPY ${PROJECT_SOURCE_DIR}/data/mnist DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data)

# Setting our output directory
set_target_properties(train_mnist PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/target/release/"
  OUTPUT_NAME train_mnist
)
