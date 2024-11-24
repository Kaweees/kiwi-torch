add_executable(example1 example1.cpp)
target_link_libraries(example1 PRIVATE ${PROJECT_NAME})

# Setting our output directory
set_target_properties(example1 PROPERTIES 
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/target/release/"
  OUTPUT_NAME example1
)
