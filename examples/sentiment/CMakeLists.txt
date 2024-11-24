add_executable(train_sentiment train_sentiment.cpp)
target_link_libraries(train_sentiment PRIVATE ${PROJECT_NAME})

# # Copy sentiment data files
# file(COPY ${PROJECT_SOURCE_DIR}/data/sst2 DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data)

# Setting our output directory
set_target_properties(train_sentiment PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/target/release/"
  OUTPUT_NAME train_sentiment
)
