enable_testing()

set(
  TESTS
  test_operators
)

foreach (name IN LISTS TESTS)
  message(STATUS "Building test: ${name}")
  add_executable("${name}" "${name}.cpp")
  target_link_libraries(
    "${name}"
    PRIVATE
    ${PROJECT_NAME}
    GTest::gtest_main
  )
endforeach ()

