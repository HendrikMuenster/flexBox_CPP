  set(TEST_FILES
    testMain.cpp
    testOperators.cpp
  )

add_executable(Tests ${TEST_FILES})
target_link_libraries(Tests PUBLIC flexBoxLib)
target_include_directories(Tests PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS Tests DESTINATION ${CMAKE_BINARY_DIR}/bin)
