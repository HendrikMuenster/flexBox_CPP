  set(TEST_FILES
    flexBoxTest.cpp
  )
  foreach(TEST_FILE ${TEST_FILES})

    get_filename_component(FILE_NAME ${TEST_FILE} NAME_WE)
    add_executable(${FILE_NAME} ${TEST_FILE})
    target_link_libraries(${FILE_NAME} PUBLIC flexBoxLib ${X11_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
    target_include_directories(${FILE_NAME} PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

    install(TARGETS ${FILE_NAME} DESTINATION ${CMAKE_BINARY_DIR}/bin)

  endforeach()
