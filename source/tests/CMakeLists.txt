if(CUDA_FOUND)
    set(TEST_FILES
       testProx.cu
       testOperators.cu
    )
    foreach(TEST_FILE ${TEST_FILES})
        get_filename_component(FILE_NAME ${TEST_FILE} NAME_WE)
        include_directories(${INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
        set(CUDA_NVCC_FLAGS --std=c++11 ${FLEXBOX_NVCC_FLAGS})
        cuda_add_executable(${FILE_NAME} testMain.cu ${TEST_FILE})
        target_link_libraries(${FILE_NAME} flexBoxLib)
        install(TARGETS ${FILE_NAME} DESTINATION ${CMAKE_BINARY_DIR}/bin)
    endforeach()
else()
    set(TEST_FILES
       testProx.cpp
       testOperators.cpp
    )
    foreach(TEST_FILE ${TEST_FILES})
        get_filename_component(FILE_NAME ${TEST_FILE} NAME_WE)
        add_executable(${FILE_NAME} testMain.cpp ${TEST_FILE})
        target_link_libraries(${FILE_NAME} PUBLIC flexBoxLib)
        target_include_directories(${FILE_NAME} PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
        install(TARGETS ${FILE_NAME} DESTINATION ${CMAKE_BINARY_DIR}/bin)
    endforeach()
endif(CUDA_FOUND)
