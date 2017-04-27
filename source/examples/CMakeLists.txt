configure_file(
  sampleFiles.h.in
  sampleFiles.h
)

if(CUDA_FOUND)

  set(SAMPLE_SRCS_CUDA
    exampleROF.cu
    exampleSegmentation.cu
  )

  foreach(SAMPLE_FILE ${SAMPLE_SRCS_CUDA})

    get_filename_component(FILE_NAME ${SAMPLE_FILE} NAME_WE)
    include_directories(${INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

    #TODO: use maximum compatibility fallback, dont use fpic on windows systems
    set(CUDA_NVCC_FLAGS --std=c++11 ${FLEXBOX_NVCC_FLAGS})
    cuda_add_executable(${FILE_NAME} ${SAMPLE_FILE})
    target_link_libraries(${FILE_NAME} ${X11_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

    install(TARGETS ${FILE_NAME} DESTINATION ${CMAKE_BINARY_DIR}/bin)

  endforeach()

else()

  set(SAMPLE_SRCS_CPP
    exampleROF.cpp
    exampleSegmentation.cpp
  )
  foreach(SAMPLE_FILE ${SAMPLE_SRCS_CPP})

    get_filename_component(FILE_NAME ${SAMPLE_FILE} NAME_WE)
    add_executable(${FILE_NAME} ${SAMPLE_FILE})
    target_link_libraries(${FILE_NAME} PUBLIC flexBoxLib ${X11_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
    target_include_directories(${FILE_NAME} PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

    install(TARGETS ${FILE_NAME} DESTINATION ${CMAKE_BINARY_DIR}/bin)

  endforeach()

endif(CUDA_FOUND)
