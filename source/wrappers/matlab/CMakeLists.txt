#functions.{cpp | cu} is matlab specific (contains mex entry functions)
if(CUDA_FOUND)

  #must use include_directories instead of target_include_directories
  #known issue in cmakes CUDA-Module
  include_directories(${Matlab_INCLUDE_DIRS})
  include_directories(${INCLUDE_DIRS})

  set(CUDA_NVCC_FLAGS --std=c++11 -Xcompiler -fPIC ${FLEXBOX_NVCC_FLAGS})
  cuda_compile(cuda_generated functions.cu SHARED)
  matlab_add_mex(NAME Matlab
        SRC ${cuda_generated}
        LINK_TO flexBoxLib ${Matlab_MEX_LIBRARY} ${Matlab_MX_LIBRARY} ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${CUDA_CUFFT_LIBRARIES}
        OUTPUT_NAME flexBoxCPP
  )

  if(MSVC)
    configure_file(project.def ${CMAKE_CURRENT_BINARY_DIR}/project.def COPYONLY)
    get_target_property(existing_link_flags Matlab LINK_FLAGS)
    if (existing_link_flags STREQUAL "existing_link_flags-NOTFOUND")
      set(existing_link_flags "")
    endif()

    set(new_link_flags "${existing_link_flags} /DEF:project.def")
    set_target_properties(Matlab PROPERTIES LINK_FLAGS ${new_link_flags})
  endif()

  set_target_properties(Matlab PROPERTIES LINKER_LANGUAGE CXX)

else()
  matlab_add_mex (NAME Matlab
    SRC functions.cpp
    LINK_TO flexBoxLib ${Matlab_MEX_LIBRARY} ${Matlab_MX_LIBRARY}
    OUTPUT_NAME flexBoxCPP
  )
endif(CUDA_FOUND)

#copy mex
install(TARGETS Matlab DESTINATION ${CMAKE_BINARY_DIR}/bin)
