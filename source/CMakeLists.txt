cmake_minimum_required(VERSION 3.0)
project(FlexBox)
set (FlexBox_VERSION_MAJOR 1)
set (FlexBox_VERSION_MINOR 4)

##########################
#        OPTIONS         #
##########################

option(USE_OPENMP "Use OpenMP" OFF)
option(USE_CUDA "Use CUDA" OFF)
option(BUILD_MATLAB "Build Matlab (mex) interface" ON)
option(BUILD_EXAMPLES "Build C++/CUDA examples" ON)
option(BUILD_TESTS "Run tests using CATCH framework" ON)

##########################
#     FILES/INCUDES      #
##########################

#includes
set(INCLUDE_DIRS
  ${CMAKE_SOURCE_DIR}/.
  ${CMAKE_SOURCE_DIR}/data
  ${CMAKE_SOURCE_DIR}/operator
  ${CMAKE_SOURCE_DIR}/prox
  ${CMAKE_SOURCE_DIR}/solver
  ${CMAKE_SOURCE_DIR}/term
)


#data
set(SOURCES_IDE_DATA
  data/flexBoxData.h
  data/flexBoxDataCPU.h
  data/flexBoxDataGPU.h
)

#operator
set(SOURCES_IDE_OP
  operator/flexDiagonalOperator.h
  operator/flexGradientOperator.h
  operator/flexIdentityOperator.h
  operator/flexLinearOperator.h
  operator/flexMatrix.h
  operator/flexFullMatrix.h
  operator/flexMatrixLogical.h
  operator/flexMatrixGPU.h
  operator/flexSuperpixelOperator.h
  operator/flexZeroOperator.h
  operator/flexConcatOperator.h
)

#prox
set(SOURCES_IDE_PROX
  prox/flexProx.h
  prox/flexProxDualBoxConstraint.h
  prox/flexProxDualDataKL.h
  prox/flexProxDualDataL1.h
  prox/flexProxDualDataL2.h
  prox/flexProxDualDataLInf.h
  prox/flexProxDualFrobenius.h
  prox/flexProxDualHuber.h
  prox/flexProxDualL1Aniso.h
  prox/flexProxDualL1Iso.h
  prox/flexProxDualL2.h
  prox/flexProxDualLabeling.h
  prox/flexProxDualLInf.h
)

#solver
set(SOURCES_IDE_SOLV
  solver/flexSolver.h
  solver/flexSolverPrimalDual.h
  solver/flexSolverPrimalDualCuda.h
)

#term
set(SOURCES_IDE_TERM
  term/flexTerm.h
)

#matlab
set(SOURCES_IDE_MATLAB
  wrappers/matlab/functions.cpp
  wrappers/matlab/functions.cu
)

#misc
set(SOURCES_IDE_MISC
  flexBox.h
  tools.h
)

#group for visual studio
source_group("data" FILES ${SOURCES_IDE_DATA})
source_group("operator" FILES ${SOURCES_IDE_OP})
source_group("prox" FILES ${SOURCES_IDE_PROX})
source_group("solver" FILES ${SOURCES_IDE_SOLV})
source_group("term" FILES ${SOURCES_IDE_TERM})
source_group("matlab" FILES ${SOURCES_IDE_MATLAB})
source_group("misc" FILES ${SOURCES_IDE_MISC})

#combine
set(SOURCES_IDE ${SOURCES_IDE_DATA} ${SOURCES_IDE_OP} ${SOURCES_IDE_PROX} ${SOURCES_IDE_SOLV} ${SOURCES_IDE_TERM} ${SOURCES_IDE_MATLAB} ${SOURCES_IDE_MISC})

##########################
#   END FILES/INCUDES    #
##########################

if (CMAKE_VERSION VERSION_LESS "3.1")
  include(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-std=c++11" CXX11_SUPPORT)
  CHECK_CXX_COMPILER_FLAG("-std=c++0x" CXX0x_SUPPORT)
  if(CXX11_SUPPORT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  elseif(CXX0x_SUPPORT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
  else()
        message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Aborting...")
  endif()
else()
  set(CMAKE_CXX_STANDARD 11)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message (STATUS  "No build type specified. Default is 'Release'.")
  set (CMAKE_BUILD_TYPE Release CACHE STRING "The type of build" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()

set (WRAPPERS_DIR "wrappers")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${DEBUG_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${DEBUG_FLAGS}")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${RELEASE_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${RELEASE_FLAGS}")

if(USE_OPENMP)
  find_package(OpenMP)

  if(OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  else()
    message (STATUS "Couldn't find OpenMP. Building without...")
  endif(OPENMP_FOUND)

endif(USE_OPENMP)

#include own modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules)


if(BUILD_MATLAB)
  find_package(Matlab COMPONENTS MX_LIBRARY)

  if(NOT Matlab_FOUND)
    message (STATUS "Couldn't find Matlab or its required components MX. Building without...")
  endif()

endif(BUILD_MATLAB)


if(USE_CUDA)
  find_package(CUDA)

  if(NOT CUDA_FOUND)
    message (STATUS "Couldn't find CUDA. Building without...")
  else()
    if(CUDA_VERSION VERSION_LESS "3.2")
      message (STATUS "At least CUDA 3.2 is needed. Only found: " ${CUDA_VERSION} ". Building without...")
      set(CUDA_FOUND OFF)
    endif()

    include(${CMAKE_CURRENT_LIST_DIR}/cmake/FlexBoxCudaFlags.cmake)
    FLEXBOX_CUDA_SELECT_NVCC_ARCH_FLAGS(FLEXBOX_NVCC_FLAGS)

  endif()
endif(USE_CUDA)

if(BUILD_EXAMPLES AND UNIX)
  find_package(X11 REQUIRED)
  find_package(Threads REQUIRED)
endif(BUILD_EXAMPLES AND UNIX)


# flexBox template library
add_library(flexBoxLib INTERFACE)
target_include_directories(flexBoxLib INTERFACE ${INCLUDE_DIRS})

add_custom_target(IDE SOURCES ${SOURCES_IDE})




# Matlab target
if(Matlab_FOUND)
  add_subdirectory(${WRAPPERS_DIR}/matlab)
endif(Matlab_FOUND)

# examples targets
if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif(BUILD_EXAMPLES)

# test targets
if(BUILD_TESTS)
  add_subdirectory(tests)
endif(BUILD_TESTS)
