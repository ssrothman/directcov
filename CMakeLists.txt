cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

set(SOURCES
    src/main.cpp
)

pybind11_add_module(_core MODULE ${SOURCES})
target_include_directories(_core PRIVATE include)

#link with Eigen3
find_package(Eigen3 CONFIG REQUIRED)
target_include_directories(_core PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(_core PRIVATE Eigen3::Eigen)

# link with openmp
find_package(OpenMP)
if(OpenMP_FOUND)
    target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP not found. Build will proceed without OpenMP support (single-threaded).")
endif()

# link with mkl
set(MKL_INTERFACE lp64) # need to use lp64 interface for MKL - NOT ilp64 which is incompatible with Eigen
find_package(MKL CONFIG)
if(MKL_FOUND)
    target_compile_options(_core PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
    target_include_directories(_core PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(_core PUBLIC $<LINK_ONLY:MKL::MKL>)
    target_compile_definitions(_core PUBLIC EIGEN_USE_MKL_ALL=1) # use all MKL features
else()
    message(WARNING "Intel MKL not found. Trying BLAS/LAPACKE.")

    set(HAS_FALLBACK_MATH OFF)

    find_package(BLAS)
    if(BLAS_FOUND)
        if(TARGET BLAS::BLAS)
            target_link_libraries(_core PRIVATE BLAS::BLAS)
        else()
            target_link_libraries(_core PRIVATE ${BLAS_LIBRARIES})
        endif()
        set(HAS_FALLBACK_MATH ON)
        target_compile_definitions(_core PRIVATE EIGEN_USE_BLAS=1)
        message(STATUS "Using BLAS")
    else()
        message(WARNING "BLAS not found.")
    endif()

    find_path(LAPACKE_INCLUDE_DIR NAMES lapacke.h)
    find_library(LAPACKE_LIBRARY NAMES lapacke)
    if(LAPACKE_LIBRARY)
        target_link_libraries(_core PRIVATE ${LAPACKE_LIBRARY})
        if(LAPACKE_INCLUDE_DIR)
            target_include_directories(_core PRIVATE ${LAPACKE_INCLUDE_DIR})
        endif()
        target_compile_definitions(_core PRIVATE EIGEN_USE_LAPACKE=1)
        set(HAS_FALLBACK_MATH ON)
        message(STATUS "Using LAPACKE")
    else()
        message(WARNING "LAPACKE not found.")
    endif()

    if(NOT HAS_FALLBACK_MATH)
        message(WARNING "No MKL/BLAS/LAPACKE backend found. Build will proceed with Eigen defaults; performance may be reduced.")
    endif()
endif()

# specify max compiler optimizations
target_compile_options(_core PRIVATE -O3)
target_compile_options(_core PRIVATE -Wall -Wextra -Wpedantic)

# specify eigen defines
target_compile_definitions(_core PRIVATE $<$<CONFIG:Release>:EIGEN_NO_DEBUG=1>)

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
if(ipo_supported)
    message(STATUS "IPO/LTO supported: enabling for release builds")
    set_property(TARGET _core PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
else()
    message(WARNING "IPO/LTO not supported: ${ipo_error}")
endif()

install(TARGETS _core DESTINATION ${SKBUILD_PROJECT_NAME})
