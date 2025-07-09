cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.h")

pybind11_add_module(_core MODULE ${SOURCES} ${HEADERS})
target_include_directories(_core PRIVATE include)

#link with Eigen3
find_package(Eigen3 CONFIG REQUIRED)
target_include_directories(_core PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(_core PRIVATE Eigen3::Eigen)

# link with openmp
find_package(OpenMP REQUIRED)
target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)

# link with mkl
set(MKL_INTERFACE lp64) # need to use lp64 interface for MKL - NOT ilp64 which is incompatible with Eigen
find_package(MKL CONFIG)
target_compile_options(_core PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(_core PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(_core PUBLIC $<LINK_ONLY:MKL::MKL>)
target_compile_definitions(_core PUBLIC EIGEN_USE_MKL_ALL=1) # use all MKL features

# specify max compiler optimizations
target_compile_options(_core PRIVATE -O3 -march=native -mtune=native)

#specify eigen defines
target_compile_definitions(_core PRIVATE EIGEN_NO_DEBUG=1)

install(TARGETS _core DESTINATION ${SKBUILD_PROJECT_NAME})
