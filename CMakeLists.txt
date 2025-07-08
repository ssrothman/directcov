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

install(TARGETS _core DESTINATION ${SKBUILD_PROJECT_NAME})
