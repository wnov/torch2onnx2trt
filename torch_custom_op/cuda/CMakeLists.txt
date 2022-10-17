cmake_minimum_required(VERSION 3.10)
project(customTorchOp)

set(CMAKE_CXX_STANDARD 11)

# set(TORCH_DIR /home/wn/applications/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

add_library(customTorchOp SHARED ${PROJECT_SOURCE_DIR}/libcustomOp/customOp.cpp)
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/install)
install(TARGETS customTorchOp LIBRARY DESTINATION lib)
install(FILE ${PROJECT_SOURCE_DIR}/libcustmOp/customOp.h DESTINATION include)
