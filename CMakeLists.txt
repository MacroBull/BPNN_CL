cmake_minimum_required(VERSION 2.6)
project(nncl)

add_executable(nncl main.cpp nn_mlp_singlehiddenlayer.cpp)


ADD_DEFINITIONS(-std=c++11 -g)
# ADD_DEFINITIONS(-std=c++11 -O2)
# ADD_DEFINITIONS(-D USE_LOGISTIC)


FIND_PACKAGE(OpenCL REQUIRED)
find_library(OpenCL_LIBS OpenCL)
INCLUDE_DIRECTORIES(${OPENCL_INCLUDE_DIR})


TARGET_LINK_LIBRARIES(nncl ${OpenCL_LIBS})


configure_file(nn_mlp_singlehiddenlayer.cl ${CMAKE_CURRENT_BINARY_DIR}/nn_mlp_singlehiddenlayer.cl COPYONLY)
configure_file(sigmoid.cl ${CMAKE_CURRENT_BINARY_DIR}/sigmoid.cl COPYONLY)

# install(TARGETS nncl RUNTIME DESTINATION bin)
