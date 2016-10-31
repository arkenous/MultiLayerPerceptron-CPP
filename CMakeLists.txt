cmake_minimum_required(VERSION 3.6)
project(MultiLayerPerceptron_CPP)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")
set(CMAKE_C_FLAGS_RELEASE "-Ofast")
set(CMAKE_VERBOSE_MAKEFILE ON)

set(SOURCE_FILES main.cpp Neuron.cpp Neuron.h MultiLayerPerceptron.cpp MultiLayerPerceptron.h)
add_executable(MultiLayerPerceptron_CPP ${SOURCE_FILES})
