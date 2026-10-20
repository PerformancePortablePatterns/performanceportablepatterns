cmake_minimum_required(VERSION 3.25)

project(
  PerformancePortablePatterns
  VERSION 0.1.0
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
enable_testing()

find_package(GTest 1.10 CONFIG)
if(GTest_FOUND)
  message(STATUS "Gtest found")
else()
  message(STATUS "Gtest not found: fetching")
  include(FetchContent)
  FetchContent_Declare(
    GTest
    URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
  )
  FetchContent_MakeAvailable(Gtest)
endif()

add_subdirectory(patterns)
