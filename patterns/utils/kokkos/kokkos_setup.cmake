include(FetchContent)

find_package(Kokkos 5.2.1)
if(Kokkos_FOUND)
  message(STATUS "Kokkos found")
else()
  message(STATUS "Kokkos not found: fetching")
  FetchContent_Declare(
    Kokkos URL https://github.com/kokkos/kokkos/archive/refs/tags/5.2.1.tar.gz
               SYSTEM)
  FetchContent_MakeAvailable(Kokkos)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/kokkos_test.cmake)
