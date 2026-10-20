include(FetchContent)
FetchContent_Declare(
  Kokkos URL https://github.com/kokkos/kokkos/archive/refs/tags/5.2.1.tar.gz)
FetchContent_MakeAvailable(Kokkos)

include(${CMAKE_CURRENT_LIST_DIR}/kokkos_test.cmake)
