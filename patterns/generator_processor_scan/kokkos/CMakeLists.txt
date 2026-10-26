include(${CMAKE_SOURCE_DIR}/patterns/utils/kokkos/kokkos_setup.cmake)

performanceportablepatterns_add_kokkos_test(
  generator_processor
  HEADERS
  UnitTest_Kokkos_generator_processor.hpp
  INCLUDE_HEADERS
  ppp_generators.hpp
  ppp_processors.hpp
  ppp_count_offset_fill.hpp
  ppp_generator_processor_scan_offset.hpp
  ppp_generator_processor_scan.hpp
  ppp_ref_cub.hpp)

# should be gperf
add_executable(generator_processor_PerfTest
               PerfTest_Kokkos_generator_processor.cpp)
target_sources(
  generator_processor_PerfTest
  PRIVATE FILE_SET
          HEADERS
          FILES
          ppp_generators.hpp
          ppp_processors.hpp
          ppp_count_offset_fill.hpp
          ppp_generator_processor_scan_offset.hpp
          ppp_generator_processor_scan.hpp
          ppp_ref_cub)
target_link_libraries(generator_processor_PerfTest Kokkos::kokkos)
set_target_properties(
  generator_processor_PerfTest PROPERTIES CXX_CLANG_TIDY
                                          "clang-tidy;-warnings-as-errors=*")
