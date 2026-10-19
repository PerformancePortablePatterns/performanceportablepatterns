function(PerformancePortablePatterns_add_kokkos_test ROOT_NAME)
  cmake_parse_arguments(PARSE "" "" "HEADERS;INCLUDE_HEADERS;ARGS" ${ARGN})
  verify_empty(PerfomancePortablePatterns_add_kokkos_test
               ${PARSE_UNPARSED_ARGUMENTS})

  set(Test_harness
      "${CMAKE_SOURCE_DIR}/patterns/utils/kokkos/kokkos_UnitTest_harness.cpp")

  foreach(header IN LISTS PARSE_HEADERS)
    list(APPEND header_includes "#include \"${header}\"\n")
  endforeach()

  foreach(_device SERIAL PTHREAD OPENMP CUDA HIP)
    if(Kokkos_ENABLE_${_device})
      string(TOUPPER ${_device} _uppercase_device)
      set(_dir ${CMAKE_CURRENT_BINARY_DIR}/${_uppercase_device})
      file(MAKE_DIRECTORY ${_dir})
      set(TEST_NAME
          UnitTest_PerformancePortablePatterns_${ROOT_NAME}_${_uppercase_device}
      )
      set(_file ${_dir}/${TEST_NAME}.cpp)
      set(Cathegory_file
          ${CMAKE_SOURCE_DIR}/patterns/utils/kokkos/Test${_device}_Category.hpp)
      string(REPLACE ";" " " header_includes "${header_includes}")
      file(WRITE ${_file} "#include \"${Cathegory_file}\"\n
        ${header_includes}\n")

      # set_source_files_properties(${_file} PROPERTIES LANGUAGE
      # ${KOKKOS_COMPILE_LANGUAGE})
      add_executable(${TEST_NAME} ${_file} ${Test_harness})
      target_sources(
        ${TEST_NAME}
        PRIVATE
          FILE_SET
          HEADERS
          BASE_DIRS
          ${CMAKE_SOURCE_DIR}/patterns/utils/kokkos;${CMAKE_CURRENT_SOURCE_DIR}
          FILES
          ${PARSE_INCLUDE_HEADERS}
          ${PARSE_HEADERS}
          ${Cathegory_file})
      target_link_libraries(${TEST_NAME} PRIVATE Kokkos::kokkos GTest::gtest)
      # We noticed problems with -fvisibility=hidden for inline static variables
      # if Kokkos was built as shared library.
      if(BUILD_SHARED_LIBS AND NOT ${TEST_NAME}_DISABLE)
        set_property(TARGET ${TEST_NAME} PROPERTY VISIBILITY_INLINES_HIDDEN ON)
        set_property(TARGET ${TEST_NAME} PROPERTY CXX_VISIBILITY_PRESET hidden)
      endif()
      add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endif()
  endforeach()
endfunction()
