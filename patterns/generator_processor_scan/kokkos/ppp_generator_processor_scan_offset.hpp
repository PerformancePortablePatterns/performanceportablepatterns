#ifndef PPP_GENERATOR_PROCESSOR_SCAN_OFFSET_HPP
#define PPP_GENERATOR_PROCESSOR_SCAN_OFFSET_HPP

namespace PPP {
template <typename ExecutionSpace, typename GeneratorType,
          typename ProcessorType, typename OffsetType>
void generator_processor_scan_offset(ExecutionSpace const &exec_space,
                                     GeneratorType const &generator,
                                     ProcessorType &processor,
                                     OffsetType &offset) {
  using memory_space         = typename ExecutionSpace::memory_space;
  using IndexingType         = typename GeneratorType::IndexingType;
  auto generator_start_index = generator.start_index();
  auto generator_end_index   = generator.end_index();

  if (offset.extent(0) != generator_end_index - generator_start_index)
    Kokkos::abort(
        "offsets must have extent of generator.end()-generator.start() ");

  // track highest processed generator index and highest processed processor
  // index across passes
  // TODO get rid of this with a custom reducer
  Kokkos::View<IndexingType[3], memory_space> failed_index("failed_index");

  IndexingType number_of_traversed_elements_this_iteration;

  Kokkos::parallel_scan(
      "Generator_processor_scan first pass",
      Kokkos::RangePolicy(exec_space, generator_start_index,
                          generator_end_index),
      KOKKOS_LAMBDA(IndexingType current_generator_index,
                    IndexingType & current_processor_index, bool is_final) {
        auto num_elements_at_current_generator_index =
            generator.number_of_elements_for_index(current_generator_index);

        if (is_final) {
          offset[current_generator_index - generator_start_index] =
              current_processor_index;
          if (num_elements_at_current_generator_index != 0) {
            bool can_process_all_elements_at_current_generator_index =
                processor.can_insert_element_at_index(
                    current_processor_index +
                    num_elements_at_current_generator_index);
            if (can_process_all_elements_at_current_generator_index) {
              Kokkos::atomic_max(&failed_index(0), current_generator_index + 1);
              Kokkos::atomic_max(&failed_index(1),
                                 current_processor_index +
                                     num_elements_at_current_generator_index);
              // if (failed_index(0) < (current_generator_index + 1)) {
              // failed_index(0) = current_generator_index + 1;
              // failed_index(1) = current_processor_index +
              // num_elements_at_current_generator_index;
              // }
              for (auto element = 0;
                   element < num_elements_at_current_generator_index; ++element)
                processor(current_processor_index + element,
                          generator(current_generator_index, element));
            }
          }
        }
        current_processor_index += num_elements_at_current_generator_index;
      },
      number_of_traversed_elements_this_iteration);

  // processor expand if necessary and rerun
  if (!processor.can_insert_element_at_index(
          number_of_traversed_elements_this_iteration - 1)) {
    processor.expand(number_of_traversed_elements_this_iteration);
    auto failed_index_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), failed_index);
    // TODO Probably will need a custom reducer or reducer with custom type, but
    // not available for scan.
    Kokkos::parallel_for(
        "Generator_processor_for final pass",
        Kokkos::RangePolicy(exec_space,
                            generator_start_index > failed_index_host(0)
                                ? generator_start_index
                                : failed_index_host(0),
                            generator_end_index),
        KOKKOS_LAMBDA(IndexingType current_generator_index) {
          auto num_elements_at_current_generator_index =
              generator.number_of_elements_for_index(current_generator_index);

          for (auto element = 0;
               element < num_elements_at_current_generator_index; ++element)
            processor(offset[current_generator_index - generator_start_index] +
                          element,
                      generator(current_generator_index, element));
        });
    exec_space.fence("End fence of generator_processor_scan_offset");
  }
}
}  // namespace PPP

#endif
