#ifndef PPP_COUNT_OFFSET_FILL_HPP
#define PPP_COUNT_OFFSET_FILL_HPP

namespace PPP {
template <typename ExecutionSpace, typename GeneratorType,
          typename ProcessorType, typename OffsetType>
void count_offset_fill(ExecutionSpace const &exec_space,
                       GeneratorType const &generator, ProcessorType &processor,
                       OffsetType &offset) {
  using IndexingType         = typename GeneratorType::IndexingType;
  auto generator_start_index = generator.start_index();
  auto generator_end_index   = generator.end_index();

  Kokkos::parallel_for(
      "Reference::count",
      Kokkos::RangePolicy(exec_space, generator_start_index,
                          generator_end_index),
      KOKKOS_LAMBDA(IndexingType i) {
        offset[i - generator_start_index] =
            generator.number_of_elements_for_index(i);
      });

  IndexingType total_count;
  Kokkos::parallel_scan(
      "Reference::compute_offset",
      Kokkos::RangePolicy{exec_space, generator_start_index,
                          generator_end_index},
      KOKKOS_LAMBDA(IndexingType i, IndexingType & partial_count,
                    bool is_final) {
        auto offset_i = offset[i - generator_start_index];
        if (is_final) offset[i - generator_start_index] = partial_count;
        partial_count += offset_i;
      },
      total_count);

  if (!processor.can_insert_element_at_index(total_count - 1))
    Kokkos::realloc(Kokkos::WithoutInitializing, processor.processed_elements,
                    total_count);

  Kokkos::parallel_for(
      "Reference:fill",
      Kokkos::RangePolicy{exec_space, generator_start_index,
                          generator_end_index},
      KOKKOS_LAMBDA(IndexingType i) {
        auto count    = generator.number_of_elements_for_index(i);
        auto offset_i = offset[i - generator_start_index];
        for (IndexingType element = 0; element < count; ++element)
          processor(offset_i + element, generator(i, element));
      });
  exec_space.fence("End fence of count_offset_fill");
}

template <typename ExecutionSpace, typename GeneratorType,
          typename ProcessorType, typename OffsetType>
void countfill_offset_fill(ExecutionSpace exec_space,
                           GeneratorType const &generator,
                           ProcessorType &processor, OffsetType &offset) {
  using IndexingType         = typename GeneratorType::IndexingType;
  auto generator_start_index = generator.start_index();
  auto generator_end_index   = generator.end_index();

  IndexingType number_of_traversed_elements_this_iteration;
  Kokkos::parallel_scan(
      "Reference::count_and_try_to_fill",
      Kokkos::RangePolicy(exec_space, generator_start_index,
                          generator_end_index),
      KOKKOS_LAMBDA(IndexingType current_generator_index,
                    IndexingType & current_processor_index, bool is_final) {
        auto count =
            generator.number_of_elements_for_index(current_generator_index);

        if (is_final) {
          offset[current_generator_index - generator_start_index] =
              current_processor_index;
          if (count != 0 && processor.can_insert_element_at_index(
                                current_processor_index + count)) {
            for (IndexingType element = 0; element < count; ++element)
              processor(current_processor_index + element,
                        generator(current_generator_index, element));
          }
        }
        current_processor_index += count;
      },
      number_of_traversed_elements_this_iteration);

  if (processor.can_insert_element_at_index(
          number_of_traversed_elements_this_iteration - 1))
    return;

  IndexingType max_storage = processor.processed_elements.extent(0);
  processor.expand(number_of_traversed_elements_this_iteration);

  IndexingType restart_index;
  Kokkos::parallel_reduce(
      "Reference::find_restart_index",
      Kokkos::RangePolicy(exec_space, generator_start_index,
                          generator_end_index),
      KOKKOS_LAMBDA(IndexingType i, IndexingType & partial_max) {
        if (i > partial_max &&
            offset[i - generator_start_index] < max_storage) {
          partial_max = i;
        }
      },
      Kokkos::Max<IndexingType>{restart_index});

  Kokkos::parallel_for(
      "Reference:fill_remainder",
      Kokkos::RangePolicy{exec_space, restart_index, generator_end_index},
      KOKKOS_LAMBDA(IndexingType i) {
        auto count    = generator.number_of_elements_for_index(i);
        auto offset_i = offset[i - generator_start_index];
        for (IndexingType element = 0; element < count; ++element)
          processor(offset_i + element, generator(i, element));
      });
  exec_space.fence("End fence of countfill_offset_fill");
}
}  // namespace PPP

#endif
