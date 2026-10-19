#ifndef PPP_PROCESSORS_HPP
#define PPP_PROCESSORS_HPP

namespace PPP::Test {
template <typename ElementType, typename MemorySpace>
struct IdentityInsertingProcessor {
  Kokkos::View<ElementType *, MemorySpace> processed_elements;

  template <typename IndexingType>
  KOKKOS_FUNCTION bool can_insert_element_at_index(IndexingType index) const {
    // Check if the processor can insert an element at the given index.
    return index < processed_elements.extent(0);
  }

  template <typename IndexingType>
  KOKKOS_FUNCTION void operator()(IndexingType processor_index,
                                  ElementType const &element) const {
    // Process the element at the given processor index.
    processed_elements(processor_index) = element;
  }

  template <typename IndexingType>
  void expand(IndexingType new_capacity) {
    // Expand the processor's capacity to accommodate more elements.
    Kokkos::resize(Kokkos::WithoutInitializing, processed_elements,
                   new_capacity);
  }
};

template <typename IndexingType>
struct ElementPair {
  IndexingType generator_index = 99;
  IndexingType element_index   = 99;

  KOKKOS_FUNCTION bool operator==(ElementPair const &other) const {
    return generator_index == other.generator_index &&
           element_index == other.element_index;
  }
};

template <typename HostElementView, typename Generator>
void serial_ElementPair_check(HostElementView const &processed_elements_host,
                              Generator const &generator) {
  using IndexingType                  = typename Generator::IndexingType;
  IndexingType processor_index_offset = 0;
  for (auto generator_index = generator.start_index();
       generator_index < generator.end_index(); ++generator_index) {
    for (auto element_index = 0;
         element_index <
         generator.number_of_elements_for_index(generator_index);
         ++element_index) {
      IndexingType actual_processor_index =
          element_index + processor_index_offset;
      EXPECT_EQ(processed_elements_host(actual_processor_index).generator_index,
                generator(generator_index, element_index).generator_index)
          << " Mismatch in generator_index"
          << " generator_index: " << generator_index
          << " processor_index_offset: " << processor_index_offset
          << " actual_processor_index: " << actual_processor_index << "\n";
      EXPECT_EQ(processed_elements_host(actual_processor_index).element_index,
                generator(generator_index, element_index).element_index)
          << " Mismatch in element_index"
          << " generator_index: " << generator_index
          << " processor_index_offset: " << processor_index_offset
          << " actual_processor_index: " << actual_processor_index << "\n";
    }
    processor_index_offset +=
        generator.number_of_elements_for_index(generator_index);
  }
}

template <typename Generator, typename OffsetType>
void serial_offset_check(Generator const &generator, OffsetType const &offset) {
  using IndexingType                  = typename Generator::IndexingType;
  IndexingType processor_index_offset = 0;
  for (auto generator_index = generator.start_index();
       generator_index < generator.end_index(); ++generator_index) {
    EXPECT_EQ(processor_index_offset,
              offset(generator_index - generator.start_index()))
        << " Mismatch in offset"
        << " generator_index: " << generator_index
        << " processor_index_offset: " << processor_index_offset
        << " actual_offset: "
        << offset(generator_index - generator.start_index()) << "\n";
    processor_index_offset +=
        generator.number_of_elements_for_index(generator_index);
  }
}

}  // namespace PPP::Test
#endif
