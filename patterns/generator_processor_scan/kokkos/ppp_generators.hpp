#ifndef PPP_GENERATORS_HPP
#define PPP_GENERATORS_HPP

namespace PPP::Test {

template <typename IdxT, typename ElementT, unsigned N>
struct ModuloNGenerator {
  using ElementType  = ElementT;
  using IndexingType = IdxT;
  IndexingType start = 0;
  IndexingType end   = 0;

  KOKKOS_FUNCTION IndexingType start_index() const {
    // Return the starting index for the generator.
    return start;
  }

  KOKKOS_FUNCTION IndexingType end_index() const {
    // Return the ending index for the generator.
    return end;
  }

  KOKKOS_FUNCTION IndexingType
  number_of_elements_for_index(IndexingType i) const {
    IndexType ret = i % N == 0 ? N : 0;
    return ret;  // Example: varying number of elements based on index
  }

  KOKKOS_FUNCTION ElementType operator()(IndexingType generator_index,
                                         IndexingType element_index) const {
    // Generate an element based on the generator index and element index.
    return ElementType{generator_index, element_index};
  }
};

template <typename IdxT, typename ElementT, unsigned N>
ModuloNGenerator<IdxT, ElementT, N> make_shared(
    ModuloNGenerator<IdxT, ElementT, N> const &src) {
  return ModuloNGenerator<IdxT, ElementT, N>{src};
}

template <typename IdxT, typename ElementT, unsigned N, typename MemorySpace>
struct RandomIndexAccessGenerator {
  using ElementType  = ElementT;
  using IndexingType = IdxT;
  Kokkos::View<IndexingType *, MemorySpace> random_offset;
  Kokkos::View<ElementType *, MemorySpace> return_values;
  IndexingType start = 0;
  IndexingType end   = 0;

  KOKKOS_FUNCTION IndexingType start_index() const {
    // Return the starting index for the generator.
    return start;
  }

  KOKKOS_FUNCTION IndexingType end_index() const {
    // Return the ending index for the generator.
    return end;
  }

  KOKKOS_FUNCTION IndexingType
  number_of_elements_for_index(IndexingType i) const {
    return random_offset(i % N) %
           N;  // Example: varying number of elements based on index
  }

  KOKKOS_FUNCTION ElementType operator()(IndexingType generator_index,
                                         IndexingType element_index) const {
    // Generate an element based on the generator index and element index.
    auto random_index(random_offset((generator_index + element_index) %
                                    random_offset.extent(0)) %
                      return_values.extent(0));
    auto ret            = return_values(random_index);
    ret.generator_index = generator_index;
    ret.element_index   = random_index;
    return ret;
  }
};

template <typename IdxT, typename ElementT, unsigned N, typename MemorySpaceSrc>
RandomIndexAccessGenerator<IdxT, ElementT, N, Kokkos::SharedSpace> make_shared(
    RandomIndexAccessGenerator<IdxT, ElementT, N, MemorySpaceSrc> const &src) {
  auto ret =
      RandomIndexAccessGenerator<IdxT, ElementT, N, Kokkos::SharedSpace>{};
  ret.start = src.start;
  ret.end   = src.end;
  Kokkos::realloc(Kokkos::WithoutInitializing, ret.random_offset,
                  src.random_offset.span());
  Kokkos::realloc(Kokkos::WithoutInitializing, ret.return_values,
                  src.return_values.span());
  Kokkos::deep_copy(ret.random_offset, src.random_offset);
  Kokkos::deep_copy(ret.return_values, src.return_values);

  return ret;
}
}  // namespace PPP::Test
#endif
