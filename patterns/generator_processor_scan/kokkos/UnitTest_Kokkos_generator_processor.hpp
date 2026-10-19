#ifndef UNIT_TEST_KOKKOS_GENERATOR_PROCESSOR
#define UNIT_TEST_KOKKOS_GENERATOR_PROCESSOR

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <gtest/gtest.h>

#include "ppp_count_offset_fill.hpp"
#include "ppp_generator_processor_scan.hpp"
#include "ppp_generator_processor_scan_offset.hpp"
#include "ppp_generators.hpp"
#include "ppp_processors.hpp"

// TODO: right now the random access generator unit test will fail on device ->
// create a host mirror and create a generator with it for testing?
// TODO: remove the function pointers ...
// TODO: split into multiple files for easier reuse (DRY)

///////////////////Helper function
template <typename MemorySpace, typename ExecutionSpace, typename Generator,
          typename IndexType>
void test_generator_processor_algo(
    void (*funcptr)(ExecutionSpace const &, Generator const &,
                    PPP::Test::IdentityInsertingProcessor<
                        typename Generator::ElementType, MemorySpace> &),
    ExecutionSpace const &exec_space, Generator const &generator,
    IndexType size_prealloc) {
  using ElementType = typename Generator::ElementType;

  PPP::Test::IdentityInsertingProcessor<ElementType, MemorySpace> processor{
      Kokkos::View<ElementType *, MemorySpace>(
          Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                             "processed_elements"),
          size_prealloc)};

  funcptr(exec_space, generator, processor);
  Kokkos::fence();

  auto processed_elements_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), processor.processed_elements);

  auto shared_generator = make_shared(generator);
  serial_ElementPair_check(processed_elements_host, shared_generator);
}

template <typename MemorySpace, typename ExecutionSpace, typename Generator,
          typename OffsetType, typename IndexType>
void test_generator_processor_offset_algo(
    void (*funcptr)(ExecutionSpace const &, Generator const &,
                    PPP::Test::IdentityInsertingProcessor<
                        typename Generator::ElementType, MemorySpace> &,
                    OffsetType &),
    ExecutionSpace const &exec_space, Generator const &generator,
    OffsetType &offset, IndexType size_prealloc) {
  using ElementType = typename Generator::ElementType;

  PPP::Test::IdentityInsertingProcessor<ElementType, MemorySpace> processor{
      Kokkos::View<ElementType *, MemorySpace>(
          Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                             "processed_elements"),
          size_prealloc)};

  funcptr(exec_space, generator, processor, offset);
  Kokkos::fence();

  auto processed_elements_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), processor.processed_elements);

  auto shared_generator = make_shared(generator);
  serial_ElementPair_check(processed_elements_host, shared_generator);

  auto offset_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), offset);
  serial_offset_check(shared_generator, offset_host);
}

TEST(TEST_CATEGORY, generator_processor_scan) {
  using IndexingType                            = int;
  std::array<IndexingType, 4> start_index_set   = {0, 2, 100, 39'998};
  std::array<IndexingType, 4> end_index_set     = {1, 40'000, 20'000, 40'000};
  std::array<IndexingType, 3> size_prealloc_set = {1, 20'000, 50'000};
  static const unsigned N                       = 5;

  using ElementType  = PPP::Test::ElementPair<IndexingType>;
  using memory_space = TEST_EXECSPACE::memory_space;
  auto exec_space    = TEST_EXECSPACE{};

  for (int i = 0; i < start_index_set.size(); ++i) {
    auto start_index = start_index_set[i];
    auto end_index   = end_index_set[i];
    for (auto size_prealloc : size_prealloc_set) {
      PPP::Test::ModuloNGenerator<IndexingType, ElementType, N>
          modulo_generator{start_index, end_index};

      test_generator_processor_algo<memory_space>(PPP::generator_processor_scan,
                                                  exec_space, modulo_generator,
                                                  size_prealloc);

      if constexpr (!std::is_same_v<TEST_EXECSPACE,
                                    Kokkos::DefaultHostExecutionSpace>) {
        Kokkos::Random_XorShift64_Pool<> rand_pool(183291);
        Kokkos::View<IndexingType *, memory_space> random_indices(
            Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                               "random_indices"),
            N);
        Kokkos::fill_random(exec_space, random_indices, rand_pool, N);

        PPP::Test::RandomIndexAccessGenerator<IndexingType, ElementType, N,
                                              memory_space>
            random_generator{
                random_indices,
                Kokkos::View<ElementType *, memory_space>(
                    Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                                       "generator_elements"),
                    N),
                start_index, end_index};

        test_generator_processor_algo<memory_space>(
            PPP::generator_processor_scan, exec_space, random_generator,
            size_prealloc);
      }
    }
  }
}

TEST(TEST_CATEGORY, generator_processor_offset_scan) {
  using IndexingType                            = int;
  std::array<IndexingType, 4> start_index_set   = {0, 2, 100, 39'998};
  std::array<IndexingType, 4> end_index_set     = {1, 40'000, 20'000, 40'000};
  std::array<IndexingType, 3> size_prealloc_set = {1, 20'000, 50'000};
  static const unsigned N                       = 5;
  using ElementType  = PPP::Test::ElementPair<IndexingType>;
  using memory_space = TEST_EXECSPACE::memory_space;
  auto exec_space    = TEST_EXECSPACE{};

  for (int i = 0; i < start_index_set.size(); ++i) {
    auto start_index = start_index_set[i];
    auto end_index   = end_index_set[i];
    for (auto size_prealloc : size_prealloc_set) {
      PPP::Test::ModuloNGenerator<IndexingType, ElementType, N>
          modulo_generator{start_index, end_index};
      Kokkos::View<IndexingType *, memory_space> offset(
          Kokkos::view_alloc(exec_space, "offset"), end_index - start_index);

      test_generator_processor_offset_algo<memory_space>(
          PPP::count_offset_fill, exec_space, modulo_generator, offset,
          size_prealloc);
      test_generator_processor_offset_algo<memory_space>(
          PPP::countfill_offset_fill, exec_space, modulo_generator, offset,
          size_prealloc);
      test_generator_processor_offset_algo<memory_space>(
          PPP::generator_processor_scan_offset, exec_space, modulo_generator,
          offset, size_prealloc);

      if constexpr (!std::is_same_v<TEST_EXECSPACE,
                                    Kokkos::DefaultHostExecutionSpace>) {
        Kokkos::Random_XorShift64_Pool<> rand_pool(183291);
        Kokkos::View<IndexingType *, memory_space> random_indices(
            Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                               "random_indices"),
            N);
        Kokkos::fill_random(exec_space, random_indices, rand_pool, N);

        PPP::Test::RandomIndexAccessGenerator<IndexingType, ElementType, N,
                                              memory_space>
            random_generator{
                random_indices,
                Kokkos::View<ElementType *, memory_space>(
                    Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                                       "generator_elements"),
                    N),
                start_index, end_index};

        test_generator_processor_offset_algo<memory_space>(
            PPP::count_offset_fill, exec_space, random_generator, offset,
            size_prealloc);
        test_generator_processor_offset_algo<memory_space>(
            PPP::countfill_offset_fill, exec_space, random_generator, offset,
            size_prealloc);
        test_generator_processor_offset_algo<memory_space>(
            PPP::generator_processor_scan_offset, exec_space, random_generator,
            offset, size_prealloc);
      }
    }
  }
}

#endif
