#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "ppp_count_offset_fill.hpp"
#include "ppp_generator_processor_scan.hpp"
#include "ppp_generator_processor_scan_offset.hpp"
#include "ppp_generators.hpp"
#include "ppp_processors.hpp"

#ifdef KOKKOS_ENABLE_CUDA
#include "ppp_ref_cub.hpp"
#endif

#include <iostream>

// TODO write deduction guides for processors and generators

///////////////////Helper function
template <typename MemorySpace, typename ExecutionSpace, typename Generator,
          typename IndexingType>
double time_generator_processor_algo(
    void (*funcptr)(ExecutionSpace const &, Generator const &,
                    PPP::Test::IdentityInsertingProcessor<
                        typename Generator::ElementType, MemorySpace> &),
    ExecutionSpace const &exec_space, Generator const &generator,
    IndexingType size_prealloc) {
  using ElementType = typename Generator::ElementType;

  PPP::Test::IdentityInsertingProcessor<ElementType, MemorySpace> processor{
      Kokkos::View<ElementType *, MemorySpace>(
          Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                             "processed_elements"),
          size_prealloc)};

  Kokkos::fence();
  Kokkos::Timer timer;
  funcptr(exec_space, generator, processor);
  Kokkos::fence();

  return timer.seconds();
}

template <typename MemorySpace, typename ExecutionSpace, typename Generator,
          typename OffsetType, typename IndexingType>
double time_generator_processor_offset_algo(
    void (*funcptr)(ExecutionSpace const &, Generator const &,
                    PPP::Test::IdentityInsertingProcessor<
                        typename Generator::ElementType, MemorySpace> &,
                    OffsetType &),
    ExecutionSpace const &exec_space, Generator const &generator,
    OffsetType &offset, IndexingType size_prealloc) {
  using ElementType = typename Generator::ElementType;

  PPP::Test::IdentityInsertingProcessor<ElementType, MemorySpace> processor{
      Kokkos::View<ElementType *, MemorySpace>(
          Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                             "processed_elements"),
          size_prealloc)};

  Kokkos::fence();
  Kokkos::Timer timer;
  funcptr(exec_space, generator, processor, offset);
  Kokkos::fence();
  return timer.seconds();
}

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    if (argc < 5) {
      printf(
          "Arguments: start_idx end_idx size_prealloc num_rep algorithm "
          "generator\n");
      Kokkos::finalize();
      return 0;
    }

    using IndexingType         = std::uint64_t;
    IndexingType start_index   = std::stoull(argv[1]);
    IndexingType end_index     = std::stoull(argv[2]);
    IndexingType size_prealloc = std::stoull(argv[3]);
    IndexingType num_rep       = std::stoull(argv[4]);
    IndexingType algo          = std::stoull(argv[5]);
    IndexingType generator     = std::stoull(argv[6]);

    static const unsigned N = 5;
    using ElementType       = PPP::Test::ElementPair<IndexingType>;
    using memory_space      = Kokkos::DefaultExecutionSpace::memory_space;
    auto exec_space         = Kokkos::DefaultExecutionSpace{};

    // std::cout << "#start_idx, end_idx, size_prealloc, num_rep, algorithm,
    // generator, time\n";

    double time = std::nan("1");
    for (std::uint64_t rep = 0; rep < num_rep; ++rep) {
      {
        Kokkos::View<IndexingType *, memory_space> offset(
            Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                               "offset"),
            end_index - start_index);

        PPP::Test::ModuloNGenerator<IndexingType, ElementType, N>
            modulo_generator{start_index, end_index};

        Kokkos::Random_XorShift64_Pool<> rand_pool(183291);
        Kokkos::View<IndexingType *, memory_space> random_indices(
            Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                               "random_indices"),
            end_index / 2);
        Kokkos::fill_random(exec_space, random_indices, rand_pool,
                            end_index / 2);

        // add some warmup
        for (auto i = 0; i < 10; ++i)
          Kokkos::fill_random(exec_space, offset, rand_pool, offset.extent(0));

        Kokkos::deep_copy(exec_space, offset, 0);

        PPP::Test::RandomIndexAccessGenerator<IndexingType, ElementType, N,
                                              memory_space>
            random_generator{
                random_indices,
                Kokkos::View<ElementType *, memory_space>(
                    Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                                       "generator_elements"),
                    end_index / 2),
                start_index, end_index};

        Kokkos::fence();

        switch (algo) {
          case 0:
            if (generator == 0) {
              time = time_generator_processor_algo<memory_space>(
                  PPP::generator_processor_scan, exec_space, modulo_generator,
                  size_prealloc);
            } else if (generator == 1) {
              time = time_generator_processor_algo<memory_space>(
                  PPP::generator_processor_scan, exec_space, random_generator,
                  size_prealloc);
            }
            break;
          case 1:
            if (generator == 0) {
              time = time_generator_processor_offset_algo<memory_space>(
                  PPP::count_offset_fill, exec_space, modulo_generator, offset,
                  size_prealloc);
            } else if (generator == 1) {
              time = time_generator_processor_offset_algo<memory_space>(
                  PPP::count_offset_fill, exec_space, random_generator, offset,
                  size_prealloc);
            }
            break;
          case 2:
            if (generator == 0) {
              time = time_generator_processor_offset_algo<memory_space>(
                  PPP::countfill_offset_fill, exec_space, modulo_generator,
                  offset, size_prealloc);
            } else if (generator == 1) {
              time = time_generator_processor_offset_algo<memory_space>(
                  PPP::countfill_offset_fill, exec_space, random_generator,
                  offset, size_prealloc);
            }
            break;
          case 3:
            if (generator == 0) {
              time = time_generator_processor_offset_algo<memory_space>(
                  PPP::generator_processor_scan_offset, exec_space,
                  modulo_generator, offset, size_prealloc);
            } else if (generator == 1) {
              time = time_generator_processor_offset_algo<memory_space>(
                  PPP::generator_processor_scan_offset, exec_space,
                  random_generator, offset, size_prealloc);
            }
            break;
#ifdef KOKKOS_ENABLE_CUDA
          case 4:
            if (generator == 0) {
              time = time_generator_processor_offset_algo<memory_space>(
                  PPP::ref_cub_three_pass, exec_space, modulo_generator, offset,
                  size_prealloc);
            } else if (generator == 1) {
              time = time_generator_processor_offset_algo<memory_space>(
                  PPP::ref_cub_three_pass, exec_space, random_generator, offset,
                  size_prealloc);
            }
            break;
#endif
          default: std::cout << "algo not found \n"; std::exit(1);
        }
      }
      std::cout << start_index << " , " << end_index << " , " << size_prealloc
                << " , " << rep << " , " << algo << " , " << generator << " , "
                << time << "\n";
    }
    // if on which algo to do
    // warmup: run the randomSeeding multiple times?
  }
  Kokkos::finalize();
  return 0;
}
