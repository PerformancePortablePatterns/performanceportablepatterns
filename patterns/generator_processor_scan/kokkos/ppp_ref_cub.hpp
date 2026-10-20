#ifndef PPP_REF_CUB_HPP
#define PPP_REF_CUB_HPP

#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace PPP {
template <class Generator>
__global__ void compute_counts(Generator g, int start, std::uint64_t n,
                               std::uint64_t *counts) {
  for (std::uint64_t j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
       j += std::uint64_t(blockDim.x) * gridDim.x) {
    const int index = start + static_cast<int>(j);
    counts[j] =
        static_cast<std::uint64_t>(g.number_of_elements_for_index(index));
  }
}

template <class T, class Generator>
__global__ void generate_elements(Generator g, int start, std::uint64_t n,
                                  const std::uint64_t *counts,
                                  const std::uint64_t *offsets, T *output) {
  for (std::uint64_t j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
       j += std::uint64_t(blockDim.x) * gridDim.x) {
    const int index                  = start + static_cast<int>(j);
    const std::uint64_t output_begin = offsets[j];

    for (std::uint64_t e = 0; e < counts[j]; ++e)
      output[output_begin + e] = g(index, static_cast<int>(e));
  }
}

template <class ExecutionSpace, class GeneratorType, typename ProcessorType,
          typename OffsetType>
void ref_cub_three_pass(ExecutionSpace exec, GeneratorType const &g,
                        ProcessorType &p, OffsetType &offset) {
  auto start = g.start_index();
  auto end   = g.end_index();
  const std::uint64_t n =
      end > start ? static_cast<std::uint64_t>(end - start) : 0;

  if (n == 0) return;

  cudaStream_t stream   = nullptr;
  std::uint64_t *counts = nullptr;
  cudaMallocAsync(&counts, n * sizeof(*counts), stream);

  constexpr int block_size = 256;
  const int grid_size = static_cast<int>((n + block_size - 1) / block_size);

  compute_counts<<<grid_size, block_size, 0, stream>>>(g, start, n, counts);

  // Determine temporary storage needed by CUB.
  void *scan_storage            = nullptr;
  std::size_t scan_storage_size = 0;

  cub::DeviceScan::ExclusiveSum(scan_storage, scan_storage_size, counts,
                                offset.data(), n, stream);

  cudaMallocAsync(&scan_storage, scan_storage_size, stream);

  cub::DeviceScan::ExclusiveSum(scan_storage, scan_storage_size, counts,
                                offset.data(), n, stream);

  // total = offsets[n - 1] + counts[n - 1]
  std::uint64_t last_offset;
  std::uint64_t last_count;

  cudaMemcpyAsync(&last_offset, offset.data() + n - 1, sizeof(last_offset),
                  cudaMemcpyDeviceToHost, stream);

  cudaMemcpyAsync(&last_count, counts + n - 1, sizeof(last_count),
                  cudaMemcpyDeviceToHost, stream);

  // Required before allocating the precisely sized output.
  cudaStreamSynchronize(stream);

  auto output_size = last_offset + last_count;

  if (output_size != 0) {
    Kokkos::realloc(Kokkos::WithoutInitializing, p.processed_elements,
                    output_size);

    generate_elements<<<grid_size, block_size, 0, stream>>>(
        g, start, n, counts, offset.data(), p.processed_elements.data());
  }
  cudaStreamSynchronize(stream);

  cudaFreeAsync(scan_storage, stream);
  cudaFreeAsync(counts, stream);
}
}  // namespace PPP

#endif
