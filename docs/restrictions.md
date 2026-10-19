---
icon: lucide/octagon-minus
---

# Restrictions in heterogeneous computing

Many of the heterogeneous processors are highly specialized and introduce several restrictions.
As a consequence, software, and thus design patterns, that do not respect all these restrictions is not portable.
The most common set of restrictions are listed in this section.
Nevertheless, not all restrictions are in place in all hardware.
Therefore, for an application to be truly performance portable, it has to respect the superset of the restrictions all targeted heterogeneous hardware imposes.

## Forward progress guarantee

On architectures that use multiple threads that execute in lock-step, all threads that are controlled by a single control unit execute the same instruction.
These groups of threads are often called warps, workgroups, wave or similar.

This raises a question about what happens when some of the threads are stuck waiting for something while others can progress with the next instructions.
Forward progress guarantee in this context means, that it is guaranteed that the treads that can continue with the next instructions will do so (if there are any left), while the ones that have to wait continue to wait.
To illustrate the impact of forward progress guarantee, consider the following example:

``` c++ linenums="1"
while (not lock_address(address)) {
  // Exponential backoff
  for (int tick = 0; tick < delay; ++tick) {
    sleep(pow(10,tick));
    }

  // do work with lock

  release_lock(address);
}
```
This code tries to acquire a lock for a certain address.
When the lock can not be acquired, the thread waits for an duration that grows exponentially with the number of failed tries.

For simplicity (but without restricting generality), lets say all threads in a group want to lock the same address.
With forward progress guarantee, the thread that succeeds in acquiring the lock continues the instructions and releases the lock.
This allows another tread in the group to acquire the lock and also succeed.

But on an architecture without forward progress guarantee, all threads in the group need to finish the `while` loop before any of them can continue with the work.
This essentially leads to a deadlock as the thread that acquired the lock can not progress, while the rest of the group can not exit the `while` loop as the lock can not be acquired.

Thus in the context of performance portability, it is mandatory to have all threads of a group enter and exit the while loop together.

## Allocation of Memory

In general, not all processors in a heterogeneous computing system can allocate dynamic memory inside a kernel.

Dynamic memory allocation means requesting memory at runtime. The memory may be in system RAM, high-bandwidth memory (HBM), or another memory space. It does not have to be physically located on the machine; it may also be accessed over a network. Even when memory is physically located on an accelerator, the accelerator may not be able to allocate from it while a kernel is running.

Consider this pseudocode example:

```cpp
void kernel(unsigned N)
{
    int* gpu_data = nullptr;
    allocate_device_memory(&gpu_data, N); // Device-side allocation may not be supported.
}

int main()
{
    const unsigned N = 10;

    int* cpu_data = nullptr;
    allocate_host_memory(&cpu_data, N); // Host-side allocation is generally available.

    // Launch the kernel on the GPU.
    kernel(N);
}
```

The allocation inside `kernel` is not portable to accelerators that do not support dynamic allocation from device code. The allocation functions and kernel-launch syntax here are illustrative; actual APIs vary by programming model.

By contrast, kernels commonly support local variables and fixed-size local arrays:

```cpp
void kernel()
{
    double local_array[3]; // A fixed-size local array.
}
```

The array’s size must be known at compile time in languages such as C++. However, this does not necessarily mean the array occupies fast on-chip memory. Depending on the compiler and hardware, it may use registers or local memory, which can be backed by device memory.

Using many local variables or large arrays can reduce performance. For example, the resource use per thread may limit how many threads can run at once.

Compilers can sometimes reuse storage for local variables whose values are no longer needed. For example:

```cpp
void kernel()
{
    double local_array[2];
    // Fill local_array.

    double squared_magnitude =
        local_array[0] * local_array[0] +
        local_array[1] * local_array[1];

    // If local_array is not used again, its storage may be reused.
}
```

The compiler may reuse the storage occupied by `local_array` after its last use. This is an optimization, however, and whether it happens depends on the compiler, the selected optimization settings, and the code.

## Dynamic Polymorphism

## Conditionals

## Memory Accessibility

In heterogeneous programming, processors often have dedicated random-access memory (RAM).
Although features such as [HMM](https://www.kernel.org/doc/html/v5.1/vm/hmm.html) can allow certain GPUs to access CPU memory, that behavior is not guaranteed across architectures.
It is therefore useful to treat memory associated with different processors as distinct memory types.

This introduces an important restriction: **memory movement may need to be managed explicitly**. Consider the following example:

```cpp
void kernel(int* data)
{
    // Do work with data.
}

int main()
{
    const unsigned N = 10;
    int* data = nullptr;
    allocate_memory(&data, N);

    // We want to do this on the CPU.
    for (unsigned i = 0; i < N; ++i)
    {
        data[i] = /* read from file */;
    }

    // We want to do this on the GPU.
    kernel(data);
    // Wait for the GPU to finish the computation.

    // We want to do this on the CPU.
    for (unsigned i = 0; i < N; ++i)
    {
        // Write to file.
    }
}
```

This may work on accelerators (and CPUs) that can access the memory in which `data` was allocated. It may fail, however, on architectures that do not provide automatic memory management or access to that memory.

Furthermore, if the code does not explicitly express where data should be copied from CPU memory to GPU memory, and back again, it can be harder to reason about what the code is doing.

Compare that with a version that uses separate pointers for CPU and GPU memory and shows when data is copied between them:

```cpp
void kernel(int* data)
{
    // Do work with data.
}

int main()
{
    const unsigned N = 10;

    int* cpu_data = nullptr;
    allocate_cpu_memory(&cpu_data, N);

    int* gpu_data = nullptr;
    allocate_gpu_memory(&gpu_data, N);

    // We want to do this on the CPU.
    for (unsigned i = 0; i < N; ++i)
    {
        cpu_data[i] = /* read from file */;
    }

    copy(gpu_data, cpu_data); // Destination, source.

    // We want to do this on the GPU.
    kernel(gpu_data);
    // Wait for the GPU to finish the computation.

    copy(cpu_data, gpu_data); // Destination, source.

    // We want to do this on the CPU.
    for (unsigned i = 0; i < N; ++i)
    {
        // Write to file.
    }
}
```

In this version, it is clear when data is moved between memory spaces.
It also makes it possible to specify whether a `copy` operation always transfers the data or transfers it only when needed.
This can help maintain consistent behavior across architectures with shared and non-shared memory.

## Reference semantics

## Parallel execution in arbitrary sequence
