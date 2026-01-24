# 17. The course github repo
- https://github.com/hamdysoltan/CUDA_Course

# 18. Mapping SW from CUDA to HW + introducing CUDA
- Host: CPU & DRAM
- Device: GPU & GDRAM

| Hardware      | Software            |
| --------      | --------            |
| **GPU**       | **GPU application** | 
| **SM**        | **Block**           |
| **SMSP**      | **Warp**            |
| **CUDA Core** | **Thread**          |

# 19. 001: Hello World program (threads - Blocks)
- kernel<<<gridDim, blockDim>>>(...);
- gridDim은 몇 개의 block을 띄울지
- blockDim은 block 하나당 몇 개의 thread를 사용할지
- A100 기준 Max Thread Block Size는 1024
- A100 기준 SM 수는 108개이고 Max Thread Blocks/SM은 32이므로 칩 전체에서 실행 가능한 Block 수는 32*108개

# 20. Compiling Cuda on Linux
- nvcc --version
- nvcc -arch=sm_75 -o <output-file> <kernel.cu>
- cudaDeviceSynchronize()
  - Host에서 Device의 동작이 모두 끝나기를 대기

# 21. 002: Hello World program (Warp_IDs)

# 22. 003: Vector addition + the Steps for any CUDA project
1. Allocate memory for vectors A, B, and C on the host and GPU
2. Initialize vectors A and B with values.
3. Copy the host vectors to the device.
4. Define the CUDA kernel vectorAdd, which adds the vectors element-wise
5. Launch the kernel with a suitable number of blocks and threads.
6. Copy the result back to the host vector C.
7. Free the allocated memory on both the host and device.

# 23. 004: Vector addition + blocks and thread indexing + GPU performance

# 24. 005: levels of parallelization - Vector addition with Extra-large vectors
