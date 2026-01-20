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
- nvcc -o <output-file> <kernel.cu>
- cudaDeviceSynchronize()
  - Host에서 Device의 동작이 모두 끝나기를 대기

# 21. 002: Hello World program (Warp_IDs)

# 22. 003: Vector addition + the Steps for any CUDA project

# 23. 004: Vector addition + blocks and thread indexing + GPU performance

# 24. 005: levels of parallelization - Vector addition with Extra-large vectors
