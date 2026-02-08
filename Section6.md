# 36. Matrices addition using 2D of block and threads
- `ix = threadIdx.x + blockIdx.x * blockDim.x`
- `iy = threadIdx.y + blockIdx.y * blockDim.y`
- `idx = iy * nx + ix`으로 표현 가능
<img width="1363" height="772" alt="image" src="https://github.com/user-attachments/assets/db37a2bc-1d10-457e-9a76-4a85f64d779a" />

# 37. Why L1 Hit-rate is zero?
- HBM → L2 Cache → L1 Cache
- 아주 당연하게도, 이전 예제에서는 input을 reuse하는 경우가 없으므로 L1 Hit rate가 zero
- L1 cache
  - Configurable (up to 192KB)
  - Cache line is 128 Bytes
  - Each warp consists of 32 threads
  - If a warp has a memory operation, it sends it for all threads in the same cyce not thread by thread
  - Once the LSU receives the request, it coalesces the data from the 32 threads.
  - If each thread needs 1 float element, the total for the warp is 128 bytes.
  - In this case, the LSU can request the whole cache line from L1.

# L2 Cache Write 개념이 신기해서 찾아본 내용
- GPU L2는 write도 캐시한다. 그래서 이미 L2에 있는 cache line에 write하면 DRAM 안 가고 L2에서 처리되는데, 그게 write hit다.
