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

# GPU L2 Cache와 Write Hit 정리

## 1. Write Hit이 존재하는 이유
- GPU L2 Cache는 **read뿐 아니라 write도 캐시**함
- 일반적인 global store는:
  - **write-allocate**: cache line이 없으면 L2로 가져와서 씀
  - **write-back**: 즉시 DRAM에 쓰지 않고 L2에 dirty로 유지
- 이미 L2에 존재하는 cache line에 write하면 → **L2 write hit**

## 2. Write Hit의 의미
- 같은 cache line에 반복 write 시:
  - 첫 write만 miss
  - 이후 write는 모두 hit
- 효과:
  - DRAM write 트래픽 감소
  - write locality가 좋다는 신호
  - 보통 성능에 유리

## 3. L2 Cache가 꽉 찼을 때 동작
- L2는 **미리 flush해서 자리를 만들지 않음**
- write 요청이 오면:
  1. eviction 발생 (on-demand)
  2. victim line이
     - clean → 그냥 제거
     - dirty → DRAM으로 write-back
  3. 공간 확보 후 write-allocate
- 결과:
  - dirty eviction이 많으면 성능 급락
  - write locality가 매우 중요

## 4. Write하는 입장에서의 동작
- 대부분의 global store는:
  - **L2에 쓸 수 있을 때까지 기다림**
  - eviction / write-back이 끝나야 진행
- 즉, 기본적으로는 **“L2 write를 목표로 stall”**

## 5. 예외 케이스
- **L2 bypass store**
  - write-through / cache-global 정책
  - DRAM으로 바로 write
  - write hit/miss 개념 없음
- **Atomic**
  - L2에서 serialize / merge
  - hit보다는 contention이 중요
- **System / Peer memory**
  - L2 bypass 또는 buffering만 수행

## 6. 성능 분석 시 같이 볼 Metric
- L2 관련:
  - `l2_write_hit_rate`
  - `l2_writeback_bytes`
  - `l2_evict_*`
- SM stall:
  - `smsp__stall_memory_dependency`

## 한 줄 요약
- GPU L2는 write도 캐시한다
- write hit은 정상적이고 중요한 개념
- L2가 부족하면 즉시 eviction + write-back
- 대부분의 store는 L2에 쓰기 위해 기다리지만, bypass 예외는 존재
