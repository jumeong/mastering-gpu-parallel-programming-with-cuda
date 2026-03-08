# 49. Naive implementation of the matrix multiplication

## 1. 개요 및 중요성

* **컴퓨팅의 초석:** 시뮬레이션, 최적화, 선형 대수 연산의 핵심.
* **AI/DL의 엔진:** 신경망의 각 레이어는 본질적으로 행렬 연산이며, 이를 최적화하면 학습과 추론 속도가 직결됨.
* **주요 라이브러리:** * **NVIDIA:** cuBLAS(고성능), CUTLASS(유연한 커스텀).
* **AMD:** Tensile(설정 기반), rocBLAS.



## 2. 하드웨어 가속기 (Specialized Units)

행렬 연산만을 위해 설계된 전용 하드웨어 유닛을 사용하면 일반 코어 대비 수십 배의 처리량(Throughput)을 얻을 수 있습니다.

* **NVIDIA Tensor Cores:** A100 기준, 일반 코어 대비 약 10~20배 이상의 성능 향상.
* **AMD Matrix Cores:** CDNA 아키텍처 등에서 동일한 역할 수행.

## 3. CPU vs GPU 구현 차이

행렬 곱 $C = A \times B$ ($A: M \times K, B: K \times N, C: M \times N$) 기준:

| 구분 | CPU 구현 (Sequential) | GPU 구현 (Naive Parallel) |
| --- | --- | --- |
| **구조** | 3중 For 루프 (i, j, k) | 2D 스레드 인덱싱 + 1중 For 루프 (k) |
| **인덱싱** | `for(i...)`, `for(j...)` | `row = blockIdx.y * blockDim.y + threadIdx.y`, `col = blockIdx.x * blockDim.x + threadIdx.x` |
| **연산 방식** | 한 스레드가 모든 결과 값을 순차 계산 | 각 스레드가 결과 행렬 $C$의 요소 하나씩 할당받아 병렬 계산 |

## 4. CUDA 프로그래밍 7단계 워크플로우

1. **Memory Allocation:** `cudaMalloc`으로 GPU 메모리 확보.
2. **Initialization:** 호스트(CPU)에서 데이터 초기화.
3. **HtoD Copy:** `cudaMemcpy` (Host to Device)로 데이터 전송.
4. **Kernel Launch:** `<<<grid, block>>>` 설정으로 커널 실행.
5. **DtoH Copy:** `cudaMemcpy` (Device to Host)로 결과 회수.
6. **Validation:** CPU 결과와 GPU 결과를 비교하여 검증 (절대 오차 합 계산).
7. **Free Memory:** `cudaFree` 및 `delete`로 자원 해제.

## 5. 성능 분석 및 최적화 전략 (Profiling)

* **Nsight Compute 활용:**
  * **Duration:** 커널 실행 시간 측정 (Naive 방식은 오버헤드가 큼).
  * **Throughput:** Compute와 Memory의 사용률 확인.
  * **Roofline Analysis:** 현재 연산이 **Memory Bound**(메모리 대역폭 제한)인지 **Compute Bound**(연산 성능 제한)인지 판별.


* **Naive 방식의 한계:** 글로벌 메모리에 너무 빈번하게 접근함 ($2 \times K$번 읽기 당 2번 연산).
* **향후 과제 (Optimization):**
  * **Tiling (타일링):** 데이터를 작은 블록 단위로 쪼개어 처리.
  * **Shared Memory (공유 메모리):** 자주 쓰는 데이터를 L1 캐시급 속도의 공유 메모리에 올려 글로벌 메모리 접근 횟수를 획기적으로 줄임.

---


# 50. Optimize the Matrix Multiplication using the shared memory

## 1. 공유 메모리(Shared Memory) 할당

* **선언:** `__shared__ float s_a[TILE_SIZE][TILE_SIZE];`
* **목적:** 전역 메모리(Global Memory)에서 읽어온 타일 조각을 저장하여, 블록 내 스레드들이 데이터를 여러 번 재사용(Reuse)할 수 있게 함.
* **메모리 계산:** TILE_SIZE=16일 때, 한 블록당 약 2KB($16 \times 16 \times 4 \times 2$)를 사용하여 하드웨어 제한(보통 48KB~164KB) 내에서 안전하게 동작함.

## 2. 커널의 핵심 로직 (The Sliding Window)

타일링 연산은 결과값 $C(i, j)$를 계산하기 위해 행렬 $A$와 $B$를 **타일 단위로 훑으며(Iterate over tiles)** 진행됩니다.

1. **Index Mapping:** 전역 메모리 주소(`row`, `col`)를 계산하여 현재 스레드가 담당할 $C$의 위치를 확정.
2. **Tile Loop (K-dimension):** * **Load:** 전역 메모리에서 타일 조각을 가져와 `s_a`, `s_b`에 저장.
* **Sync:** `__syncthreads()`를 호출하여 블록 내 모든 스레드가 로드를 마칠 때까지 대기.
* **Compute:** 공유 메모리에 올라온 타일 데이터를 사용하여 부분 합(Partial Sum)을 계산.
* **Sync:** 다음 타일을 로드하기 전, 현재 타일 연산이 끝났음을 보장하기 위해 다시 동기화.


3. **Store:** 모든 타일에 대한 연산이 끝나면 최종 `sum`을 전역 메모리 $C$에 저장.

## 3. 성능 최적화 결과 (Benchmarks)

강의 내 실험 결과(1024x1024 기준)는 다음과 같습니다:

| 커널 종류 | 실행 시간 (approx.) | 성능 향상 |
| --- | --- | --- |
| **Naive Kernel** | ~3,000 $\mu s$ (3ms) | 기준점 |
| **Tiled Kernel** | ~2,000 $\mu s$ (2ms) | **약 33% 성능 향상** |

## 4. 하드웨어 제약 사항 (Critical Constraints)

강사가 강조한 **실패 사례(64x64 타일)**를 통해 배우는 CUDA 제약 조건:

* **스레드 블록 한계:** 하나의 블록은 최대 **1,024개**의 스레드만 가질 수 있음.
* **계산:** $32 \times 32 = 1,024$ (최대치), $64 \times 64 = 4,096$ (**오류 발생/실행 불가**).
* **권장 사항:** 보통 $16 \times 16$ (256개) 또는 $32 \times 32$ (최대한 활용 시)를 주로 사용하며, 연산 복잡도에 따라 최적의 `TILE_SIZE`를 찾아야 함.

---
