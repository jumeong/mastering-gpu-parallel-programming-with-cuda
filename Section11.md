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

# 51. Optimization of the MM using float4 (important)

## 🚀 CUDA 행렬 곱셈 최적화: `float4` & 벡터화 요약

강의의 핵심은 **"스레드 하나가 4개의 데이터를 한 번에 처리하게 만들어, 메모리 대역폭과 연산 효율을 동시에 잡는 것"**입니다.

### 1. 주요 최적화 기법: `float4`의 도입

* **개념:** 기존에는 스레드 하나가 `float` 데이터 1개를 읽었지만, 이제는 `float4`라는 내장 자료형을 사용해 **128-bit(4바이트 × 4)**를 한 번에 읽어옵니다.
* **효과:** * **메모리 대역폭 효율:** 메모리 요청 횟수가 줄어들어 메모리 바운드(Memory Bound) 문제가 해결됩니다.
* **연산 효율:** `sum[0]~sum[3]`처럼 루프를 풀어서(Unrolling) 작성함으로써 제어 오버헤드를 줄이고 연산 바운드(Compute Bound)를 개선합니다.


* **구조적 변화:** 스레드 수가 1/4로 줄어듭니다. 각 스레드가 4개 분량의 일을 하기 때문입니다.

## 2. 코드 레벨의 변화 및 주의점

| 구분 | 내용 |
| --- | --- |
| **인덱스 계산** | 각 스레드의 시작 주소에 `threadIdx.x * 4`를 곱해 겹치지 않게 오프셋을 설정합니다. |
| **데이터 로딩** | `reinterpret_cast<float4*>(...)`를 사용하여 전역 메모리의 주소를 `float4` 포인터로 강제 변환(Casting)해 한꺼번에 읽어옵니다. |
| **연산부** | `dot(.)` 연산자(`.x`, `.y`, `.z`, `.w`)를 사용하여 4개 성분의 부분합을 각각 계산합니다. |
| **동기화** | `__syncthreads()`를 통해 공유 메모리에 데이터 로딩이 완료된 후 연산을 시작하도록 보장합니다. |

## 3. 성능 개선 결과 (강의 내 벤치마크)

강의에서 보여준 실행 시간 변화는 매우 드라마틱합니다:

* **Naive (기본형):** 360 $\mu s$
* **Shared Memory (최적화 1단계):** 250 $\mu s$
* **Float4 + Unrolling (최종 최적화):** **150 $\mu s$** (약 2.4배 성능 향상)

---

## 💡 보충 설명: 왜 casting(`reinterpret_cast`)이 필요한가요?

강사님은 연구적 관점에서 **전역 메모리(Global)와 공유 메모리(Shared)의 주소 체계 차이**를 언급하셨습니다.

1. 기본적으로 행렬은 `float` 배열로 선언되어 있습니다.
2. 이를 `float4` 단위로 읽으려면 하드웨어에 "이 주소부터는 4개씩 묶인 데이터다"라고 알려줘야 합니다.
3. 이때 `reinterpret_cast`를 통해 데이터 타입의 해석 방식을 강제로 바꿔주는 것입니다.

---

