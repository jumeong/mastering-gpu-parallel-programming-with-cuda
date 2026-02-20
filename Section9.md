# 41. Vector Reduction using global memory only (baseline)

## 1. 벡터 리덕션(Vector Reduction)의 정의

* **개념:** 대규모 벡터의 모든 요소를 하나의 값(합계 등)으로 줄여나가는 병렬 알고리즘입니다.
* **특징:** 두 벡터를 더하는 '벡터 덧셈'과 달리, **단일 벡터** 내에서 연산이 이루어지며 단계마다 데이터의 양이 절반씩 줄어듭니다.
* **용도:** 행렬 곱셈, 통계 계산 등 다양한 GPU 연산의 핵심 요소입니다.

---

## 2. 핵심 알고리즘: 트리 기반 접근 (Tree-based Approach)

이 방식은 데이터를 마치 거꾸로 된 나무 모양처럼 단계적으로 합쳐 나갑니다.

* **보폭(Stride)의 원리:**
* **Step 1:** 보폭이 1입니다. $input[0]$과 $input[1]$을 더합니다.
* **Step 2:** 보폭이 2입니다. $input[0]$과 $input[2]$를 더합니다.
* **Step 3:** 보폭이 4입니다. $input[0]$과 $input[4]$를 더합니다.


* 이처럼 보폭은 매 단계마다 **2배**씩 증가하고, 실제 연산에 참여하는 스레드 수는 **절반**씩 감소합니다.

---

## 3. 구현상의 주요 문제와 해결책

### ① 유휴 스레드(Idle Threads) 문제

연산 단계가 진행될수록 필요한 스레드 수는 줄어듭니다. 모든 스레드를 활성화하면 불필요한 연산이 발생하고 메모리 오류가 날 수 있습니다.

* **해결책 (필터링):** `if (tid % (2 * stride) == 0)` 조건을 사용해 현재 단계에서 연산이 필요한 짝수 번째 스레드만 골라냅니다.

### ② 메모리 경계 확인 (Memory Violation)

스레드가 `index + stride` 위치의 데이터를 참조할 때, 벡터의 크기()를 벗어나면 에러가 발생합니다.

* **해결책:** `if (index + stride < n)` 조건을 추가하여 안전한 범위 내에서만 메모리에 접근하도록 합니다.

### ③ 블록 간 동기화 문제 (Global Synchronization)

CUDA는 블록(Block) 간의 실시간 동기화를 지원하지 않습니다.

* **해결책 (커널 분리):**
1. **Kernel 1:** 각 블록이 담당 영역의 **부분 합(Partial Sums)**을 구합니다.
2. **연속 배치:** 각 블록의 결과값을 벡터의 앞부분(`input[blockIdx.x]`)에 모읍니다.
3. **Kernel 2:** 모인 부분 합들을 입력으로 하여 최종 합계가 나올 때까지 다시 리덕션을 실행합니다.



---

## 4. 최종 코드 논리 구조

```cpp
__global__ void reduce_inplace(int *input, int n) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 1. 단계별 리덕션 (for 루프)
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        // 짝수 보폭 스레드만 연산 수행
        if (i + stride < n && tid % (2 * stride) == 0) {
            input[i] += input[i + stride];
        }
        // 다음 단계로 넘어가기 전, 블록 내 모든 스레드 동기화
        __syncthreads();
    }

    // 2. 각 블록의 최종 부분 합을 벡터의 앞부분으로 수집
    if (tid == 0) {
        input[blockIdx.x] = input[i];
    }
}

```

---

## 5. 요약 및 시사점

* **동기화:** `__syncthreads()`는 이전 단계의 연산 결과가 메모리에 완전히 기록된 후 다음 단계로 넘어가게 해주는 필수 장치입니다.
* **효율성:** 위 코드는 전역 메모리(Global Memory)만 사용하여 구현이 단순하지만, 성능 최적화(공유 메모리 사용, 분기 예측 개선 등)의 여지가 많이 남아 있습니다.
