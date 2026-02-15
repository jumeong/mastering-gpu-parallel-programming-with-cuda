# 38. The shared memory

## 1. GPU 메모리 계층 구조 요약

GPU의 메모리는 접근 범위와 속도에 따라 여러 단계로 나뉩니다.

* **Global Memory (전역 메모리):** 모든 스레드와 CPU가 접근 가능. 용량이 크지만 지연 시간(Latency)이 가장 김 (**300+ cycles**).
* **L2 Cache:** 모든 SM(Streaming Multiprocessor)이 공유. Global Memory보다 빠름 (**~200 cycles**).
* **L1 Cache & Shared Memory:** SM 내부에 위치. 매우 빠름 (**L1: ~33 cycles, Shared: ~25 cycles**).
* **Registers (레지스터):** 각 스레드별 개별 공간. 가장 빠름.

---

## 2. Shared Memory의 특징 (Software Cache)

Shared Memory는 L1 캐시와 물리적으로 같은 위치에 있지만, 동작 방식이 다릅니다.

* **가시성:** 같은 블록(Block) 내의 스레드들끼리 데이터를 공유하고 통신할 수 있음.
* **프로그래밍 가능:** 개발자가 직접 변수를 할당(`__shared__`)하고 제어할 수 있는 "Software Cache"임. (L1은 하드웨어가 자동으로 관리)
* **용도:** 데이터 재사용(Data Reuse)을 통해 Global Memory 트래픽을 줄이고 성능 병목 현상을 해결함.
* **속도:** L1(33 cycles)보다 Shared Memory(25 cycles)가 더 빠른데, 이는 L1 캐시가 수행하는 태그 검색(Tag Search) 등의 하드웨어 로직이 Shared Memory에는 필요 없기 때문임.

### [물리적 통합 구조: Ampere(A100) 아키텍처 예시]

최신 아키텍처에서는 L1과 Shared Memory가 하나의 물리적 단위(예: 192KB)로 통합되어 있습니다.

* **가변 설정:** 컴파일러나 설정을 통해 L1과 Shared Memory의 비율을 조정할 수 있음. (예: Shared 100KB 설정 시 L1은 92KB 사용)

---

## 3. Shared Memory의 하드웨어 구조: Banks (뱅크)

Shared Memory는 효율적인 병렬 접근을 위해 **Bank**라는 단위로 나뉩니다.

* **Cache Line (Row):** 한 행은 128 Bytes.
* **Bank (Column):** 한 행은 **32개의 뱅크**로 나뉨. (128B / 32 = **4 Bytes당 1개 뱅크**)
* **접근 규칙:** 사이클당 각 뱅크에서 4바이트씩 접근 가능. 32개 뱅크가 서로 다르면 한 사이클에 128바이트를 동시에 읽을 수 있음.

---

## 4. Bank Conflict (뱅크 충돌)

성능 저하의 주요 원인인 뱅크 충돌을 이해하는 것이 중요합니다.

* **정의:** 여러 스레드가 **동일한 사이클에 동일한 뱅크(열) 내의 서로 다른 데이터**에 접근하려고 할 때 발생.
* **결과:** 충돌이 발생하면 접근이 직렬화(Serialized)되어 완료될 때까지 더 많은 사이클이 소모됨.
* **Broadcasting (브로드캐스팅):** 여러 스레드가 동일한 뱅크의 **"정확히 같은 데이터"**를 읽을 때는 충돌 없이 한 번에 처리됨.

### [충돌 예시 및 계산]

1. **연속 접근 (No Conflict):** Thread 0 -> Bank 0, Thread 1 -> Bank 1... (32개 뱅크 모두 다름) → **1 Cycle**
2. **Stride 접근 (Conflict 발생):** 16-byte stride로 접근 시, 여러 스레드가 동일한 열(Bank)에 배치됨.
* 한 열에 4개의 스레드가 몰린다면: **4 Cycles** 소요.


3. **Double Precision (8-byte):** 각 스레드가 8바이트를 읽으므로 기본적으로 2개의 뱅크를 점유함. 따라서 최소 2 사이클이 필요할 수 있음.

---

## 5. 요약 및 결론

* **Shared Memory**는 데이터 재사용이 빈번한 알고리즘(예: Vector Reduction)에서 성능을 최적화하는 핵심 도구입니다.
* **Bank Conflict**를 피하기 위해 스레드가 뱅크를 골고루 점유하도록 데이터 접근 패턴(Stride)을 설계해야 합니다.
* 성능 분석 시 **Nsight Compute**와 같은 툴을 활용하여 실제 뱅크 충돌 메트릭을 확인하는 것이 좋습니다.

