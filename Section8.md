# 40. Debugging using visual studio
## 1. NVIDIA Nsight Integration 개요

이 도구는 독립 실행형(Standalone)이 아니라 **Visual Studio에 통합**되어 작동하는 디버깅 도구입니다.

* **목적**: 코드 실행 중 특정 지점에서 멈추어(Breakpoint), 메모리 상태, 변수 값, 레지스터 상태 등을 확인하기 위함입니다.
* **특징**: 성능 분석(Profiling)보다는 **논리적 오류 수정(Debugging)** 에 초점이 맞춰져 있습니다.

## 2. 설치 및 기본 설정

* **설치 방법**:
1. Visual Studio의 `Extensions(확장)` 메뉴 -> `Manage Extensions` -> **"Nsight Integration"** 검색 후 설치.
2. 추가적으로 **"Nsight Visual Studio Edition"** 을 설치하면 레지스터, 어셈블리 코드 확인 등 더 강력한 기능을 사용할 수 있습니다.


* **실행**: 설치 후 Visual Studio를 재시작하면 상단 메뉴에 `Extensions` > `Nsight` 항목이 생깁니다.

## 3. GPU 디버깅 실습 (Vector Addition 예시)

GPU 코드를 디버깅할 때는 일반적인 CPU 디버거가 아닌 **Nsight 메뉴의 디버거**를 사용해야 합니다.

* **Breakpoint**: 코드 옆을 클릭하여 중단점을 설정합니다.
* **Start CUDA Debugging**: `Extensions` > `Nsight` > `Start CUDA Debugging`을 통해 실행합니다.
* **주요 확인 정보**:
* **Launch Details**: 현재 실행 중인 Block ID, Thread ID, Grid 크기 등을 확인.
* **Locals/Variables**: 커널 내부의 변수(예: `A`, `B`, `C`, `i`) 값을 단계별로 확인.
* **Memory View**: 특정 메모리 주소나 배열 이름을 입력하여 실제 할당된 데이터 값(16진수)을 모니터링.
* **Disassembly (어셈블리)**: C/C++ 코드가 실제 GPU에서 실행되는 어셈블리 명령어로 어떻게 변환되었는지 확인.



## 4. Visual Studio 내 프로파일링 도구 활용

별도의 복잡한 설정 없이 Visual Studio 내에서 NVIDIA의 강력한 성능 분석 도구들을 바로 호출할 수 있습니다.

* **Nsight Compute**:
* 커널의 실행 시간, 처리량(Throughput), 캐시 히트율, 파이프라인 활용도 등을 상세 분석합니다.
* **Vector Reduction**과 같은 알고리즘 최적화 시 실행 시간 비교에 필수적입니다.


* **Nsight Systems**:
* 전체 애플리케이션의 **타임라인**을 보여줍니다.
* CPU-GPU 간 데이터 복사 시간, 커널 실행 시점, 하드웨어 유닛(GPC, SM)의 활동 상태를 시각화하여 병목 현상을 찾습니다.



## 5. 요약 및 결론

| 도구 이름 | 주요 용도 | 특징 |
| --- | --- | --- |
| **Nsight Integration** | 디버깅 (Debugging) | 중단점 설정, 변수 및 레지스터 값 확인 |
| **Nsight Compute** | 커널 프로파일링 | 상세 메트릭(L1/L2, 점유율, 처리량) 분석 |
| **Nsight Systems** | 시스템 프로파일링 | CPU/GPU 전체 타임라인 및 데이터 전송 분석 |
