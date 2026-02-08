# 25. Query the device properties using the Runtime APIs
## The runtime APIs
- High-level interface to CUDA
- Harness the power of NVIDIA GPUs
- Managing the GPU devices, memory allocation, and execution of parallel kernels
- 대부분의 API는 cudaError_t 구조체를 return히고, 이외에 return할 값이 있으면 포인터 입출력을 통해 반환함
- https://docs.nvidia.com/cuda/cuda-runtime-api/index.html

## cudaGetDeviceCount()
  - 현재 시스템에서 nvidia gpu가 몇 개인지 조회
## cudaGetDeviceProperties()
  - cudaDeviceProp 구조체의 name, memoryClockRate, regsPerBlock, regsPerMultiprocessor, totalGlobalMem, multiProcessorCount 등 여러가지 파라미터를 조회할 수 있음

# 26. Nvidia-smi and its configurations (Linux User)
## nvidia-smi (NVIDIA System Management Interface)
```bash
/content# nvidia-smi
Fri Jan 30 15:32:55 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   38C    P8              9W /   70W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

- Performance monitoring
  - Utilization, memory usage, temperature and power
- Settings management
  - Controlling the clock speed and power limits
- Device information querying
  - GPU name, driver version
- nvidia-smi에서의 CUDA Version과 nvcc에서의 CUDA Version는 다를 수 있음!!!
  - 강사의 환경에서는 두 버전이 같았고, 의미하는 바가 다른 것을 언급하지 않음. 
  - nvidia-smi = “이 GPU 드라이버로 어디까지 실행 가능?”
  - nvcc = “나는 지금 어떤 CUDA로 컴파일 중?”

- Various Options
  - Monitoring GPUs Continuously
    - command: nvidia-smi -l 5
    - 사용자가 끊기 전까지 5초 간격으로 nvidia-smi를 반복 출력
  - Displaying Specific Information
    - command: nvidia-smi --query-gpu=gpu_name,driver_version,temperature.gpu --format=csv
    - csv 파일 형태로 입력한 파라미터를 출력
  - Setting Power Limits
    - command: nvidia-smi -i 0 -pl 150
    - 입력한 값으로 Power를 제한
    - ```bash
      /content# nvidia-smi -i 0 -pl 150
      Provided power limit 150.00 W is not a valid power limit which should be between 60.00 W and 70.00 W for GPU 00000000:00:04.0
      Terminating early due to previous errors.
      ```
  - Persistence mode
    - 강사의 설명이 Permission mode로 Clock이나 Power를 조절할 때, 이것에 따라 좌우될 수 있다고 하는데 틀린 설명인듯...
    - Gemini의 설명
      - **Persistence Mode가 필요한 이유**

        보통 리눅스 환경에서 GPU를 사용하지 않을 때는 전력 소모를 줄이기 위해 드라이버가 언로드(Unload)되거나 GPU가 휴면 상태로 들어갑니다. 그런데 다시 작업을 시작하려고 하면 드라이버를 다시 불러오는 과정에서 **수 초 정도의 지연(Latency)**이 발생하게 되죠.
        
        pm 옵션은 이 과정을 생략하고 GPU를 항상 대기 상태로 유지하게 만듭니다.
        
        **주요 특징 및 장점**
        응답 속도 향상: 드라이버 로딩 오버헤드가 없어져서 커맨드 실행이나 작업 시작이 즉각적입니다.
        
        상태 유지: GPU의 설정값들이 작업 사이사이에 초기화되지 않도록 도와줍니다.
        
        안정성: 빈번하게 드라이버가 로드/언로드되면서 생길 수 있는 미세한 오류를 방지합니다.
 
        **사용 방법**

        터미널에서 다음과 같은 명령어로 설정하거나 확인할 수 있습니다.
        
        현재 상태 확인: nvidia-smi -q | grep "Persistence Mode"
        
        모드 활성화 (Enable): nvidia-smi -pm 1
        
        모드 비활성화 (Disable): nvidia-smi -pm 0
        
        참고: 최근 최신 드라이버나 데이터센터용 GPU(A100, H100 등)에서는 이 기능이 Persistence Daemon으로 대체되거나 기본적으로 활성화되어 있는 경우가 많습니다. 개인용 PC(RTX 시리즈)보다는 서버 환경에서 훨씬 중요하게 다뤄지는 옵션이에요.
  - -d 옵션
    - -d,   --display=          Display only selected information: MEMORY,
                                UTILIZATION, ECC, TEMPERATURE, POWER, CLOCK,
                                COMPUTE, PIDS, PERFORMANCE, SUPPORTED_CLOCKS,
                                PAGE_RETIREMENT, ACCOUNTING, ENCODER_STATS,
                                SUPPORTED_GPU_TARGET_TEMP, VOLTAGE, FBC_STATS
                                ROW_REMAPPER, RESET_STATUS, GSP_FIRMWARE_VERSION
    - curernt clock, supported clock과 같이 특정 부분을 집어서 출력 가능
   
# 27. The GPU's Occupancy and Latency hiding
## cudaDeviceGetAttribute()
- 각종 Attribute를 지정해서 Fetch할 수 있음
- e.g., cudaDeviceGetAttribute(&maxThreadsPerMP, cudaDevAttrMaxThreadsPerMultiProcessor, device)
  
## Occupancy
- Occupancy is a measure of the utilization of the resources in a GPU
- Theoretical occupancy: the ideal case (active warps per SM / maximum warps per SM)
  👉 정적 값 (launch configuration으로 결정)
  - Optimal conditions where there are enough independent tasks.
  - 강의에서는 max warps per SM이 48
  - kernel의 Block Size를 32에서 64로 변경하면서 Theroetical occupancy가 두배가 되는 것을 보여줌
  - 이 계산을 할때, SM, Registers, Shared Mem, Warps 등 Block 수를 제한하는 여러 요소에 의해 계산된 것 중 최소로 계산해야 함.
  - ```bash
    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block            8
    Block Limit Shared Mem                block           16
    Block Limit Warps                     block            2
    Theoretical Active Warps per SM        warp           32
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        82.32
    Achieved Active Warps Per SM           warp        26.34
    ------------------------------- ----------- ------------
    ```
- Achived occupancy: average active warps per cycle / max warps per SM
  👉 동적 값 (실행 중 실제로 관측)
  - scenario 1: no memory or dependency
  ```bash
  # for 4 warps
  FMUL
  FMUL
  ISETP
  IMAD
  ```

  | Cycle | FP32 Units 32 Cores | 비고 |
  | :--- | :--- | :--- |
  | **1** | **Warp 0: FMUL1** | Warp 0의 첫 번째 FMUL (32스레드 동시 처리) |
  | **2** | **Warp 1: FMUL1** | Warp 1의 첫 번째 FMUL |
  | **3** | **Warp 2: FMUL1** | Warp 2의 첫 번째 FMUL |
  | **4** | **Warp 3: FMUL1** | Warp 3의 첫 번째 FMUL |
  | **5** | **Warp 0: FMUL2** | Warp 0의 두 번째 FMUL |
  | **6** | **Warp 1: FMUL2** | Warp 1의 두 번째 FMUL |
  | **7** | **Warp 2: FMUL2** | Warp 2의 두 번째 FMUL |
  | **8** | **Warp 3: FMUL2** | Warp 3의 두 번째 FMUL |
  | **9** | **Warp 0: ISETP** | Warp 0의 비교 연산 (Condition Check) |
  | **10** | **Warp 1: ISETP** | Warp 1의 비교 연산 |
  | **11** | **Warp 2: ISETP** | Warp 2의 비교 연산 |
  | **12** | **Warp 3: ISETP** | Warp 3의 비교 연산 |
  | **13** | **Warp 0: IMAD** | Warp 0의 정수 곱셈-가산 (Integer Multiply-Add) |
  | **14** | **Warp 1: IMAD** | Warp 1의 정수 곱셈-가산 |
  | **15** | **Warp 2: IMAD** | Warp 2의 정수 곱셈-가산 |
  | **16** | **Warp 3: IMAD** | Warp 3의 정수 곱셈-가산 |
    
  - scenario 2: memory request, 1 inst. dependency
  ```bash
  # for 4 warps
  FMUL
  ISETP
  LDG.E.SYS
  IMAD (dependent w/ LDG)
  ```

  | Cycle | FP32 Units 32 Cores | 비고 |
  | :--- | :--- | :--- |
  | **1** | **Warp 0: FMUL** | Warp 0 시작 |
  | **2** | **Warp 1: FMUL** | Warp 1 시작 |
  | **3** | **Warp 2: FMUL** | Warp 2 시작 |
  | **4** | **Warp 3: FMUL** | Warp 3 시작 |
  | **5** | **Warp 0: ISETP** | Warp 0 비교 연산 |
  | **6** | **Warp 1: ISETP** | Warp 1 비교 연산 |
  | **7** | **Warp 2: ISETP** | Warp 2 비교 연산 |
  | **8** | **Warp 3: ISETP** | Warp 3 비교 연산 |
  | **9** | **Warp 0: LDG** | Warp 0 메모리 요청 (Stall 시작) |
  | **10** | **Warp 1: LDG** | Warp 1 메모리 요청 |
  | **11** | **Warp 2: LDG** | Warp 2 메모리 요청 |
  | **12** | **Warp 3: LDG** | Warp 3 메모리 요청 |
  | **13** | (Memory Waiting) | Warp 0 데이터 아직 안 옴 (Stall) |
  | **14** | (Memory Waiting) | Warp 1 데이터 아직 안 옴 |
  | **15** | **Warp 0: IMAD** | **Warp 0 데이터 도착!** (IMAD 실행) |
  | **16** | **Warp 1: IMAD** | **Warp 1 데이터 도착!** (IMAD 실행) |

- Summary
  - High occupancy doesn't always equate to high perforamnce
  - Identifying and understanding occupancy can help us pinpoint performance issues.
  - Low occupancy, on the other hand, suggests that there's a bottleneck preventing the GPU from being fully utilized.
 
## Latency Hiding
- CPU에서는 dependency를 조사해서 기다릴 필요가 없는 Instruction이 뒤쪽에 있으면 순서를 바꿔서 실행해버리기도 함 (Out of Order)
- DSP에서는 보통 scratch pad memory를 두기 때문에 연산기 뿐만 아니라 Load/Store 명령어까지 Cycle이 Static함. 그래서, OoO보다는 Compile 단계에서 VLIW로 여러 개의 명령어를 묶어버려서 Latency Hiding을 함.
- GPU는 dependency 때문에 stall되는 warp가 생기면 다른 warp로 context switching해버린다는 철학

# 28. Allocated active blocks per SM
- 동시에 실행가능한 block 수
- 여러가지 HW 자원에 의해 계산되는 block 수 중 minimum

## Max Thread blocks/SM
- A100 기준 32개
  
## Max warps/SM
- A100 기준 64개
- Thread Block의 Size에 따라 몇개의 Block이 SM에 할당될지는 달라질 수 있음
- 예를 들어, 한 개 Block이 128 Threads (4 Warps)로 구성된다면 이 SM에는 16개 Block만 할당 가능
- 한 개 Block이 64 Threads (2 Warps)로 구성된다면 32개 Block으로 구성 가능

## Max registers/SM
- A100 기준 64K개
- 1024 Threads로 구성된 1 Block을 가정, Each thread requires 100 registers.
- thread 수를 줄이지 않으면 register spilling 발생!
- register spilling이 발생하면 register가 모자라므로 local memory까지 끌어쓰게 됨. local memory는 register에 비해 latency가 느리므로 performance degradation

## Shared Memory/SM
- A100 기준 up to 164KB

# 29. Starting with the nsight compute

# 30. All profiling tools from NVidia (Nsight systems - compute - nvprof ...)
- CUDA-MEMCHECK
  - Identify and diagnose memory errors in CUDA applications
- CUDA-GDB
- NVIDIA Visual Profiler (nvvp)
  - Detailed timing info and hw counters for CUDA, OpenCL, Direct3D...
  - Graphical view of the applications timelines and achieved occupancy
- NVIDIA Nsight Systems
  - Comprehensive workload level performance
- NVIDIA Nsight Compute
  - Dive into top CUDA kernels by using metrics/counter collection
- NVIDIA Nsight Graphics
  - Detailed frame/render performance

# 31. Error Checking APIs
- Checking erros to ensure Cuda functions operate smoothly.
- Example
  - Application compiles correctly, but fails to execute properly
  - Malfunctioning malloc because no enough space in the memory
- Two catergories
  - Synchronous
  - Asynchronous 
- Usage
  ```cpp
  cudaError_t err = cudaMalloc((void **)&d_A, size);
  if (err != cudaSuccess) {
      fprintf(stderr, "Failed to allocated device memory %s\n", cudaGetErrorString(err));
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      fprintf(stderr, "Kernel launch failed %s\n", cudaGetErrorString(err));
  }
  ```

# 32. Nsight Compute performance using command line analysis
## 두 가지 분석 방법

| 방법 | 특징 | 사용 시기 |
| --- | --- | --- |
| **CLI (Command Line)** | 특정 메트릭만 빠르게 수집 | 몇 가지 수치만 확인할 때 |
| **GUI (Graphical)** | Roofline 분석, 차트, 상세 시각화 | 심층 성능 분석할 때 |

**기본 명령어**
```basah
ncu ./my_cuda_app                    # 기본 4개 섹션 출력
ncu -o profile ./my_cuda_app         # 결과를 파일로 저장
```

## Sections (섹션)

기본 실행 시 4개 섹션만 표시되지만, **총 23개 섹션**이 존재함

### 기본 4개 섹션
| 섹션 | 내용 |
| --- | --- |
| **GPU Speed of Light** | DRAM, L1/L2 캐시 throughput, SM utilization |
| **Launch Statistics** | block size, grid size, registers/thread, shared memory |
| **Occupancy** | theoretical vs achieved occupancy |
| **Memory Workload** | DRAM, L1, L2의 active cycles |

### 특정 섹션만 보기
```bash
# Launch Statistics만 보기
ncu --section LaunchStats ./my_cuda_app

# Warp State Statistics 보기
ncu --section WarpStateStats ./my_cuda_app
```

### 주요 섹션 목록
| Identifier | 설명 |
| --- | --- |
| `SpeedOfLight` | GPU 전체 throughput |
| `LaunchStats` | 커널 런치 정보 |
| `Occupancy` | Warp occupancy |
| `MemoryWorkloadAnalysis` | 메모리 워크로드 상세 |
| `WarpStateStats` | Warp 상태 통계 |
| `SchedulerStats` | 스케줄러 통계 |
| `SourceCounters` | 소스 레벨 카운터 |
| `NVLink` | NVLink 통신 분석 |


## Metrics (메트릭)

> 💡 Nsight Compute에는 약 10만 개의 메트릭이 있음
> 참고 : https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metric-collection

### 메트릭 전체 목록 보기

```bash
ncu --query-metrics-mode all > metrics.txt
```

### 메트릭 명명 규칙

```jsx
[하드웨어 유닛]__[메트릭명].[suffix]
```

**예시**: `dram__bytes.avg`

- `dram` = 하드웨어 유닛 (DRAM)
- `bytes` = 메트릭 (바이트 수)
- `avg` = suffix (평균값)

### 하드웨어 유닛

| 접두어 | 하드웨어 |
| --- | --- |
| `dram` | DRAM (글로벌 메모리) |
| `l1tex` | L1 텍스처 캐시 |
| `lts` | L2 캐시 |
| `sm` | Streaming Multiprocessor |
| `smsp` | SM 내 파티션 (SM의 1/4) |
| `gpu` | GPU 전체 |

### Suffix (접미사)

| Suffix | 의미 |
| --- | --- |
| `.min` | 모든 SM 중 최소값 |
| `.max` | 모든 SM 중 최대값 |
| `.avg` | 모든 SM의 평균값 |
| `.sum` | 전체 GPU 합계 (= max × SM 개수) |

**예시**

- 100개 SM이 있고 L2 캐시 사용 사이클이 SM마다 다를 때:
    - `.min` = 가장 적게 사용한 SM의 값
    - `.max` = 가장 많이 사용한 SM의 값
    - `.avg` = 전체 평균
    - `.sum` = 전체 합계
 
## 실전 사용법

### 특정 메트릭 수집

```bash
# L1 캐시 hit rate
ncu --metrics l1tex__t_sector_hit_rate ./my_app

# 여러 메트릭 동시 수집 (쉼표로 구분)
ncu --metrics l1tex__t_sector_hit_rate,lts__t_sector_hit_rate ./my_app

# suffix 생략하면 모든 suffix 수집
ncu --metrics sm__inst_executed ./my_app
# → sm__inst_executed.avg, .max, .min, .sum 모두 출력

```

### CSV로 내보내기

```bash
ncu --metrics sm__inst_executed --csv ./my_app > output.csv

```

### 특정 하드웨어 유닛의 모든 메트릭 수집

```bash
# shared memory 관련 모든 메트릭
ncu --metrics regex:.*shared.* ./my_app --csv > shared_metrics.csv

# L1 캐시 관련 모든 메트릭
ncu --metrics regex:.*l1tex.* ./my_app
```

---

## 핵심 메트릭 예시

### 캐시 성능

```bash
# L1 hit rate (0%면 문제!)
ncu --metrics l1tex__t_sector_hit_rate ./my_app

# L2 hit rate
ncu --metrics lts__t_sector_hit_rate ./my_app

```

> ⚠️ L1 hit rate가 0%면 모든 메모리 연산이 글로벌 메모리에서 읽는 것
> → 수백 사이클 vs L1 히트 시 ~30 사이클

### 명령어 실행

```bash
# SM당 실행된 명령어 수
ncu --metrics sm__inst_executed ./my_app

# FP64 (double precision) 명령어
ncu --metrics sm__inst_executed_pipe_fp64 ./my_app

# FP16 (half precision) 명령어
ncu --metrics sm__inst_executed_pipe_fp16 ./my_app
```

### Warp 상태

```bash
ncu --section WarpStateStats ./my_app
```

- `warp_cycles_per_issued_instruction` - 명령어당 warp 사이클
- `active threads per warp` - warp당 활성 스레드 (이상적: 32)

## 분석 팁

### .sum 계산 방식

```jsx
.sum = .max × SM 개수

```

**검증 예시** (RTX 3060, 38 SM):

```jsx
sm__inst_executed.sum / sm__inst_executed.avg ≈ 38

```

### Nsight Compute가 주는 조언

실행 결과에 자동으로 분석/경고가 포함됨:

```jsx
The local speedup is 93%, which is good.
On average each warp stalled for 111 cycles due to scoreboard dependency.

```

→ 이런 메시지를 읽고 병목 파악

---

## Quick Reference

| 명령어 | 용도 |
| --- | --- |
| `ncu ./app` | 기본 4개 섹션 분석 |
| `ncu --section <name> ./app` | 특정 섹션만 |
| `ncu --metrics <metric> ./app` | 특정 메트릭 수집 |
| `ncu --metrics regex:.*<pattern>.* ./app` | 패턴 매칭 메트릭 |
| `ncu --csv ./app > out.csv` | CSV 출력 |
| `ncu --query-metrics-mode all` | 전체 메트릭 목록 |
| `ncu -o profile ./app` | 결과 파일 저장 (GUI에서 열기) |

---

## 핵심 포인트

**섹션 vs 메트릭**
- 섹션: 관련 메트릭들의 그룹 (예: Launch Statistics)
- 메트릭: 개별 측정값 (예: block size, register count)

**10만 개 메트릭?**

- 실제로 다 볼 필요 없음
- 명명 규칙만 알면 1분에 100개 메트릭 파악 가능
- 하드웨어 유닛 + 메트릭명 + suffix 구조

**실전에서 자주 보는 것**

- L1/L2 hit rate → 캐시 효율
- SM utilization → GPU 활용도
- Occupancy → Warp 스케줄링 효율
- inst_executed → 실제 실행된 명령어

### 데모 시나리오

- `ncu ./vector_add` 실행해서 기본 4개 섹션 보여주기
- L1 hit rate 0% 나오는 거 보여주기 → "이건 문제다"
- `-csv`로 Excel에서 열어보기

### 다음 강의 예고

- block/thread 수 변경이 실행 시간에 미치는 영향 분석
- GUI 분석 상세 설명

## 실습 예제

CLI 분석 연습용 예제:

### 01_vector_add_[basic.cu](http://basic.cu/)

기본 Memory Bound 커널. CLI 사용법 익히기에 적합.

```bash
# 기본 4개 섹션 확인
ncu ./01_basic

# 특정 섹션만
ncu --section LaunchStats ./01_basic
ncu --section SpeedOfLight ./01_basic

# 특정 메트릭
ncu --metrics l1tex__t_sector_hit_rate,lts__t_sector_hit_rate ./01_basic

# CSV 출력
ncu --metrics dram__bytes.sum --csv ./01_basic > bandwidth.csv

# GUI용 파일 저장
ncu -o 01_basic_profile ./01_basic

```

### 05_error_[cases.cu](http://cases.cu/)

의도적으로 에러를 발생시키는 6가지 케이스. ncu가 에러를 어떻게 보고하는지 확인:

```bash
# 각 케이스별로 실행
./05_error 1  # Invalid grid size
./05_error 2  # Invalid block size
./05_error 3  # Too many threads
./05_error 4  # Out of memory
./05_error 5  # Invalid device
./05_error 6  # Kernel timeout

# ncu로 프로파일링 시도 (에러 메시지 확인)
ncu ./05_error 3
```

# 33. Graphical Nsight Compute (windows and linux)
# Nsight Compute GUI 분석

NVIDIA Nsight Compute 그래픽 인터페이스를 사용한 심층 성능 분석

---

## CLI vs GUI

| 방법 | 장점 | 단점 |
| --- | --- | --- |
| **CLI** | 빠른 메트릭 수집, 스크립트 자동화 | 시각화 없음 |
| **GUI** | 그래프/차트, 의존성 시각화, 조언 제공 | 설정 필요 |

> 💡 둘 다 같은 메트릭을 수집하지만, GUI는 시각화와 자동 분석/조언이 핵심 차별점
> 

---

## 설치 및 실행

**설치**

- CUDA Toolkit 설치 시 자동 포함
- 별도 설치: NVIDIA 웹사이트에서 "Nsight Compute" 다운로드

**프로파일링 시작**

```bash
# 1. 실행파일 컴파일
nvcc -o my_app.exe my_kernel.cu

# 2. 프로세스를 일시정지 상태로 시작
ncu --mode launch ./my_app.exe

# 3. GUI에서 Attach → 프로세스 선택 → Profile Kernel

```

**GUI 워크플로우**

1. File → New Project
2. Application Executable 경로 설정
3. Working Directory 설정
4. 프로세스 Attach
5. Metrics Selection에서 분석할 섹션 선택
6. Profile Kernel 클릭

## Metrics Selection

프로파일링 전에 수집할 섹션 선택:

| 섹션 | 내용 |
| --- | --- |
| **Speed of Light Throughput** | SM/메모리 throughput |
| **Roofline Chart** | Compute vs Memory bound 시각화 |
| **Compute Workload Analysis** | 연산 유닛별 활용도 |
| **Memory Workload Analysis** | 메모리 계층 간 데이터 흐름 |
| **Scheduler Statistics** | Warp 스케줄링 통계 |
| **Warp State Statistics** | Warp stall 원인 분석 |
| **Instruction Statistics** | 명령어별 실행 횟수 |
| **Occupancy** | Warp occupancy 분석 |

---

## 핵심 분석 화면

### 1. Summary 탭

기본 정보 요약:

- Achieved Occupancy (예: 81%)
- Theoretical Occupancy (예: 100%)
- 주요 병목 요약

### 2. Details 탭 (가장 중요)

View → **Expand Sections**로 그래프 활성화

---

## GPU Speed of Light

**Compute vs Memory Bound 판단**

```jsx
Compute Throughput: 16%  ← 낮음
Memory Throughput:  95%  ← 높음

```

→ **Memory Bound** 애플리케이션

**해석**

- Memory 95%: 대부분의 시간을 메모리 연산에 사용
- Compute 16%: ALU를 거의 활용하지 못함
- 목표: Memory throughput ↓, Compute throughput ↑

---

## Memory Workload Analysis

메모리 계층 간 데이터 흐름 시각화:

```jsx
[SM] → 3.15M requests → [L1 Cache] → [L2 Cache] → [DRAM]
                         Hit: 0%      Hit: 33%
이거 실제그림으로 바꾸면 좋을듯...
```

**차트 색상 의미**

- 🟢 밝은색: 높은 활용도 (peak에 가까움)
- 🔴 어두운색: 낮은 활용도

**데이터 전송량**

- L2 → L1: 268MB (읽기: vector A, B)
- L1 → L2: 134MB (쓰기: vector C)
- 읽기가 쓰기의 2배 = 2개 읽고 1개 씀

**문제 진단**

- L1 hit rate 0% → 모든 요청이 L2 이상으로 감
- L2 hit rate 33% → 2/3가 DRAM까지 감
- Memory bandwidth 95% → DRAM 접근 과다

## Compute Workload Analysis

연산 유닛별 활용도:

| 유닛 | Active Cycles % | Peak Instructions % |
| --- | --- | --- |
| **Load/Store** | - | 16% ← 가장 높음 |
| **FMA** (Fused Multiply-Add) | 3.55% | - |
| **ALU** (Int, FP32, FP16 등) | 4% | - |
| **FP64** | 0% | - |
| **Tensor** | 0% | - |

→ Load/Store가 지배적 = **Memory Bound 확인**

---

## Warp State Statistics

**Stall 원인 분석**

```jsx
Warp Cycles per Issued Instruction: 119 cycles
Stall Long Scoreboard: 111 cycles (93%)

```

**해석**

- 매 명령어 발행마다 warp가 평균 119 사이클 대기
- 111 사이클은 **scoreboard dependency** 때문
- Scoreboard dependency = 이전 메모리 로드 결과를 기다림

**Nsight Compute 조언 예시**

> "On average each warp stalled for 111 cycles waiting for scoreboard dependency on L1 texture cache"

→ 메모리 지연이 stall의 주원인

## Source 탭: 어셈블리 분석

CUDA 코드와 SASS (어셈블리) 매핑:

```c
// CUDA 코드
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) {
    C[i] = A[i] + B[i];
}

```

```jsx
// SASS 어셈블리
S2R R0, SR_TID.X      // threadIdx.x 읽기
S2R R1, SR_CTAID.X    // blockIdx.x 읽기
IMAD R6, R1, R2, R0   // i = blockIdx.x * blockDim.x + threadIdx.x
LDG R4, [R8]          // A[i] 로드
LDG R3, [R10]         // B[i] 로드
FADD R9, R4, R3       // A[i] + B[i]  ← 93% stall 원인!
STG [R12], R9         // C[i] 저장

```

**의존성 시각화**

- `FADD R9, R4, R3`는 R4(A[i])와 R3(B[i])에 의존
- R4는 `LDG R4` 완료를 기다려야 함
- R3는 `LDG R3` 완료를 기다려야 함
- → **Load가 끝날 때까지 Add 불가능**

**GUI에서 확인**

- ⚠️ 아이콘: stall 원인 명령어
- 삼각형 화살표: 의존성 방향 표시
- 마우스 오버: "This line is responsible for 84% of all warp stalls"

## Occupancy 분석

**현재 상태**

- Theoretical: 100%
- Achieved: 81%

**Occupancy Calculator 그래프**

Registers per Thread:

```jsx
현재: 16 registers → 48% occupancy
40 registers까지 증가해도 occupancy 유지
40+ registers → occupancy 감소 시작

```

Block Size:

```jsx
현재: 96 threads/block
128까지 증가 → 영향 없음
224+ → occupancy 감소 시작
800+ → 심각한 감소 (48% → 24%)
```

Shared Memory:

```jsx
현재: 0 bytes (shared memory 미사용)
증가 시 → occupancy 감소
```

---

## API Statistics 탭

CUDA Runtime API별 소요 시간:

| API | 시간 | 설명 |
| --- | --- | --- |
| `cudaMemcpy` | 40ms | CPU↔GPU 데이터 전송 |
| `cudaMalloc` | 7ms | GPU 메모리 할당 |
| `cudaFree` | - | GPU 메모리 해제 |
| `cudaLaunchKernel` | - | 커널 실행 |

→ 커널 실행 시간 외에 **데이터 전송 오버헤드** 파악 가능

## 실습 예제

각 분석 개념을 실습할 수 있는 예제 코드:

| 예제 | 학습 목표 | 관련 섹션 |
| --- | --- | --- |
| `01_vector_add_basic.cu` | Memory Bound 커널 분석 | GPU Speed of Light, Memory Workload |
| `02_compute_bound.cu` | Compute Bound 커널 분석 (FMA 집약) | Compute Workload Analysis |
| `03_shared_memory.cu` | Global vs Shared Memory 캐시 효율 비교 | Memory Workload, L1/L2 Hit Rate |
| `04_occupancy_test.cu` | Block Size, Register 수가 Occupancy에 미치는 영향 | Occupancy Calculator |
| `06_warp_stall.cu` | Scoreboard, Branch Divergence, Barrier Stall 원인 분석 | Warp State Statistics, Source 탭 |
| `07_memory_coalescing.cu` | Coalesced vs Strided Access 패턴 비교 | Memory Workload, DRAM Throughput |

### 실습 순서 권장

```bash
# 1. Memory Bound 기본 (GUI 익숙해지기)
ncu -o 01_basic ./01_basic
# → Speed of Light에서 Memory > Compute 확인

# 2. Compute Bound 비교
ncu -o 02_compute ./02_compute
# → Speed of Light에서 Compute > Memory 확인

# 3. Shared Memory 효과
ncu -o 03_shared ./03_shared
# → L1 Hit Rate 비교 (global vs shared)

# 4. Occupancy 실험
ncu --section Occupancy ./04_occupancy 64   # block size 64
ncu --section Occupancy ./04_occupancy 256  # block size 256
ncu --section Occupancy ./04_occupancy 1024 # block size 1024
# → Occupancy Calculator 그래프와 비교

# 5. Warp Stall 분석
ncu --section WarpStateStats -o 06_stall ./06_warp_stall
# → Source 탭에서 stall 원인 명령어 확인

# 6. Memory Coalescing
ncu -o 07_coalesced ./07_coalescing coalesced
ncu -o 07_strided ./07_coalescing strided
# → DRAM Throughput 비교

```

---
