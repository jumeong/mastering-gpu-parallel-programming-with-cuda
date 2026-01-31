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
- Theoretical occupancy: the ideal case (warp used in a kernel / max warps per SM)
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
- Achived occupancy: the actual usage of the GPU's resources
  - scenario 1: no memory or dependency
  ```bash
  # for 4 warps
  FMUL
  FMUL
  ISETP
  IMAD
  ```

  | Cycle | FP32 Units (32 Cores 가정) | 비고 |
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

  -------------------------
  
  | Cycle | FP32 Units (16 Cores) | 비고 |
  | :--- | :--- | :--- |
  | **1** | **Warp 0: FMUL1** (1/2) | Warp 0의 앞쪽 16개 스레드 처리 |
  | **2** | **Warp 0: FMUL1** (2/2) | Warp 0의 뒤쪽 16개 스레드 처리 (Warp 0 완료) |
  | **3** | **Warp 1: FMUL1** (1/2) | Warp 1의 앞쪽 16개 스레드 처리 시작 |
  | **4** | **Warp 1: FMUL1** (2/2) | Warp 1의 뒤쪽 16개 스레드 처리 완료 |
  | **5** | **Warp 2: FMUL1** (1/2) | Warp 2 처리 시작 |
  | **6** | **Warp 2: FMUL1** (2/2) | Warp 2 처리 완료 |
  | **7** | **Warp 3: FMUL1** (1/2) | Warp 3 처리 시작 |
  | **8** | **Warp 3: FMUL1** (2/2) | Warp 3 처리 완료 |
  | **9** | **Warp 0: FMUL2** (1/2) | Warp 0의 두 번째 명령어 시작 |
  | **10** | **Warp 0: FMUL2** (2/2) | Warp 0의 두 번째 명령어 완료 |
  | **11** | **Warp 1: FMUL2** (1/2) | Warp 1의 두 번째 명령어 시작 |
  | **12** | **Warp 1: FMUL2** (2/2) | Warp 1의 두 번째 명령어 완료 |
  | **13** | **Warp 2: FMUL2** (1/2) | Warp 2의 두 번째 명령어 시작 |
  | **14** | **Warp 2: FMUL2** (2/2) | Warp 2의 두 번째 명령어 완료 |
  | **15** | **Warp 3: FMUL2** (1/2) | Warp 3의 두 번째 명령어 시작 |
  | **16** | **Warp 3: FMUL2** (2/2) | Warp 3의 두 번째 명령어 완료 |
    
  - scenario 2: memory request, 1 inst. dependency
  ```bash
  # for 4 warps
  FMUL
  ISETP
  LDG.E.SYS
  IMAD (dependent w/ LDG)
  ```

  | Cycle | FP32 Units (32 Cores 가정) | 비고 |
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
