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
<img width="752" height="348" alt="image" src="https://github.com/user-attachments/assets/c08d9f80-f472-4cdb-b097-117d997f6fee" />

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
    - <img width="1055" height="56" alt="image" src="https://github.com/user-attachments/assets/fab90f15-1cbb-4223-b76b-714cdb8e680d" />
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
- Achived occupancy: the actual usage of the GPU's resources
