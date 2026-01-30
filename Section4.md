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
- nvidia-smi에서의 CUDA Version과 nvcc에서의 CUDA Version이 다른 이유

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
    - /content# nvidia-smi -i 0 -pl 150
      Provided power limit 150.00 W is not a valid power limit which should be between 60.00 W and 70.00 W for GPU 00000000:00:04.0
      Terminating early due to previous errors.
