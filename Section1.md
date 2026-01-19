# 1. GPU vs CPU (very important)
- CPU
  - Powerful ALU
  - Large Cache memory per ALU
  - Good for sequential applications
- GPU
  - Thousands of small ALUs
  - Small caches per core
  - Good for parallelizd applcations
- CPU와 GPU 통신은 PCI Express로

# 2. Nvidia's history

# 3. Architecures and Generations relationshop [Hopper, Ampere, GeForce and Tesla]
- NVIDIA GPU는 크게 두 가지 카테고리로 나뉨 (Standard, HPC GPUs)
- Architecture는 Standard GPU와 HPC 모두로 만들어질 수 있음 (e.g., RTX 3090 and A100)

# 4. How to know the Architecture and Generation

# 5. The difference between the GPU and the GPU Chip
- 쉽게 말해, GPU 보드와 그 안의 칩의 차이에 대해 설명

# 6. The architectures and the corresponding chips
- GA100, GA102 등의 Chip이 탑재되는 그래픽카드 예시
- 동일 칩에서도 오버클럭을 한다던지 해서 스펙이 다르게 제공되는 경우도 있음

------------------------------------------------------------------------------------------------------------------------------

# 7. Nvidia GPU architectures From Fermi to hopper
## Fermi, 2010
- GPU Chip Example: GF100
- GPU HPC Example: Tesla X2070
- GPU N-HPC Example: RTX 480
- Double Precision에서 Tesla X2070이 GTX 480보다 약 4배의 성능으로 과학 시뮬레이션에 적합
- Pixel Rate, Texture Rate 등 Single Precision 성능을 요하는 부분에서는 RTX 480이 우세하므로 용도에 맞게 설계된 것을 알 수 있음

## Kepler, 2012
- GPU Chip Example: GK210
- GPU HPC Example: K80
- 하나의 보드에 두 개의 GK210 칩이 탑재된 것을 강조 Why?
  - 1) 2012년 당시 상황
    - 미세공정의 한계: 28nm 공정에서 단일 칩 크기를 계속 키우기에는 수율과 발열 문제가 있음
    - 이에, 단일 칩 성능 한계를 해결하고자 한 기판에 두 개의 풀스펙 칩을 넣음
    2) HPC 전용 설계
    - GTX 780 등에 탑재된 GK110에서 레지스터 파일을 두 배로 하여 데이터센터용 고성능 연산을 위해 GK210을 별도로 설계
    3) 요즘의 패러다임과 비교
    - 요즘은 공정 한계에 의한 수율과 발열 문제를 해결하기 위해 대부분 칩렛 형태를 채택함 (e.g., AMD, Rebellions REBEL-Quad)
    - 단일 칩 성능의 한계를 뛰어넘어 시스템 전체의 성능을 높이기 위한 노력의 역사(?)로 보임
    
## Maxwell, 2014
- GPU Chip Example: GM200
- GPU HPC Example: M60
- GPU N-HPC Example: RTX 980 Ti
- 데이터센터에서의 Energy 효율을 위한 클럭 스피드 조절

## Volta, 2017
- Matrix 연산을 위한 Tensor Core가 처음으로 탑재

## Ampere, 2020
- MIG (Multi-Instance GPU) 지원
- BF16 Support
- TF32 (10-bit mantissa FP32) Support
- Structured Sparsity (2:4) 지원

## Hopper, 2022
- FP8 Support
- Transformer Engine 도입 (더 자세한 공부 필요)

# 8. Parameters required to compare between different Architecures
- Memory bandwidth (memory speed + bus width)
  - A100의 bus width는 5120bit, memory type은 HBM2
  - RTX 3090의 bus width는 384bit, memory type은 GDDR6X
- The throughput TFLOPS (The core count + Speed)
  - Memory bandwidth와 비슷하게 Core 수와 Core의 Clock speed 모두 중요함
  - 전성비도 중요
- New features (supporting new data types - The tensor cores)
  - 예를 들어, Volta의 Tensor core

# 9. Half, single and double precision operations
- integer
- floating point
  - half precision (fp16)
  - single precision (fp32)
  - double precision (fp64)
    - scientific computing
  - <img width="660" height="421" alt="image" src="https://github.com/user-attachments/assets/4e8e2fe1-077b-4b02-817f-669dbb16ae3c" />

# 10. Compute capability and utilizations of the GPUs
- Compute capability
  - 예를 들어, Volta는 7.0으로 표기되고 Ampere는 8.0으로 표기됨
  - https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications
- Software capability
  - 예를 들어, Hopper는 연산 능력이 9.x 대에 해당하며, CUDA 11.8 이상의 버전이 필요함

# 11. Before reading any whitepapers
- New features
- SM 구조 변화
  - Volta vs Ampere Tensor core 수
    - Tensor core가 처음으로 탑재된 Volta에서는 SM 내에서 8개였다가 Ampere 이후로 4개로 유지
  - INT32/FP32 통합 ALU
    - 스케줄링 단순화 가능
    - 다이 면적 효율
- Performance
- Technical specifications
  - Cores, Memory, Power

# 11. Volta+Ampere+Pascal+SIMD
## Volta
- 21 billion transistors
- SM Architecture Optimized for Deep Learning
  - 50% more energy efficient than the previous generation
  - 매년 단순히, 코어 수를 늘리거나 메모리 사이즈를 키우는 것 뿐만 아니라 유닛의 성능도 끌어올림
  - Tensor Core 도입
  - independent parallel integer, floating-point data path
  - new combined L1 data cache and shared memory
- HBM2 Memory
  - 16GB
  - 900GB/s peak memory bandwidth
- Volta Optimized Software
- Extreme performance for AI and HPC
  - 125 Tensor TFLOPS of Mixed Precision
- GV100 GPU Hardware Architecture in-depth
  - 84 SMs
    - 64 FP32 cores
    - 64 INT32 cores
    - 8 Tensor cores
    - 125 TFLOPS = 640 Tensor cores x 1530 MHz x 128 (64FMA)
  - Eight 512-bit memory controllers (4096 bits total)
  - SM partition
    - L0 I-Cache
    - Warp Scheduler (SIMT)
    - Dispatcher unit
    - Registers
    - Computational units

<img width="675" height="942" alt="image" src="https://github.com/user-attachments/assets/fc0e191b-2a8d-4faf-be0d-83fe30fc1df7" />
<img width="675" height="427" alt="image" src="https://github.com/user-attachments/assets/ad83f392-d53e-42e9-a721-6614492d2686" />

- NVLink second generation
  - more links, more faster
    - Pascal 때는 gpu끼리 오로지 한 개의 connection만. Volta에서는 두 개까지 가능
    - bandwidth 향상 (160GB/s → 300GB/s)
- DGX-1
  - NVIDIA는 GPU 뿐만 아니라, CPU, DDR Memor, Power 등까지 붙여서 하나의 시스템으로 구성한 제품을 팔기도 함
