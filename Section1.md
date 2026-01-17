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
