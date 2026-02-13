/**
 * 03_shared_memory.cu
 * 
 * Shared Memory 사용 예제
 * 
 * 강의에서 01_basic은 shared memory 미사용 → Memory Workload에서 shared 요청 0
 * 이 예제는 shared memory 사용 → Memory Workload 차트에서 shared 요청 보임
 * 
 * 실습:
 * 1. nvcc -o 03_shared 03_shared_memory.cu
 * 2. ncu --section MemoryWorkloadAnalysis ./03_shared
 * 3. GUI에서 Memory Workload 차트의 Shared Memory 부분 확인
 * 
 * 두 버전 비교:
 * ./03_shared 0  → Global memory만 사용
 * ./03_shared 1  → Shared memory 사용
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
	do { \
		cudaError_t err = call; \
		if (err != cudaSuccess) { \
			fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
			exit(EXIT_FAILURE); \
		} \
	} while(0)

#define BLOCK_SIZE 256

// Version 0: No shared memory - repeated global memory access
__global__ void noSharedMemory(float *input, float *output, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		float sum = 0.0f;
		// 의도적으로 같은 데이터를 여러 번 global에서 읽음
		for (int j = 0; j < 10; j++) {
			sum += input[i];  // 매번 global memory 접근
		}
		output[i] = sum;
	}
}

// Version 1: With shared memory - load once, use multiple times
__global__ void withSharedMemory(float *input, float *output, int n) {
	__shared__ float sdata[BLOCK_SIZE];

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;

	// Global → Shared 한 번만 로드
	sdata[tid] = (i < n) ? input[i] : 0.0f;
	__syncthreads();

	// Shared memory에서 반복 접근 (훨씬 빠름)
	float sum = 0.0f;
	for (int j = 0; j < 10; j++) {
		sum += sdata[tid];
	}

	if (i < n) {
		output[i] = sum;
	}
}

// Version 2: Reduction using shared memory
__global__ void reductionShared(float *input, float *output, int n) {
	__shared__ float sdata[BLOCK_SIZE];

	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

	// Load and first add
	sdata[tid] = 0.0f;
	if (i < n) sdata[tid] = input[i];
	if (i + blockDim.x < n) sdata[tid] += input[i + blockDim.x];
	__syncthreads();

	// Reduction in shared memory
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		output[blockIdx.x] = sdata[0];
	}
}

int main(int argc, char *argv[]) {
	int version = 0;
	if (argc > 1) version = atoi(argv[1]);

	int n = 16 * 1024 * 1024;
	size_t size = n * sizeof(float);

	const char* names[] = {
		"No Shared Memory (repeated global access)",
		"With Shared Memory (load once, use many)",
		"Reduction with Shared Memory"
	};

	printf("Shared Memory Test: %s\n", names[version]);
	printf("Elements: %d\n", n);

	float *h_input = (float*)malloc(size);
	for (int i = 0; i < n; i++) h_input[i] = 1.0f;

	float *d_input, *d_output;
	int outputSize = (version == 2) ? ((n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2)) : n;

	CUDA_CHECK(cudaMalloc(&d_input, size));
	CUDA_CHECK(cudaMalloc(&d_output, outputSize * sizeof(float)));
	CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

	int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
	if (version == 2) gridSize = (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

	printf("Grid: %d, Block: %d\n", gridSize, BLOCK_SIZE);

	// Warmup
	switch(version) {
		case 0: noSharedMemory<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n); break;
		case 1: withSharedMemory<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n); break;
		case 2: reductionShared<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n); break;
	}
	CUDA_CHECK(cudaDeviceSynchronize());

	// Timed run
	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	CUDA_CHECK(cudaEventRecord(start));
	switch(version) {
		case 0: noSharedMemory<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n); break;
		case 1: withSharedMemory<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n); break;
		case 2: reductionShared<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n); break;
	}
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));

	float ms = 0;
	CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
	printf("Kernel time: %.3f ms\n", ms);

	printf("\n=== Nsight Compute에서 확인할 것 ===\n");
	switch(version) {
		case 0:
			printf("- Memory Workload: Shared Memory 요청 없음\n");
			printf("- Global Memory 접근만 보임\n");
			break;
		case 1:
		case 2:
			printf("- Memory Workload: Shared Memory 요청 있음\n");
			printf("- L1 → Shared 경로 활성화\n");
			printf("- Barrier stall 가능성 (syncthreads)\n");
			break;
	}

	CUDA_CHECK(cudaFree(d_input));
	CUDA_CHECK(cudaFree(d_output));
	free(h_input);

	return 0;
}

