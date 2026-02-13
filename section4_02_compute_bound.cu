/**
 * 02_compute_bound.cu
 * 
 * Compute Bound 예제 - FMA 연산 집중
 * 
 * 예상 결과:
 * - Compute throughput: 높음 (60%+)
 * - Memory throughput: 낮음
 * - FMA utilization 높음
 * 
 * 01_basic과 비교해서 Speed of Light 그래프 차이 확인!
 * 
 * 실습:
 * 1. nvcc -o 02_compute 02_compute_bound.cu
 * 2. ncu --section SpeedOfLight ./02_compute
 * 3. GUI에서 Compute Workload Analysis 확인
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

// Heavy computation kernel - many FMA operations per memory access
__global__ void computeIntensive(float *input, float *output, int n, int iterations) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		float val = input[i];

		// 많은 FMA 연산 수행 (메모리 접근 1번당 연산 많이)
#pragma unroll 8
		for (int j = 0; j < iterations; j++) {
			val = val * 1.000001f + 0.000001f;  // FMA
			val = val * 0.999999f + 0.000001f;  // FMA
			val = val * 1.000001f - 0.000001f;  // FMA
			val = val * 0.999999f - 0.000001f;  // FMA
		}

		output[i] = val;
	}
}

int main(int argc, char *argv[]) {
	int n = 10 * 1024 * 1024;  // 10M elements
	int iterations = 100;  // 연산 반복 횟수

	if (argc > 1) iterations = atoi(argv[1]);

	size_t size = n * sizeof(float);
	printf("Compute Intensive: %d elements, %d iterations\n", n, iterations);
	printf("FMA operations per thread: %d\n", iterations * 4);

	float *h_input = (float*)malloc(size);
	float *h_output = (float*)malloc(size);

	for (int i = 0; i < n; i++) {
		h_input[i] = 1.0f;
	}

	float *d_input, *d_output;
	CUDA_CHECK(cudaMalloc(&d_input, size));
	CUDA_CHECK(cudaMalloc(&d_output, size));
	CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

	int blockSize = 256;
	int gridSize = (n + blockSize - 1) / blockSize;

	// Warmup
	computeIntensive<<<gridSize, blockSize>>>(d_input, d_output, n, iterations);
	CUDA_CHECK(cudaDeviceSynchronize());

	// Timed run
	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	CUDA_CHECK(cudaEventRecord(start));
	computeIntensive<<<gridSize, blockSize>>>(d_input, d_output, n, iterations);
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));

	float ms = 0;
	CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
	printf("Kernel time: %.3f ms\n", ms);

	// Calculate GFLOPS (4 FMA per iteration, 2 FLOPs per FMA)
	double flops = (double)n * iterations * 4 * 2;
	printf("GFLOPS: %.2f\n", flops / (ms * 1e6));

	CUDA_CHECK(cudaFree(d_input));
	CUDA_CHECK(cudaFree(d_output));
	free(h_input);
	free(h_output);

	return 0;
}

