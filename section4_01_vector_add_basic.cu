/**
 * 01_vector_add_basic.cu
 * 
 * 기본 Vector Addition - Memory Bound 예제
 * 
 * 예상 결과:
 * - L1 hit rate: ~0% (매우 낮음)
 * - L2 hit rate: ~33%
 * - Memory throughput: ~95% (높음)
 * - Compute throughput: ~16% (낮음)
 * → Memory Bound 애플리케이션
 * 
 * 실습:
 * 1. nvcc -o 01_basic 01_vector_add_basic.cu
 * 2. ncu ./01_basic (CLI)
 * 3. ncu --mode launch ./01_basic (GUI용)
 */

#include <stdio.h>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call) \
	do { \
		cudaError_t err = call; \
		if (err != cudaSuccess) { \
			fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
					__FILE__, __LINE__, cudaGetErrorString(err)); \
			exit(EXIT_FAILURE); \
		} \
	} while(0)

#define CUDA_KERNEL_CHECK() \
	do { \
		cudaError_t err = cudaGetLastError(); \
		if (err != cudaSuccess) { \
			fprintf(stderr, "Kernel Error at %s:%d - %s\n", \
					__FILE__, __LINE__, cudaGetErrorString(err)); \
			exit(EXIT_FAILURE); \
		} \
	} while(0)

// Simple vector addition kernel
__global__ void vectorAdd(float *A, float *B, float *C, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		C[i] = A[i] + B[i];
	}
}

int main(int argc, char *argv[]) {
	// Default: 64M elements (~256MB per vector)
	int n = 64 * 1024 * 1024;

	if (argc > 1) {
		n = atoi(argv[1]) * 1024 * 1024;
	}

	size_t size = n * sizeof(float);
	printf("Vector Addition: %d elements (%.2f MB per vector)\n", n, size / (1024.0 * 1024.0));

	// Host allocation
	float *h_A = (float*)malloc(size);
	float *h_B = (float*)malloc(size);
	float *h_C = (float*)malloc(size);

	// Initialize
	for (int i = 0; i < n; i++) {
		h_A[i] = 1.0f;
		h_B[i] = 2.0f;
	}

	// Device allocation
	float *d_A, *d_B, *d_C;
	CUDA_CHECK(cudaMalloc(&d_A, size));
	CUDA_CHECK(cudaMalloc(&d_B, size));
	CUDA_CHECK(cudaMalloc(&d_C, size));

	// Copy to device
	CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

	// Launch config
	int blockSize = 256;
	int gridSize = (n + blockSize - 1) / blockSize;
	printf("Grid: %d blocks, Block: %d threads\n", gridSize, blockSize);

	// Warmup
	vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
	CUDA_KERNEL_CHECK();
	CUDA_CHECK(cudaDeviceSynchronize());

	// Timed run
	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	CUDA_CHECK(cudaEventRecord(start));
	vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
	CUDA_KERNEL_CHECK();
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));

	float ms = 0;
	CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

	// Copy back
	CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

	// Verify
	bool correct = true;
	for (int i = 0; i < 100; i++) {
		if (h_C[i] != 3.0f) {
			correct = false;
			break;
		}
	}

	printf("Result: %s\n", correct ? "PASS" : "FAIL");
	printf("Kernel time: %.3f ms\n", ms);
	printf("Bandwidth: %.2f GB/s\n", (3.0 * size) / (ms * 1e6));

	// Cleanup
	CUDA_CHECK(cudaFree(d_A));
	CUDA_CHECK(cudaFree(d_B));
	CUDA_CHECK(cudaFree(d_C));
	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}

