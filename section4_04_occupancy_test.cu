/**
 * 04_occupancy_test.cu
 * 
 * Occupancy 변화 실습
 * 
 * 강의 내용:
 * - Block size 96 → 128까지 occupancy 유지
 * - Block size 224+ → occupancy 감소 시작
 * - Registers per thread 16 → 40까지 유지 가능
 * - Registers 40+ → occupancy 감소
 * 
 * GUI에서 Occupancy Calculator 그래프 확인!
 * 
 * 실습:
 * ./04_occupancy 32    → 작은 블록
 * ./04_occupancy 256   → 일반적인 블록
 * ./04_occupancy 1024  → 최대 블록
 * ./04_occupancy 256 1 → 레지스터 많이 사용
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

// Simple kernel - low register usage
__global__ void simpleKernel(float *A, float *B, float *C, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		C[i] = A[i] + B[i];
	}
}

// High register kernel - uses many local variables
__global__ void highRegisterKernel(float *A, float *B, float *C, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	// 의도적으로 많은 레지스터 사용
	float r0, r1, r2, r3, r4, r5, r6, r7;
	float r8, r9, r10, r11, r12, r13, r14, r15;
	float r16, r17, r18, r19, r20, r21, r22, r23;
	float r24, r25, r26, r27, r28, r29, r30, r31;
	float r32, r33, r34, r35, r36, r37, r38, r39;

	if (i < n) {
		r0 = A[i]; r1 = B[i];
		r2 = r0 + r1; r3 = r0 - r1; r4 = r0 * r1;
		r5 = r2 + r3; r6 = r3 + r4; r7 = r4 + r2;
		r8 = r5 * r6; r9 = r6 * r7; r10 = r7 * r5;
		r11 = r8 + r9; r12 = r9 + r10; r13 = r10 + r8;
		r14 = r11 - r12; r15 = r12 - r13; r16 = r13 - r11;
		r17 = r14 * r15; r18 = r15 * r16; r19 = r16 * r14;
		r20 = r17 + r18; r21 = r18 + r19; r22 = r19 + r17;
		r23 = r20 - r21; r24 = r21 - r22; r25 = r22 - r20;
		r26 = r23 + r24; r27 = r24 + r25; r28 = r25 + r23;
		r29 = r26 * r27; r30 = r27 * r28; r31 = r28 * r26;
		r32 = r29 + r30; r33 = r30 + r31; r34 = r31 + r29;
		r35 = r32 - r33; r36 = r33 - r34; r37 = r34 - r32;
		r38 = r35 + r36 + r37;
		r39 = r38 / (r0 + 1.0f);

		C[i] = r39;
	}
}

// Very high register kernel with __launch_bounds__
__global__ __launch_bounds__(256, 2)  // max 256 threads, min 2 blocks per SM
	void limitedBlocksKernel(float *A, float *B, float *C, int n) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < n) {
			float val = A[i] + B[i];
			// Some computation
			for (int j = 0; j < 10; j++) {
				val = val * 1.01f + 0.01f;
			}
			C[i] = val;
		}
	}

int main(int argc, char *argv[]) {
	int blockSize = 256;
	int highReg = 0;
	int limitBlocks = 0;

	if (argc > 1) blockSize = atoi(argv[1]);
	if (argc > 2) highReg = atoi(argv[2]);
	if (argc > 3) limitBlocks = atoi(argv[3]);

	// Validate block size
	if (blockSize < 32 || blockSize > 1024) {
		printf("Block size must be between 32 and 1024\n");
		return 1;
	}

	int n = 16 * 1024 * 1024;
	size_t size = n * sizeof(float);

	printf("Occupancy Test\n");
	printf("Block Size: %d\n", blockSize);
	printf("High Register: %s\n", highReg ? "YES" : "NO");
	printf("Launch Bounds: %s\n", limitBlocks ? "YES (max 2 blocks/SM)" : "NO");

	// Get device properties
	cudaDeviceProp prop;
	CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
	printf("\nGPU: %s\n", prop.name);
	printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
	printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
	printf("Registers per SM: %d\n", prop.regsPerMultiprocessor);
	printf("Shared mem per SM: %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);

	float *h_A = (float*)malloc(size);
	float *h_B = (float*)malloc(size);
	for (int i = 0; i < n; i++) {
		h_A[i] = 1.0f;
		h_B[i] = 2.0f;
	}

	float *d_A, *d_B, *d_C;
	CUDA_CHECK(cudaMalloc(&d_A, size));
	CUDA_CHECK(cudaMalloc(&d_B, size));
	CUDA_CHECK(cudaMalloc(&d_C, size));
	CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

	int gridSize = (n + blockSize - 1) / blockSize;
	printf("\nGrid: %d blocks\n", gridSize);

	// Calculate theoretical occupancy
	int maxActiveBlocksPerSM;
	if (highReg) {
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
					&maxActiveBlocksPerSM, highRegisterKernel, blockSize, 0));
	} else if (limitBlocks) {
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
					&maxActiveBlocksPerSM, limitedBlocksKernel, blockSize, 0));
	} else {
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
					&maxActiveBlocksPerSM, simpleKernel, blockSize, 0));
	}

	int maxWarpsPerSM = prop.maxThreadsPerMultiProcessor / 32;
	int activeWarps = maxActiveBlocksPerSM * (blockSize / 32);
	float theoreticalOccupancy = (float)activeWarps / maxWarpsPerSM * 100.0f;

	printf("Max active blocks per SM: %d\n", maxActiveBlocksPerSM);
	printf("Theoretical occupancy: %.1f%%\n", theoreticalOccupancy);

	// Warmup & run
	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	if (highReg) {
		highRegisterKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
		CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaEventRecord(start));
		highRegisterKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
	} else if (limitBlocks) {
		limitedBlocksKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
		CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaEventRecord(start));
		limitedBlocksKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
	} else {
		simpleKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
		CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaEventRecord(start));
		simpleKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
	}

	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));

	float ms = 0;
	CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
	printf("Kernel time: %.3f ms\n", ms);

	printf("\n=== Nsight Compute에서 확인할 것 ===\n");
	printf("- Occupancy 섹션의 Theoretical vs Achieved\n");
	printf("- Occupancy Calculator 그래프\n");
	printf("- Block Size vs Occupancy 그래프\n");
	printf("- Registers per Thread vs Occupancy 그래프\n");

	CUDA_CHECK(cudaFree(d_A));
	CUDA_CHECK(cudaFree(d_B));
	CUDA_CHECK(cudaFree(d_C));
	free(h_A);
	free(h_B);

	return 0;
}

