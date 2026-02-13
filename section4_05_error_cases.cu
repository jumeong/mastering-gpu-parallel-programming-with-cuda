/**
 * 05_error_cases.cu
 * 
 * 의도적 에러 발생 예제 - Error Checking 실습
 * 
 * 에러 체킹 강의(32번) 복습용
 * - cudaGetLastError()
 * - cudaError_t 반환값 체크
 * - 동기/비동기 에러 차이
 * 
 * 실습:
 * ./05_error 1  → 80GB 할당 시도 (메모리 부족)
 * ./05_error 2  → 잘못된 블록 사이즈
 * ./05_error 3  → Out of bounds 접근
 * ./05_error 4  → 잘못된 커널 파라미터
 * ./05_error 5  → Double free
 */

#include <stdio.h>
#include <cuda_runtime.h>

// 에러 체킹 매크로 (exit 안 함, 보여주기용)
#define CUDA_CHECK_SHOW(call) \
	do { \
		cudaError_t err = call; \
		if (err != cudaSuccess) { \
			printf("❌ CUDA Error: %s\n", cudaGetErrorString(err)); \
			printf("   Error code: %d\n", (int)err); \
			printf("   Location: %s:%d\n", __FILE__, __LINE__); \
			printf("   Call: " #call "\n"); \
		} else { \
			printf("✅ Success: " #call "\n"); \
		} \
	} while(0)

#define CUDA_KERNEL_CHECK_SHOW() \
	do { \
		cudaError_t err = cudaGetLastError(); \
		if (err != cudaSuccess) { \
			printf("❌ Kernel Launch Error: %s\n", cudaGetErrorString(err)); \
		} else { \
			printf("✅ Kernel launch OK (cudaGetLastError)\n"); \
		} \
		err = cudaDeviceSynchronize(); \
		if (err != cudaSuccess) { \
			printf("❌ Kernel Execution Error: %s\n", cudaGetErrorString(err)); \
		} else { \
			printf("✅ Kernel execution OK (cudaDeviceSynchronize)\n"); \
		} \
	} while(0)

__global__ void simpleKernel(float *data, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		data[i] = data[i] * 2.0f;
	}
}

// Bounds check 없는 위험한 커널
__global__ void unsafeKernel(float *data, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// 의도적으로 bounds check 생략!
	data[i] = data[i] * 2.0f;
}

// 잘못된 메모리 접근
__global__ void illegalAccessKernel(float *data) {
	// NULL 근처 주소 접근 시도
	float *badPtr = (float*)0x1;
	*badPtr = 1.0f;
}

void test1_allocation_failure() {
	printf("\n");
	printf("╔════════════════════════════════════════════════════════════╗\n");
	printf("║ Test 1: Memory Allocation Failure                          ║\n");
	printf("╚════════════════════════════════════════════════════════════╝\n");
	printf("Requesting 80GB (impossible on most GPUs)\n\n");

	float *d_huge;
	size_t huge_size = 80ULL * 1024 * 1024 * 1024;  // 80GB

	printf("Attempting: cudaMalloc(&d_huge, %zu bytes = 80GB)\n\n", huge_size);
	CUDA_CHECK_SHOW(cudaMalloc(&d_huge, huge_size));

	printf("\n💡 에러 체킹이 없었다면 d_huge는 쓰레기값이고,\n");
	printf("   이후 접근 시 Segmentation fault 발생!\n");
}

void test2_invalid_config() {
	printf("\n");
	printf("╔════════════════════════════════════════════════════════════╗\n");
	printf("║ Test 2: Invalid Kernel Configuration                       ║\n");
	printf("╚════════════════════════════════════════════════════════════╝\n");
	printf("Block size 2048 (max is usually 1024)\n\n");

	float *d_data;
	CUDA_CHECK_SHOW(cudaMalloc(&d_data, 1024 * sizeof(float)));

	printf("\nLaunching kernel with blockSize=2048...\n");
	simpleKernel<<<1, 2048>>>(d_data, 1024);  // 2048 > max
	CUDA_KERNEL_CHECK_SHOW();

	printf("\n💡 cudaGetLastError()로 커널 설정 에러를 즉시 감지!\n");
	printf("   (동기적 에러 - 커널 실행 전에 발생)\n");

	cudaFree(d_data);
}

void test3_out_of_bounds() {
	printf("\n");
	printf("╔════════════════════════════════════════════════════════════╗\n");
	printf("║ Test 3: Out of Bounds Access                               ║\n");
	printf("╚════════════════════════════════════════════════════════════╝\n");
	printf("Allocating 1024 elements, launching for 2048\n\n");

	float *d_data;
	int actualSize = 1024;
	CUDA_CHECK_SHOW(cudaMalloc(&d_data, actualSize * sizeof(float)));
	CUDA_CHECK_SHOW(cudaMemset(d_data, 0, actualSize * sizeof(float)));

	// 의도적으로 할당 크기 초과
	int wrongN = 2048;
	int gridSize = (wrongN + 255) / 256;

	printf("\nLaunching unsafe kernel for %d elements (allocated: %d)...\n", wrongN, actualSize);
	unsafeKernel<<<gridSize, 256>>>(d_data, wrongN);
	CUDA_KERNEL_CHECK_SHOW();

	printf("\n⚠️  주의: Out of bounds는 항상 에러로 잡히지 않음!\n");
	printf("   때로는 조용히 메모리 손상이 발생할 수 있음\n");
	printf("   compute-sanitizer 사용 권장: compute-sanitizer ./05_error 3\n");

	cudaFree(d_data);
}

void test4_invalid_pointer() {
	printf("\n");
	printf("╔════════════════════════════════════════════════════════════╗\n");
	printf("║ Test 4: Invalid Device Pointer                             ║\n");
	printf("╚════════════════════════════════════════════════════════════╝\n");
	printf("Passing NULL pointer to kernel\n\n");

	float *d_null = NULL;

	printf("Launching kernel with NULL pointer...\n");
	simpleKernel<<<1, 256>>>(d_null, 256);
	CUDA_KERNEL_CHECK_SHOW();

	printf("\n💡 cudaDeviceSynchronize()에서 비동기 에러 감지\n");
	printf("   커널 실행 중 발생한 에러는 동기화 시점에 확인\n");
}

void test5_double_free() {
	printf("\n");
	printf("╔════════════════════════════════════════════════════════════╗\n");
	printf("║ Test 5: Double Free                                        ║\n");
	printf("╚════════════════════════════════════════════════════════════╝\n\n");

	float *d_data;
	CUDA_CHECK_SHOW(cudaMalloc(&d_data, 1024 * sizeof(float)));

	printf("\nFirst free:\n");
	CUDA_CHECK_SHOW(cudaFree(d_data));

	printf("\nSecond free (same pointer):\n");
	CUDA_CHECK_SHOW(cudaFree(d_data));

	printf("\n💡 Double free는 cudaErrorInvalidDevicePointer 발생\n");
}

void test6_memcpy_direction() {
	printf("\n");
	printf("╔════════════════════════════════════════════════════════════╗\n");
	printf("║ Test 6: Wrong Memcpy Direction                             ║\n");
	printf("╚════════════════════════════════════════════════════════════╝\n\n");

	float h_data[100];
	float *d_data;
	CUDA_CHECK_SHOW(cudaMalloc(&d_data, 100 * sizeof(float)));

	printf("Correct: Host to Device\n");
	CUDA_CHECK_SHOW(cudaMemcpy(d_data, h_data, 100 * sizeof(float), cudaMemcpyHostToDevice));

	printf("\nWrong direction: Treating device as source with H2D flag\n");
	// 이건 실제로 에러가 안 날 수 있음 (UMA 때문에)
	// 하지만 논리적으로 잘못됨
	CUDA_CHECK_SHOW(cudaMemcpy(h_data, d_data, 100 * sizeof(float), cudaMemcpyHostToDevice));

	cudaFree(d_data);
}

void printGPUInfo() {
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("GPU: %s\n", prop.name);
	printf("Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0*1024.0*1024.0));
	printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
}

int main(int argc, char *argv[]) {
	printf("╔══════════════════════════════════════════════════════════════╗\n");
	printf("║          CUDA Error Cases - Error Checking Practice          ║\n");
	printf("╚══════════════════════════════════════════════════════════════╝\n\n");

	printGPUInfo();

	int test = 0;
	if (argc > 1) test = atoi(argv[1]);

	switch(test) {
		case 1: test1_allocation_failure(); break;
		case 2: test2_invalid_config(); break;
		case 3: test3_out_of_bounds(); break;
		case 4: test4_invalid_pointer(); break;
		case 5: test5_double_free(); break;
		case 6: test6_memcpy_direction(); break;
		default:
			printf("\nUsage: %s <test_number>\n", argv[0]);
			printf("  1: Memory allocation failure (80GB)\n");
			printf("  2: Invalid kernel config (block 2048)\n");
			printf("  3: Out of bounds access\n");
			printf("  4: Invalid device pointer (NULL)\n");
			printf("  5: Double free\n");
			printf("  6: Wrong memcpy direction\n");
			printf("\nRunning all tests...\n");

			test1_allocation_failure();
			test2_invalid_config();
			test3_out_of_bounds();
			test4_invalid_pointer();
			test5_double_free();
			test6_memcpy_direction();
	}

	// Reset any errors
	cudaGetLastError();

	printf("\n");
	printf("═══════════════════════════════════════════════════════════════\n");
	printf("Error Checking 핵심:\n");
	printf("1. 모든 CUDA API 호출 후 반환값 체크\n");
	printf("2. 커널 런치 후 cudaGetLastError() (동기적 에러)\n");
	printf("3. 커널 완료 후 cudaDeviceSynchronize() (비동기적 에러)\n");
	printf("4. 메모리 버그는 compute-sanitizer 사용\n");
	printf("═══════════════════════════════════════════════════════════════\n");

	return 0;
}

