#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>

//#define SIZE 2048
#define TOTAL_SIZE ( 1024ULL*1024*1024 )
#define CHUNK_SIZE ( 1024ULL*1024*128 )
#define BLOCK_SIZE 1024

__global__ void vectorAdd(int *A, int *B, int *C, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	C[idx] = A[idx] + B[idx];
}

int main()
{
	int *A, *B, *C;
	int *dev_A, *dev_B, *dev_C;
	int size = CHUNK_SIZE * sizeof(int);
	int err_cnt = 0;

	A = (int *)malloc(size);
	B = (int *)malloc(size);
	C = (int *)malloc(size);

	cudaMalloc((void **)&dev_A, size);
	cudaMalloc((void **)&dev_B, size);
	cudaMalloc((void **)&dev_C, size);

	printf("iter_cnt %llu \n", TOTAL_SIZE/CHUNK_SIZE);

	for(int iter = 0; iter < TOTAL_SIZE/CHUNK_SIZE; iter++)
	{
		for(int i = 0; i < CHUNK_SIZE; i++)
		{
			A[i] = i;
			B[i] = i;
		}

		cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice);

		//cudaEvent_t start, stop;
		//cudaEventCreate(&start);
		//cudaEventCreate(&stop);
		//cudaEventRecord(start);

		//vectorAdd <<<1, 1024>>> (dev_A, dev_B, dev_C, SIZE);
		//vectorAdd <<<2, 1024>>> (dev_A, dev_B, dev_C, SIZE);

		//vectorAdd <<<CHUNK_SIZE/BLOCK_SIZE, BLOCK_SIZE>>> (dev_A, dev_B, dev_C, CHUNK_SIZE);
		vectorAdd <<<CHUNK_SIZE/BLOCK_SIZE, BLOCK_SIZE>>> (dev_A, dev_B, dev_C, CHUNK_SIZE);

		//cudaEventRecord(stop);
		cudaDeviceSynchronize();

		cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);

		for(int i = 0; i < CHUNK_SIZE; i++)
		{
			if(C[i] != i*2)
			{
				err_cnt++;
			}
		}
	}

	printf("err_cnt %d \n", err_cnt);
	//float ms = 0;
	//cudaEventElapsedTime(&ms, start, stop);
	//printf("Execution time: %f ms\n", ms);

	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);
	free(A);
	free(B);
	free(C);

	return 0;
}
