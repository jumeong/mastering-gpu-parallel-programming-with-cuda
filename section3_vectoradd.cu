#include <stdio.h>
#include <cuda_runtime.h>

#define SIZE 1024

__global__ void vectorAdd(int *A, int *B, int *C, int n)
{
	int i = threadIdx.x;
	C[i] = A[i] + B[i];
}

int main()
{
	int *A, *B, *C;
	int *dev_A, *dev_B, *dev_C;
	int size = SIZE * sizeof(int);

	A = (int *)malloc(size);
	B = (int *)malloc(size);
	C = (int *)malloc(size);

	cudaMalloc((void **)&dev_A, size);
	cudaMalloc((void **)&dev_B, size);
	cudaMalloc((void **)&dev_C, size);

	for(int i = 0; i < SIZE; i++)
	{
		A[i] = i;
		B[i] = i;
	}

	cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice);

	vectorAdd <<<1, SIZE>>> (dev_A, dev_B, dev_C, SIZE);
	cudaDeviceSynchronize();

	cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);

	int err_cnt = 0;
	for(int i = 0; i < SIZE; i++)
	{
		if(C[i] != i*2)
		{
			err_cnt++;
		}
	}

	printf("err_cnt %d \n", err_cnt);

	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);
	free(A);
	free(B);
	free(C);

	return 0;
}
