#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void test01()
{
	printf("BlockId %d ThreadId %d WarpId %d\n", blockIdx.x, threadIdx.x, threadIdx.x/32);
}

int main()
{
	test01 <<<1, 1024>>> ();
	cudaDeviceSynchronize();
};
