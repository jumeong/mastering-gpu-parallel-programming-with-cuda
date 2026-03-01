#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define VER 3

__global__ void vector_reduction_v0(float* input, int n)
{
	int tid = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	for(int stride = 1; stride < blockDim.x; stride *= 2)
	{
		if(tid % (stride<<1) == 0 && idx + stride < n)
		{
			input[idx] += input[idx + stride];
		}

		__syncthreads();
	}

	if(tid == 0)
	{
		input[blockIdx.x] = input[idx];
	}
}

__global__ void vector_reduction_v1(float* input, int n)
{
	int tid = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//for(int stride = 1; stride < blockDim.x; stride *= 2)
	for(int stride = blockDim.x/2; stride > 0; stride /= 2)
	{
		if(idx + stride < n)
		{
			input[idx] += input[idx + stride];
		}

		__syncthreads();
	}

	if(tid == 0)
	{
		input[blockIdx.x] = input[idx];
	}
}

__global__ void vector_reduction_v2(float* input, int n)
{
	int tid = threadIdx.x;
	int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

	if(idx + blockDim.x < n)
	{
		input[idx] += input[idx + blockDim.x];
	}

	for(int stride = 1; stride < blockDim.x; stride *= 2)
	{
		if(idx + stride < n)
		{
			input[idx] += input[idx + stride];
		}

		__syncthreads();
	}

	if(tid == 0)
	{
		input[blockIdx.x] = input[idx];
	}
}

__global__ void vector_reduction_v3(float *input, int n)
{
    __shared__ float psum[8];

	int tid = threadIdx.x;
	int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float sum = 0.0f;
    if(idx < n)
        sum += input[idx];
    if(idx + blockDim.x < n)
        sum += input[idx + blockDim.x];

    for(int offset = warpSize>>1; offset > 0; offset /= 2)
    {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if(tid % warpSize == 0)
        psum[tid/warpSize] = sum;

    __syncthreads();

    if(tid < warpSize)
    {
        sum = tid < blockDim.x / warpSize ? psum[tid] : 0.0f;
        for(int offset = warpSize>>1; offset > 0; offset /= 2)
        {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if(tid == 0)
            input[blockIdx.x] = sum;
    }
}

int main()
{
	int n = 1024 * 1024;
	size_t size = n * sizeof(float);

	float *d_input;
	float *h_input = new float[n];
	cudaMalloc(&d_input, size);

	float cpu_sum = 0;
	for(int i = 0; i < n; i++)
	{
		h_input[i] = static_cast<float>(i);
		cpu_sum += h_input[i];
	}

	cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

	int block_size = 256;
    #if (VER < 2)
	int grid_size = (n + block_size - 1) / block_size;
    #else
	int grid_size = (n + block_size*2 - 1) / (block_size*2);
    #endif

	while (n > 1)
	{
        #if (VER == 0)
		vector_reduction_v0<<<grid_size, block_size>>>(d_input, n);
        #elif (VER == 1)
		vector_reduction_v1<<<grid_size, block_size>>>(d_input, n);
        #elif (VER == 2)
		vector_reduction_v2<<<grid_size, block_size>>>(d_input, n);
        #else
		vector_reduction_v3<<<grid_size, block_size>>>(d_input, n);
        #endif
		cudaDeviceSynchronize();

		n = grid_size;
        #if (VER < 2)
		grid_size = (n + block_size - 1) / block_size;
        #else
		grid_size = (n + block_size*2 - 1) / (block_size*2);
        #endif
	}

	float gpu_sum;
	cudaMemcpy(&gpu_sum, d_input, sizeof(float), cudaMemcpyDeviceToHost);
	printf("cpu_sum: %f\n gpu_sum: %f\n", cpu_sum, gpu_sum);

	cudaFree(d_input);
	delete[] h_input;

	return 0;
}
