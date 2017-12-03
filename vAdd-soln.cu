// vAdd.cu
//
// driver and kernel call

#include <stdio.h>

#define THREADS_PER_BLOCK 32
 
__global__ void vAdd_d (int *a_d, int *b_d, int *c_d, int n)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < n)
		c_d[x] = a_d[x] + b_d[x];
}

extern "C" void gpuAdd (int *a, int *b, int *c, int arraySize)
{
	int *a_d, *b_d, *c_d;

	cudaMalloc ((void**) &a_d, sizeof(int) * arraySize);
	cudaMalloc ((void**) &b_d, sizeof(int) * arraySize);
	cudaMalloc ((void**) &c_d, sizeof(int) * arraySize);
	cudaMemcpy (a_d, a, sizeof(int) * arraySize, cudaMemcpyHostToDevice);
	cudaMemcpy (b_d, b, sizeof(int) * arraySize, cudaMemcpyHostToDevice);

	vAdd_d <<< ceil((float) arraySize/THREADS_PER_BLOCK), THREADS_PER_BLOCK >>> (a_d, b_d, c_d, arraySize);
	
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf ("CUDA error: %s\n", cudaGetErrorString(err));
		
	cudaMemcpy (c, c_d, sizeof(int) * arraySize, cudaMemcpyDeviceToHost);
	cudaFree (a_d);
	cudaFree (b_d);
	cudaFree (c_d);
}

