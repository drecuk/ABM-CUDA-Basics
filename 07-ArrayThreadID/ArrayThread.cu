//	##########################################################
//	By Eugene Ch'ng | www.complexity.io
//	Email: genechng@gmail.com
//	----------------------------------------------------------
//  The ERC 'Lost Frontiers' Project
//  Development for the Parallelisation of ABM Simulation
//	----------------------------------------------------------
//	A Basic CUDA Application for ABM Development
//
//  Filling arrays with thread generated IDs
//  identify a specific thread ID and make changes for that kernel
//
//  LIMITS OF THREADS AND BLOCKS (use 01.DeviceInfo to check your GPU)
//  The particular GPU used here has 1024 threads per block
//  This presents a limit, but we can also use blocks per grid
//  Each block (for this old AlienWare GPU) has 65535 blocks per grid
//  Blocks and Threads have 3 dimensions (type dim3)
//  We will explore how to combine both blocks and threads to create
//  arbitrarily long numbers
//  ----------------------------------------------------------
//  How to compile:
//  nvcc <filename>.cu -o <outputfile>
//	##########################################################
#include <stdio.h>
#include <iostream>
using namespace std;

// as threads have a size limit of 1024, setting N > 1024 will cause an error
#define N 10

// --------------------- CUDA KERNELS
// Fill arrays with device thread IDs
__global__ void fillArray(int *dev_arr)
{
  // note that we no longer use the for loop here
  // threadIdx.x is a device variable
  // fillArray kernel is called for each thread and has its own ID
  // you can also use threadIdx.y and threadIdx.z if the kernel Chevron Syntax
  // uses dim3 as input
  int tid = threadIdx.x;

	// assign the dev_array element with threadIDx.x
	dev_arr[tid] = tid;

  // identifying a threads
  if(tid == 5)
  {
    printf("**threadIdx.x 5 is called!!\n");
    dev_arr[tid] = tid + 100;
  }
}

// the main is a host code
int main(int argc, const char * argv[])
{
	cout << "------------ initialising device and host arrays" << endl;
  int arr[N];				// host variable
	// int *arr;
  // arr = (int*)malloc(N*sizeof(int));

	int *dev_arr;  		// device variable
	for(int i=0; i<N; i++)
	{
		arr[i] = 0;
		printf("host arr[%d] = %d\n", i, arr[i]);
	}

	cout << "------------ allocate device memory dev_arr" << endl;
  // allocating a device array to copy to
	// note the N * sizeof(int)
	cudaMalloc( (void**)&dev_arr, N * sizeof(int) );

	cout << "------------ copy arr to dev_arr" << endl;
  // copying host array to device
  // note the N * sizeof(int)
	cudaMemcpy(dev_arr, arr, N * sizeof(int), cudaMemcpyHostToDevice);

  cout << "------------ calling kernel fillArray" << endl;
  // 1 block, and N threads
  fillArray<<<1,N>>>(dev_arr);

  cout << "------------ copy dev_arr to arr" << endl;
	// note the N * sizeof(int)
	cudaMemcpy(arr, dev_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

  cout << "------------ printing changed host array" << endl;
	for(int i=0; i<N; i++)
	{
		printf("** changed host arr[%d] = %d\n", i, arr[i]);
	}

  // ---- FREE ALLOCATED KERNEL MEMORY
	cudaFree( dev_arr );

  return 0;
}
