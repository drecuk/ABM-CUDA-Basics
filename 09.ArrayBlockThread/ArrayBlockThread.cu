//	##########################################################
//	By Eugene Ch'ng | www.complexity.io
//	Email: genechng@gmail.com
//	----------------------------------------------------------
//  The ERC 'Lost Frontiers' Project
//  Development for the Parallelisation of ABM Simulation
//	----------------------------------------------------------
//	A Basic CUDA Application for ABM Development
//
//  Filling arrays with block generated IDs
//  This application uses both blocks and threads
//  to generate arbitrarily unsigned long numbers
//
//  Using a combination of blocks and threads for this GPU (GTX 750M)
//  we are able to generate a number with 12 zeros - about 4.3 trillion
//  This should be sufficient for a large population of agents!
//  Even larger numbers can be generated using dim3 types for blocks and threads
//  or by incrementing thread IDs after a kernel operation (discussed later)
//
//  The GPU (GTX 750M) used for this development has a limit below:
//  blockIdx = {0 ... 65535}
//  blockDim = 65535
//  threadIdx = {0 ... 1024}
//  65535 * 65535 * 1024 = 4,397,912,294,400
//
//  4 trillion thread IDs should be sufficient for ABM simulation
//  But there are also alternate ways to increase IDs for arbitrarily long arrays
//  Besides limits with blocks*threads, memory is also an issue (see below)
//
//  ----------------------------------------------------------
//  How to compile:
//  nvcc <filename>.cu -o <outputfile>
//	##########################################################
#include <stdio.h>
#include <iostream>
#include "book.h"
using namespace std;

// N = 0.22 billion (1 billion = 9 zeros)
// here we are able to reach 220,000,000+ unsigned integers with the memory
// available (GTX 750M device global memory = 2,097,086,464 = 2GB)
// unsigned long int is 8 byte, 8 x 220,000,000 = 1,760,000,000 bytes = 1.76GB
// this leaves some memory for processing the kernel code.
// putting anything larger will yield invalid argument error using cudaMemcpy
// #define N 500000000
const unsigned long int N = 220000000; //200000000

// THREADMAX is measured from using 01.DeviceInfo for GTX 750M
#define THREADMAX 1024

// --------------------- CUDA KERNELS
// Fill arrays with device thread IDs
__global__ void fillArray(unsigned long int *dev_arr)
{
  // we allow blocks and threads to cooperate in generating unsigned long numbers
  // the code below linearise the block and threads into tid used for unsigned long arrays
  // we were previously limited by the thread (1024) and blocks (65535) available
  // to the current GPU (NVIDIA GTX 750M) used for preparing this code
  // using the code below, we can generate
  // threadIdx.x and blockIdx.x is incremental
  // blockDim.x is constant calculated with (N + (THREADMAX-1)/THREADMAX) = 488282.25
  unsigned long int tid = threadIdx.x + blockIdx.x * blockDim.x;

	// assign the dev_array element with tid
  // until it reaches N
  if(tid < N)
  {
      dev_arr[tid] = tid;
  }

}

int main(int argc, const char * argv[])
{
	cout << "------------ initialising device and host arrays" << endl;
  // declaring the array on the stack would cause a segmentation fault
  // as program stack has a limit
  // otherwise, declare the array "unsigned long int arr[N]" outside main
  // to assign it as a global variable - globals are in the heap
  // unsigned long int arr[N];				// host variable

  // here we are instantiating the array on the heap for long int within main
  unsigned long int *arr;
  arr = (unsigned long int*)malloc(N*sizeof(unsigned long int));

  // cout << "-- passed malloc" << endl;

	unsigned long int *dev_arr;  		// device variable
  for(int i=0; i<N; i++)
	{
    // cout << "-- start: " << i << endl;
		arr[i] = 0;

    // commented so we don't need to print all the way up to large values
		// printf("host arr[%d] = %d\n", i, arr[i]);
	}
  size_t s = sizeof(unsigned long int);
  cout << "size of arr: " << s*N << " bytes" << endl;


  cout << "** the last second item of arr[N-1] is:" << "" << arr[N-1] << endl;

	cout << "------------ allocate device memory dev_arr" << endl;
  // allocating a device array to copy to
	// note the N * sizeof(int)
	cudaMalloc( (void**)&dev_arr, N * sizeof(unsigned long int) );

	cout << "------------ copy arr to dev_arr" << endl;
  // copying host array to device
  // note the N * sizeof(int)
  size_t size = N * sizeof(unsigned long int);
  cout << size << endl;
	HANDLE_ERROR( cudaMemcpy(dev_arr, arr, size, cudaMemcpyHostToDevice));

  cout << "------------ calling kernel fillArray" << endl;
  // What's happening here?
  // what we are doing here is to determine the number of blocks needed, in
  // combination with the thread, to generate thread IDs larger than N
  // Let's say we are using threads = 128, with N = 1000 elements
  // (N + (threads-1)/threads = 8.8
  // we will need 8.8 blocks to generate a number > N, adequate for the num ber of
  // thread IDs needed for an array of N size (8.8 * 128 = 1126.4)
  fillArray<<<(unsigned long int)(N + (THREADMAX-1))/THREADMAX,THREADMAX>>>(dev_arr);

  cout << "------------ copy dev_arr to arr" << endl;
	// note the N * sizeof(int)

	HANDLE_ERROR( cudaMemcpy(arr, dev_arr, s, cudaMemcpyDeviceToHost));

  cout << "------------ printing changed host array" << endl;
	for(unsigned long int i=0; i<N; i++)
	{
    // we want to print only 0-9 and the last 10 values of N
    if(i < 10 || i > N-10)
		  printf("** changed host arr[%ld] = %ld\n", i, arr[i]);
	}

  // ---- FREE ALLOCATED KERNEL MEMORY
	cudaFree( dev_arr );
  free(arr);

  return 0;
}
