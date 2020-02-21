//	##########################################################
//	By Eugene Ch'ng | www.complexity.io
//	Email: genechng@gmail.com
//	----------------------------------------------------------
//  The ERC 'Lost Frontiers' Project
//  Development for the Parallelisation of ABM Simulation
//	----------------------------------------------------------
//	A Basic CUDA Application for ABM Development
//
//  Managing Arrays between Host and Device
//  ----------------------------------------------------------
//  How to compile:
//  nvcc <filename>.cu -o <outputfile>
//	##########################################################
#include <stdio.h>
#include <iostream>
using namespace std;

#define N 10

// --------------------- CUDA KERNELS
// kernel function for changing host copied arrays
__global__ void showArray(int *dev_arr)
{
	// print the variables copied to the kernel
	for(int i=0; i<N; i++)
	{
		// change the array elements in the device
		dev_arr[i] = i;
		// printf("** from device _arr[%d] = %d\n", i, _arr[i]);
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

  cout << "------------ calling kernel showArray" << endl;
  showArray<<<1,1>>>(dev_arr);

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
