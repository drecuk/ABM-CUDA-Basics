//	##########################################################
//	By Eugene Ch'ng | www.complexity.io
//	Email: genechng@gmail.com
//	----------------------------------------------------------
//  The ERC 'Lost Frontiers' Project
//  Development for the Parallelisation of ABM Simulation
//	----------------------------------------------------------
//	A Basic CUDA Application for ABM Development
//
//  Allocating and copying from device memory
//  Change value in device and copy back to host
//  ----------------------------------------------------------
//  How to compile:
//  nvcc <filename>.cu -o <outputfile>
//	##########################################################
#include <stdio.h>
#include <iostream>
using namespace std;

// --------------------- CUDA KERNELS
// kernel function with 1 parameters
__global__ void changeVar(int *var)
{
	printf("Variable before change: %d\n", *var);
	*var = 100;

}

// the main is a host code
int main(int argc, const char * argv[])
{
  cout << "------------ initialising host and device variables" << endl;
  int host_int = 50;  // host variable
	int *dev_int;  // device variable


  cout << "------------ allocating memory dev_int" << endl;
  // allocate dev_int memory in the device so that we can work with it
	cudaMalloc( (void**)&dev_int, sizeof(int) );

  cout << "------------ copy host_int to dev_int" << endl;
  // Here we copy host_int to dev_int
  // bring it back to the host by using cudaMemcpy(destination, source, ...)
	cudaMemcpy(dev_int, &host_int, sizeof(int), cudaMemcpyHostToDevice);

  cout << "------------ calling kernel changeVar" << endl;
  // using just 1 block and 1 thread
  changeVar<<<1,1>>>(dev_int);

  cout << "------------ copy host_int to dev_int" << endl;
  // once we have assigned the sum of 2 + 7 to dev_c, we need to
  // bring it back to the host by using cudaMemcpy(destination, source, ...)
	cudaMemcpy(&host_int, dev_int, sizeof(int), cudaMemcpyDeviceToHost);

  // ---- DISPLAY RESULTS
	printf("Variable after change: %d\n", host_int);

  // ---- FREE ALLOCATED KERNEL MEMORY
	cudaFree( dev_int );

  return 0;
}
