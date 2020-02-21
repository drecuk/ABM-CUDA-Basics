//	##########################################################
//	By Eugene Ch'ng | www.complexity.io
//	Email: genechng@gmail.com
//	----------------------------------------------------------
//  The ERC 'Lost Frontiers' Project
//  Development for the Parallelisation of ABM Simulation
//	----------------------------------------------------------
//	A Basic CUDA Application for ABM Development
//
//  Allocating device memory
//  Copying memory from host to device and printing
//  ----------------------------------------------------------
//  How to compile:
//  nvcc <filename>.cu -o <outputfile>
//	##########################################################
#include <stdio.h>
#include <iostream>
#include "book.h"
using namespace std;

// --------------------- CUDA KERNELS
// kernel function with 1 parameters
__global__ void showVar(int *var)
{
	printf("The variable is %d\n", *var);
}

// the main is a host code
int main(int argc, const char * argv[])
{
  cout << "------------ initialising host and device variables" << endl;
  int host_int = 50;  // host variable
	int *dev_int;  // device variable


  cout << "------------ allocating memory dev_int" << endl;
    // allocate dev_int memory in the device so that we can work with it
	HANDLE_ERROR( cudaMalloc( (void**)&dev_int, sizeof(int) ) );

  cout << "------------ copy host_int to dev_int" << endl;
  // Here we copy host_int to dev_int
  // bring it back to the host by using cudaMemcpy(destination, source, ...)
	HANDLE_ERROR( cudaMemcpy(dev_int, &host_int, sizeof(int), cudaMemcpyHostToDevice));

  cout << "------------ calling kernel showVar" << endl;
  // using just 1 block and 1 thread
  showVar<<<1,1>>>(dev_int);



  // ---- FREE ALLOCATED KERNEL MEMORY
	cudaFree( dev_int );

  return 0;
}
