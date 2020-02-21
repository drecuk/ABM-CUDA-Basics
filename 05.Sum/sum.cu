//	##########################################################
//	By Eugene Ch'ng | www.complexity.io
//	Email: genechng@gmail.com
//	----------------------------------------------------------
//  The ERC 'Lost Frontiers' Project
//  Development for the Parallelisation of ABM Simulation
//	----------------------------------------------------------
//	A Basic CUDA Application for ABM Development
//
//  Adding two numbers in the device
//  ----------------------------------------------------------
//  How to compile:
//  nvcc <filename>.cu -o <outputfile>
//	##########################################################
#include <stdio.h>
#include <iostream>
using namespace std;

// --------------------- CUDA KERNELS
// kernel function sum with three parameters
__global__ void sum(int a, int b, int *c)
{
	*c = a + b;
}

// the main is a host code
int main(int argc, const char * argv[])
{
  cout << "------------ initialising host and device variables" << endl;
  int c;        // host variable
  int a = 2;
  int b = 7;
	int *dev_c;  // device variable

  cout << "------------ allocate device memory dev_c" << endl;
  // allocate dev_c memory in the device
  // we need to return the summed value to the host for printing
  // and thus the need to create a device variable.
	cudaMalloc( (void**)&dev_c, sizeof(int) );

  cout << "------------ calling kernel" << endl;
  // sum 2 + 7 on the kernel, using just 1 block and 1 thread
  sum<<<1,1>>>(a, b, dev_c);

  cout << "------------ copy dev_c to c from device to host" << endl;
  // once we have assigned the sum of 2 + 7 to dev_c, we need to
  // bring it back to the host by using cudaMemcpy(destination, source, ...)
	cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

  cout << "------------ display results" << endl;
	printf( "%d + %d = %d\n", a, b, c);

  // ---- FREE ALLOCATED KERNEL MEMORY
	cudaFree( dev_c );

  return 0;
}
