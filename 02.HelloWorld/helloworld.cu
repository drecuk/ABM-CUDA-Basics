//	##########################################################
//	By Eugene Ch'ng | www.complexity.io
//	Email: genechng@gmail.com
//	----------------------------------------------------------
//  The ERC 'Lost Frontiers' Project
//  Development for the Parallelisation of ABM Simulation
//	----------------------------------------------------------
//	A Basic CUDA Application for ABM Development
//
//  Calling Hello World from the CUDA Device
//  ----------------------------------------------------------
//  How to compile:
//  nvcc <filename>.cu -o <outputfile>
//	##########################################################
#include <stdio.h>

// --------------------- CUDA KERNELS
// The declaration __global__ alerts the compiler that this should be compiled
// on the device (gpu) rather than the host machine
__global__ void kernel( void ) {
	// an empty kenel
}

// the main is a host code
int main(int argc, const char * argv[])
{
  // invoking the kernel with the Chevron Syntax <<blockPerGrid,threadPerBlock>>
  // in the case below, <<1 block, 1 thread in the block>>
	// the Chevron Syntax can take in dim3 type (int x,int y, int z)
	kernel<<<1,1>>>();

  // printing hello world on the host
  printf("Hello, World!\n");

  return 0;
}
