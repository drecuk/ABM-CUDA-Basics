//	##########################################################
//	By Eugene Ch'ng | www.complexity.io
//	Email: genechng@gmail.com
//	----------------------------------------------------------
//  The ERC 'Lost Frontiers' Project
//  Development for the Parallelisation of ABM Simulation
//	----------------------------------------------------------
//	A Basic CUDA Application for ABM Development
//
//	The program uses each threads to sum numbers in parallel
//	If you test a serial program for adding large arrays, you'll
//	you'll notice that GPU accelerated summation is significantly faster
//	although copying data to device memory takes time	
//  ----------------------------------------------------------
//  How to compile:
//  nvcc <filename>.cu -o <outputfile>
//	##########################################################
#include <stdio.h>
#include <iostream>

using namespace std;

#define N 10

__global__ void sum(int *a, int *b, int *c)
{
	int tid = threadIdx.x; 	// handle the data at this index
	if(tid < N)
		c[tid] = a[tid] + b[tid];
}

int main ( void )
{
	cout << "------------ initialising device and host arrays" << endl;
	int a[N], b[N], c[N];				// host arrays
	int *dev_a, *dev_b, *dev_c;	// device arrays

	cout << "------------ initialise arrays" << endl;
	for(int i=0; i<N; i++) {
		a[i] = i;
		b[i] = i * i;
	}

	cout << "------------ allocate device memory" << endl;
	cudaMalloc( (void**)&dev_a, N * sizeof(int) );
	cudaMalloc( (void**)&dev_b, N * sizeof(int) );
	cudaMalloc( (void**)&dev_c, N * sizeof(int) );

	cout << "------------ copy a and b to dev_a and dev_b" << endl;
	cudaMemcpy( dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy( dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

	cout << "------------ calling kernel" << endl;
	sum<<<1,N>>>(dev_a, dev_b, dev_c);

	cout << "------------ copy results back to host" << endl;
	cudaMemcpy( &c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost );

	cout << "------------ printing results" << endl;
	for(int i=0; i<N; i++)
	{
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

  // ---- FREE ALLOCATED KERNEL MEMORY
	cudaFree( dev_a );
	cudaFree( dev_b );
	cudaFree( dev_c );

	return 0;
}
