//	##########################################################
//	By Eugene Ch'ng | www.complexity.io
//	Email: genechng@gmail.com
//	----------------------------------------------------------
//  The ERC 'Lost Frontiers' Project
//  Development for the Parallelisation of ABM Simulation
//	----------------------------------------------------------
//	A Basic CUDA Application for ABM Development
//
//  Passing a struct to the kernel
//  ----------------------------------------------------------
//  How to compile:
//  nvcc <filename>.cu -o <outputfile>
//	##########################################################

#include "book.h"
#include <iostream>
using namespace std;

// struct to contain data of all agents
struct MYSTRUCT {
	int id; // array of IDs

  // constructor taking in an array of agent IDs
	MYSTRUCT(int _id)
	{
    id = _id;
	}

};

// kernel code
__global__ void changeID(MYSTRUCT *_struct)
{
	_struct->id = _struct->id - 1;
}

int main(void)
{
	cout<<"\n------------- assigning variables in host"<<endl;
	MYSTRUCT *dev_struct;				// the device struct

  cout<<"\n------------- instantiating host id"<<endl;

  // send the IDs into the agent struct constructor
  MYSTRUCT *mystruct = new MYSTRUCT(100);
  cout<<"** host mystruct->id: "<<mystruct->id<<endl;

	cout<<"\n------------- allocate memory to device"<<endl;
	HANDLE_ERROR( cudaMalloc( (void**)&dev_struct, sizeof(MYSTRUCT) ) );

	cout<<"\n------------- copy mystruct to dev_struct"<<endl;
	cudaMemcpy( dev_struct, mystruct, sizeof(MYSTRUCT), cudaMemcpyHostToDevice);

	cout<<"\n------------- calling device kernel and change the id in the struct"<<endl;
	changeID<<<1,1>>>(dev_struct);

	// copy changed dev_agent to the struct, output the printing in the kernel
	cout<<"\n------------- copying memory from device to host and printing"<<endl;
	HANDLE_ERROR( cudaMemcpy( mystruct, dev_struct, sizeof(MYSTRUCT), cudaMemcpyDeviceToHost ) );

	cout<<"\n------------- output changed results"<<endl;
  cout<<"** host mystruct->id: "<<mystruct->id<<endl;


  cout<<"\n------------- cleaning up"<<endl;
	delete mystruct;
	cudaFree(dev_struct);

	return 0;
}
