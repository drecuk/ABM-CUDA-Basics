//	##########################################################
//	By Eugene Ch'ng | www.complexity.io
//	Email: genechng@gmail.com
//	----------------------------------------------------------
//  The ERC 'Lost Frontiers' Project
//  Development for the Parallelisation of ABM Simulation
//	----------------------------------------------------------
//	A Basic CUDA Application for ABM Development
//
//  In this application, we extend the Struct.cu with
//  an array of IDs that can represent agents from simulation code
//  that are passed in so that they can be processed using the GPU
//  The code adds position information of agents as a basis
//  for future development.
//
//  The idea is to allow each kernel to process the interaction of
//  agents based on their positional information
//
//  The sizes of structs and arrays are printed at the end
//  so that you can gauge how you should manage your memory
//  in large agent simulations
//  ----------------------------------------------------------
//  How to compile:
//  nvcc <filename>.cu -o <outputfile>
//	##########################################################

#include "book.h"
#include <iostream>
using namespace std;

// number of agents
#define N 1000

// struct to contain the position of each agent
struct POSITION {
  float x;
  float y;
  float z;
};

// struct to contain data of all agents
struct AGENTS {
	int id[N];         // array representing agent IDs
  POSITION pos[N];    // array of agent positions

  // constructor taking in an array of agent IDs
	AGENTS(int *_id)
	{
    // assign IDs to the ID array passed in
    for(int i=0; i<N; i++)
  	{
      id[i] = _id[i];
    }
	}

};

// kernel code
__global__ void changeID(AGENTS *dev_agent)
{
	// loop through all IDs in the agent struct
	for(int i=0; i<N; i++)
	{
    // change agent IDs in the kernel
    dev_agent->id[i] *= 2;

    // print agent IDs
    if(i < 10 || i  > N-10)
      printf("** device agent %d\n", dev_agent->id[i]);
  }
}

int main(void)
{

	cout<<"\n------------- assigning variables in host"<<endl;
	AGENTS *dev_agent;				// the device INFO

  cout<<"\n------------- instantiating host id"<<endl;
  int id[N];
  for(int i=0; i<N; i++)
  {
    id[i] = i;  // assign i to each ID
    // output assignment
    if(i < 10 || i  > N-10)
  	 cout<<"** host id[0]: "<<id[i]<<endl;
  }
  // send the IDs into the agent struct constructor
  AGENTS *agent = new AGENTS(id);

	cout<<"\n------------- allocate memory to device"<<endl;
	HANDLE_ERROR( cudaMalloc( (void**)&dev_agent, sizeof(AGENTS) ) );

	// copy the instantiated struct agent to device as dev_agent
	cout<<"\n------------- copy agent to dev_agent"<<endl;
	cudaMemcpy( dev_agent, agent, sizeof(AGENTS), cudaMemcpyHostToDevice);

	cout<<"\n------------- calling device kernel and change the id in the struct"<<endl;
	changeID<<<1,1>>>(dev_agent);

	// copy changed dev_agent to the struct, output the printing in the kernel
	cout<<"\n------------- copying memory from device to host and printing"<<endl;
	HANDLE_ERROR( cudaMemcpy( agent, dev_agent, sizeof(AGENTS), cudaMemcpyDeviceToHost ) );

	cout<<"\n------------- output changed results"<<endl;
  for(int i=0; i<N; i++)
  {
    if(i < 10 || i  > N-10)
      cout<<"** host agent->id[0]: "<<agent->id[i]<<endl;
  }

  cout<<"\n------------- size of struct and arrays:"<<endl;
  cout << "size of struct AGENTS: " << sizeof(AGENTS) << " bytes" << endl;
  cout << "size of struct POSITION: " << sizeof(POSITION) << " bytes" << endl;
  cout << "size of struct id[N]: " << N * sizeof(int) << " bytes" << endl;


  cout<<"\n------------- cleaning up"<<endl;
	delete agent;
	cudaFree(dev_agent);

	return 0;
}
