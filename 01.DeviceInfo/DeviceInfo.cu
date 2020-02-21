//	##########################################################
//	By Eugene Ch'ng | www.complexity.io
//	Email: genechng@gmail.com
//	----------------------------------------------------------
//  The ERC 'Lost Frontiers' Project
//  Development for the Parallelisation of ABM Simulation
//	----------------------------------------------------------
//	A Basic CUDA Application for ABM Development
//
//  Acquiring Device Info and Selecting a CUDA Device
//  ----------------------------------------------------------
//  How to compile:
//  nvcc <filename>.cu -o <outputfile>
//	##########################################################
#include <iostream>
#include "deviceInfo.h"
using namespace std;

// --------------------- HOST PROTOTYPES
void chooseDevice();

int main(int argc, const char * argv[])
{
	printf("------------------- SHOWING DEVICE INFO\n");
	chooseDevice();

  return 0;
}

// chooseDevice() is a host code
void chooseDevice()
{
	int count;
	cudaGetDeviceCount(&count);
	cudaDeviceProp prop;

	cout << "You have access to " << count << " NVIDIA GPU devices" << endl << endl;
	int dev;

	cudaGetDevice(&dev);

	for(int i=0; i<count; i++)
	{
		// cout << "------- Current CUDA device is [" << dev << "]" << endl;
		cudaGetDeviceProperties( &prop, i );

		cout << endl << "------------ Memory Information for Current Device [" << i << "]" << endl;
		cout << "Name: " << prop.name << endl;
		cout << "Compute Capability: " << prop.major << "." << prop.minor << endl;
		cout << "Clock rate: " << prop.clockRate << endl;

		cout << "Total global memory: " << prop.totalGlobalMem << " bytes" << endl;
		cout << "Total constant memory: " << prop.totalConstMem << " bytes" << endl;
		cout << "Multiprocessor count: " << prop.multiProcessorCount << endl;

		cout << "Shared memory per block: " << prop.sharedMemPerBlock << " bytes" << endl;

		cout << "Max threads per block: " << prop.maxThreadsPerBlock << endl;
		cout << "Max thread dimensions: " << prop.maxThreadsDim[0] << " " <<
		 prop.maxThreadsDim[1] << " " << prop.maxThreadsDim[0] << endl;
		cout << "Max grid dimensions: " << prop.maxGridSize[2] << " " <<
		 prop.maxGridSize[1] << " " << prop.maxGridSize[2] << endl;

	}

	cout << "----------------------------------------------" << endl;

  cout << "------------ Choosing a CUDA device based on highest Compute Capability" << endl;
	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 3;
	prop.minor = 0;

	cudaChooseDevice(&dev, &prop);
	cout << "------------ selected CUDA device ID: " << dev << endl << endl;
	cudaSetDevice(dev);

}
