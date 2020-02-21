// this struct is not used, it is only a sample
struct cudaDeviceInfo {
	char name[256];
	size_t sharedMemPerBlock;
	int regsPerBlock;
	int warpSize;
	size_t memPitch;
	int maxThreadsPerBlock;
	int maxThreadsDim[3];
	int maxGridSize[3];
	size_t totalConstMem;
	int major;
	int minor;
	int clockRate;
	size_t textureAlignment;
	int deviceOverlap;
	int multiProcessorCount;
	int kernelExecTimeoutEnabled;
	int integrated;
	int canMapHostMemory;
	int computeMode;
	int maxTexture1D;
	int maxTexture2D[2];
	int maxTexture3D[3];
	int maxTexture2DArray[3];
	int concurrentKernels;
};
