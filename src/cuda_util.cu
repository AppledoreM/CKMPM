#include "cuda_util.cuh"



// Single GPU only
CudaUtil::CudaUtil() : context_(CudaContext{0})
{
    int availableDeviceCount_ = 0;
    cudaCheckError(cudaGetDeviceCount(&availableDeviceCount_));

    if(availableDeviceCount_ == 0)
    {
        printf("[Cuda] No available device support cuda. Exiting\n");
        exit(0);
    }

    cudaCheckError(cudaSetDevice(0)); 
    cudaCheckError(cudaGetDeviceProperties(&prop_, 0));
    printf("[Cuda] Device Property:\n"
           "\tGPU Device: 0\n"
           "\tGlobal Memory: %llu bytes\n"
           "\tShared Memory: %llu bytes\n"
           "\tRegister Per SM: %d\n"
           "\tMulti-processor count: %d\n"
           "\tSM compute capabilities: %d.%d.\n",
           static_cast<long long unsigned int>(prop_.totalGlobalMem),
           static_cast<long long unsigned int>(prop_.sharedMemPerBlock),
           prop_.regsPerBlock, 
           prop_.multiProcessorCount,
           prop_.major,
           prop_.minor
           );

    for(int i = 0; i < static_cast<int>(StreamIndex::kTotal); ++i)
    {
        cudaCheckError(cudaStreamCreate(&stream_[i]));
    }
    printf("[Cuda] Created %u streams for device 0.\n", static_cast<uint32_t>(StreamIndex::kTotal));

    cudaCheckError(cudaDeviceSynchronize());
}
