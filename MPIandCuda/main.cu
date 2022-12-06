#include "cudaUtils.cuh"
#include <stdio.h>
#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }   
    return result;
}

extern "C++" void cudaMallocate(void **devPtr, size_t size)
{
    checkCuda(cudaMalloc(devPtr, size));
}

extern "C++" void cudaMemorySet(void *devPtr, int value, size_t count)
{
    checkCuda(cudaMemset(devPtr, value, count));
}

extern "C++" void cudaMemoryCopy(void *dst, const void *src, size_t count, int kind)
{
    checkCuda(cudaMemcpy(dst, src, count, kind == 0 ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost));
}

extern "C++" void findClosestCentroidExterior(float *data, int *cluster_assignment, float *centroids, int numSongs, int K, int blockDim1, int blockDim2, int blockDim3)
{
    dim3 block(std::ceil(numSongs/blockDim1),blockDim2,blockDim3);
    dim3 grid(blockDim1,blockDim2,blockDim3);
    findClosestCentroid<<<grid, block>>>(data, cluster_assignment, centroids, numSongs, K);
}

extern "C++" void resetCentroidsExterior(float *centroids, int K, int numSongs, int blockDim1, int blockDim2, int blockDim3)
{
    dim3 block(std::ceil(numSongs/blockDim1),blockDim2,blockDim3);
    dim3 grid(blockDim1,blockDim2,blockDim3);
    resetCentroids<<<grid, block>>>(centroids, K);
}

extern "C++" void cudaDeviceSync()
{
    cudaDeviceSynchronize();
}

extern "C++" void sumCentroidsExterior(float *data, int *cluster_assignment, float *centroids, int *cluster_sizes, int numSongs, int blockDim1, int blockDim2, int blockDim3){
    dim3 block(std::ceil(numSongs/blockDim1),blockDim2,blockDim3);
    dim3 grid(blockDim1,blockDim2,blockDim3);
    sumCentroids<<<block, grid>>>(data, cluster_assignment, centroids, cluster_sizes, numSongs);
}
