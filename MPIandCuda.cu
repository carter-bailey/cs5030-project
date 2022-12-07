#include "lib/cudaUtils.cuh"
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
    if(kind == 0){
      cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice); 
    }else{
      cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost); 
    }
}

extern "C++" void findClosestCentroidExterior(float *data, int *cluster_assignment, float *centroids, int* numSongs, int* K, int blockDim1, int songsCount)
{
    //printf("songsCount: %d, blockDim1: %d, songsCount/blockDim1: %d\n", songsCount, blockDim1, std::ceil(songsCount/(float)blockDim1));
    findClosestCentroid<<<std::ceil(songsCount/(float)blockDim1), blockDim1>>>(data, cluster_assignment, centroids, numSongs, K);
}

extern "C++" void resetCentroidsExterior(float *centroids, int* K, int numSongs, int blockDim1)
{
    resetCentroids<<<std::ceil(numSongs/(float)blockDim1), blockDim1>>>(centroids, K);
}

extern "C++" void cudaDeviceSync()
{
    cudaDeviceSynchronize();
}

extern "C++" void sumCentroidsExterior(float *data, int *cluster_assignment, float *centroids, int *cluster_sizes, int* numSongs, int blockDim1, int songCount){
    sumCentroids<<<std::ceil(songCount/(float)blockDim1), blockDim1>>>(data, cluster_assignment, centroids, cluster_sizes, numSongs);
}
