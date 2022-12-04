// Imports for cuda and reading/writing files
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <fstream>

#include "song.hpp"

#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <vector>
#define DELIMITER ','




// #include "utils.hpp"

// This the the amount of songs int he data
#define N 64

// This is the size of the centroids so the amount of k
#define K 3

// this is the threads per block
#define TPB 32

#define centroid_count 11

#define ROUNDS 10


__device__ float distance(song s1, song s)
{

	float total = 0;
	total += (s1.danceability - s.danceability) * (s1.danceability - s.danceability);
	total += (s1.energy - s.energy) * (s1.energy - s.energy);
	total += (s1.loudness - s.loudness) * (s1.loudness - s.loudness);
	total += (s1.speechiness - s.speechiness) * (s1.speechiness - s.speechiness);
	total += (s1.acousticness - s.acousticness) * (s1.acousticness - s.acousticness);
	total += (s1.instrumental - s.instrumental) * (s1.instrumental - s.instrumental);
	total += (s1.liveness - s.liveness) * (s1.liveness - s.liveness);
	total += (s1.valence - s.valence) * (s1.valence - s.valence);
	total += (s1.tempo - s.tempo) * (s1.tempo - s.tempo);
	return sqrtf(total);
}



__global__ void findClosestAndUpdateCentroids(song *data_d, int *cluster_assignment_d, song *centroids_d)
{
    // Get id for datapoint to be updated
    const int id = blockIdx.x*blockDim.x + threadIdx.x;

    // Check the bounds
    if (id >= N) return;

    // Find the closest centroid to the datapoint 
    float min_distance = INFINITY;
    int closest_centroid= -1;

    for (int c = 0; c < K; c++)
    {
        float dist = distance(data_d[id],centroids_d[c]);

        if (dist < min_distance)
        {
            min_distance = dist;
            closest_centroid = c;
        }

    }

    // set the closest cluster id for this datapoint/threadId
    cluster_assignment_d[id] = closest_centroid;
}

__global__ void updateCentroids(song *data_d, int *cluster_assignment_d, song *centroids_d, int *cluster_sizes_d)
{
    // get the id
    const int grid_id = blockIdx.x*blockDim.x + threadIdx.x;

    // check bounds
    if (grid_id > N) return;

    const int thread_id = threadIdx.x;

    // put the datapoints and assigned clusters into shared memory
    __shared__ song data_s[TPB];
    data_s[thread_id] = data_d[grid_id];

    __shared__ int cluster_assignment_s[TPB];
    cluster_assignment_s[thread_id] = cluster_assignment_d[grid_id];

    __syncthreads();

    // have thread 0 of each block sum up the values
    if (thread_id == 0)
    {

		// ????how to make this a song?
        song data_block_sums[K] = 0.0;
        int cluster_block_sizes[K] = {0};

        for (int j=0; j < blockDim.x; ++j)
        {
            int cluster_id = cluster_assignment_s[j];
            data_block_sums[cluster_id].danceability += data_s[j].danceability;
            data_block_sums[cluster_id].energy += data_s[j].energy;
            data_block_sums[cluster_id].loudness += data_s[j].loudness;
            data_block_sums[cluster_id].speechiness += data_s[j].speechiness;
            data_block_sums[cluster_id].acousticness += data_s[j].acousticness;
            data_block_sums[cluster_id].instrumental += data_s[j].instrumental;
            data_block_sums[cluster_id].liveness += data_s[j].liveness;
            data_block_sums[cluster_id].valence += data_s[j].valence;
            data_block_sums[cluster_id].tempo += data_s[j].tempo;

        cluster_block_sizes[cluster_id] += 1;
        }

        for (int z = 0; z < K; ++z)
        {
            atomicAdd(&centroids_d[z].danceability , data_block_sums[z].danceability);
            atomicAdd(&centroids_d[z].energy , data_block_sums[z].energy);
            atomicAdd(&centroids_d[z].loudness, data_block_sums[z].loudness);
            atomicAdd(&centroids_d[z].speechiness, data_block_sums[z].speechiness);
            atomicAdd(&centroids_d[z].acousticness, data_block_sums[z].acousticness);
            atomicAdd(&centroids_d[z].instrumental , data_block_sums[z].instrumental);
            atomicAdd(&centroids_d[z].liveness , data_block_sums[z].liveness);
            atomicAdd(&centroids_d[z].valence , data_block_sums[z].valence);
            atomicAdd(&centroids_d[z].tempo , data_block_sums[z].tempo);

            atomicAdd(&cluster_sizes_d[z], cluster_block_sizes[z]);

        }

    }
    __syncthreads();

    if (thread_id < K)
    {

    // divide sums by the size
    centroids_d[grid_id].danceability /= cluster_sizes_d[grid_id];
	centroids_d[grid_id].energy /= cluster_sizes_d[grid_id];
	centroids_d[grid_id].loudness /= cluster_sizes_d[grid_id];
	centroids_d[grid_id].speechiness /= cluster_sizes_d[grid_id];
	centroids_d[grid_id].acousticness /= cluster_sizes_d[grid_id];
	centroids_d[grid_id].instrumental /= cluster_sizes_d[grid_id];
	centroids_d[grid_id].liveness /= cluster_sizes_d[grid_id];
	centroids_d[grid_id].valence /= cluster_sizes_d[grid_id];
	centroids_d[grid_id].tempo /= cluster_sizes_d[grid_id];
    }

}

int main()
// int main(song *centroids_h, song *data_h, int *cluster_sizes_h)
{
    song *data_d;
    int *cluster_assignment_d;
    song *centroids_d;
    int *cluster_sizes_d;


    cudaMalloc(&data_d, N*sizeof(song));
    cudaMalloc(&cluster_assignment_d, N*sizeof(int));
    cudaMalloc(&centroids_d, N*sizeof(song));
    cudaMalloc(&cluster_sizes_d, N*sizeof(int));

	// See if these three can be comented out using the other moethod declaration
    song *centroids_h = (song*)malloc(K*sizeof(song));
    song *data_h = (song*)malloc(N*sizeof(song));
    int *cluster_sizes_h = (int*)malloc(K*sizeof(int));

	// ???????????????????????????
		srand(time(0));

	//initialize centroids
	for(int c=0;c<K;++c)
	{
		centroids_h[c]=song((float) rand() / (double)RAND_MAX);
		printf("%f\n", centroids_h[c]);
		cluster_sizes_h[c]=0;
	}

	//initalize datapoints
	for(int d = 0; d < N; ++d)
	{
		data_h[d] = song((float) rand() / (double)RAND_MAX);
	}
	// ??????????????????????????????????

    cudaMemcpy(centroids_d, centroids_h, K*sizeof(song), cudaMemcpyHostToDevice);
    cudaMemcpy(data_d, data_h, K*sizeof(song), cudaMemcpyHostToDevice);
    cudaMemcpy(cluster_sizes_d, cluster_sizes_h, K*sizeof(int), cudaMemcpyHostToDevice);

    int current_iteration = 1;

    while (current_iteration < ROUNDS)
    {
        findClosestAndUpdateCentroids<<<(N + TPB - 1) / TPB, TPB>>>(data_d, cluster_assignment_d, centroids_d);

        cudaMemcpy(centroids_h, centroids_d, K*sizeof(song), cudaMemcpyDeviceToHost);

        for (int i = 0; i < K; ++i)
        {
            printf("Iteration %d: centroid %d: %f\n",current_iteration,i,centroids_h[i].danceability);
        }

        cudaMemset(centroids_h,0.0,K*sizeof(song));
		cudaMemset(cluster_sizes_d,0,K*sizeof(int));

        updateCentroids<<<(N+TPB-1)/TPB,TPB>>>(data_d,cluster_assignment_d, centroids_d, cluster_sizes_d);

        current_iteration+=1;

    }

    cudaFree(data_d);
	cudaFree(cluster_assignment_d);
	cudaFree(centroids_d);
	cudaFree(cluster_sizes_d);

	free(centroids_h);
	free(data_h);
	free(cluster_sizes_h);

	return 0;

}