// Imports for cuda and reading/writing files
#include <iostream>
#include <cuda.h>
#include "utils.hpp"
#include <cuda_runtime.h>
#include <string>
#include <fstream>
#include <unordered_map>
#include "kmeans.hpp"

#include "song.hpp"

#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <vector>
#include <cmath>
#define DELIMITER ','

// #include "utils.hpp"

// This the the amount of songs int he data
#define N 42305

// This is the size of the centroids so the amount of k
#define K 12


#define ROUNDS 20

__global__ void findClosestAndUpdateCentroids(song *data_d, int *cluster_assignment_d, song *centroids_d)
{
    // Get id for datapoint to be updated
    const int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Check the bounds
    if (id >= N)
        return;

    // Find the closest centroid to the datapoint
    float min_distance = INFINITY;
    int closest_centroid = -1;

    for (int c = 0; c < K; ++c)
    {

        float total = 0;
        total += (data_d[id].danceability - centroids_d[c].danceability) * (data_d[id].danceability - centroids_d[c].danceability);
        total += (data_d[id].energy - centroids_d[c].energy) * (data_d[id].energy - centroids_d[c].energy);
        total += (data_d[id].loudness - centroids_d[c].loudness) * (data_d[id].loudness - centroids_d[c].loudness);
        total += (data_d[id].speechiness - centroids_d[c].speechiness) * (data_d[id].speechiness - centroids_d[c].speechiness);
        total += (data_d[id].acousticness - centroids_d[c].acousticness) * (data_d[id].acousticness - centroids_d[c].acousticness);
        total += (data_d[id].instrumental - centroids_d[c].instrumental) * (data_d[id].instrumental - centroids_d[c].instrumental);
        total += (data_d[id].liveness - centroids_d[c].liveness) * (data_d[id].liveness - centroids_d[c].liveness);
        total += (data_d[id].valence - centroids_d[c].valence) * (data_d[id].valence - centroids_d[c].valence);
        total += (data_d[id].tempo - centroids_d[c].tempo) * (data_d[id].tempo - centroids_d[c].tempo);
        float dist = sqrtf(total);

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
    const int id = blockIdx.x * blockDim.x + threadIdx.x;

    // check bounds
    if (id >= N){return;}
            
            int cluster_id = cluster_assignment_d[id];

            centroids_d[cluster_id].danceability = 0;
            centroids_d[cluster_id].energy =0;
            centroids_d[cluster_id].loudness = 0;
            centroids_d[cluster_id].speechiness = 0;
            centroids_d[cluster_id].acousticness = 0;
            centroids_d[cluster_id].instrumental = 0;
            centroids_d[cluster_id].liveness = 0;
            centroids_d[cluster_id].valence = 0;
            centroids_d[cluster_id].tempo = 0;
            cluster_sizes_d[cluster_id] = 0;

            __syncthreads();

            atomicAdd(&centroids_d[cluster_id].danceability, data_d[id].danceability);
            atomicAdd(&centroids_d[cluster_id].energy, data_d[id].energy);
            atomicAdd(&centroids_d[cluster_id].loudness, data_d[id].loudness);
            atomicAdd(&centroids_d[cluster_id].speechiness, data_d[id].speechiness);
            atomicAdd(&centroids_d[cluster_id].acousticness, data_d[id].acousticness);
            atomicAdd(&centroids_d[cluster_id].instrumental, data_d[id].instrumental);
            atomicAdd(&centroids_d[cluster_id].liveness, data_d[id].liveness);
            atomicAdd(&centroids_d[cluster_id].valence, data_d[id].valence);
            atomicAdd(&centroids_d[cluster_id].tempo, data_d[id].tempo);
            atomicAdd(&cluster_sizes_d[cluster_id], 1);

    __syncthreads();

    if (id == 0)
    {
        for (int f = 0; f < K; ++f)
        {
            if (cluster_sizes_d[f] == 0)
            {
                printf("\nmade it here %d \n", f);
            }else
            {
                   // divide sums by the size
        centroids_d[f].danceability /= cluster_sizes_d[f];
        centroids_d[f].energy /= cluster_sizes_d[f];
        centroids_d[f].loudness /= cluster_sizes_d[f];
        centroids_d[f].speechiness /= cluster_sizes_d[f];
        centroids_d[f].acousticness /= cluster_sizes_d[f];
        centroids_d[f].instrumental /= cluster_sizes_d[f];
        centroids_d[f].liveness /= cluster_sizes_d[f];
        centroids_d[f].valence /= cluster_sizes_d[f];
        centroids_d[f].tempo /= cluster_sizes_d[f];

            }



         

        }

        
    }
}

// int main()
void launcher(song *centroids_h, song *data_h, int *cluster_assignment_h)
{
    song *data_d;
    int *cluster_assignment_d;
    song *centroids_d;
    int *cluster_sizes_d;

    cudaMalloc(&data_d, N * sizeof(song));
    cudaMalloc(&cluster_assignment_d, N * sizeof(int));
    cudaMalloc(&centroids_d, N * sizeof(song));
    cudaMalloc(&cluster_sizes_d, N * sizeof(int));

    int *cluster_sizes_h = (int *)malloc(K * sizeof(int));

    cudaMemcpy(centroids_d, centroids_h, K * sizeof(song), cudaMemcpyHostToDevice);
    cudaMemcpy(data_d, data_h, N * sizeof(song), cudaMemcpyHostToDevice);
    cudaMemcpy(cluster_sizes_d, cluster_sizes_h, K * sizeof(int), cudaMemcpyHostToDevice);

    int current_iteration = 1;

    while (current_iteration < ROUNDS)
    {
        findClosestAndUpdateCentroids<<<std::ceil(N/256.0), 256>>>(data_d, cluster_assignment_d, centroids_d);

        cudaMemcpy(centroids_h, centroids_d, K * sizeof(song), cudaMemcpyDeviceToHost);

        for (int i = 0; i < K; ++i)
        {
            printf("Iteration %d: centroid %d: %f\n",current_iteration,i,centroids_h[i].danceability);
        }

        // cudaMemset(centroids_d, 0.0, K * sizeof(song));
        // cudaMemset(cluster_sizes_d, 0, K * sizeof(int));

        updateCentroids<<<std::ceil(N/256.0), 256>>>(data_d, cluster_assignment_d, centroids_d, cluster_sizes_d);


        cudaMemcpy(data_h, data_d, N * sizeof(song), cudaMemcpyDeviceToHost);
        cudaMemcpy(cluster_assignment_h, cluster_assignment_d, N * sizeof(int), cudaMemcpyDeviceToHost);

        current_iteration += 1;
    }

    cudaFree(data_d);
    cudaFree(cluster_assignment_d);
    cudaFree(centroids_d);
    cudaFree(cluster_sizes_d);

    // free(centroids_h);
    // free(data_h);
    free(cluster_sizes_h);
}

int main()
{
    int cluster_assignment_h[N * sizeof(int)];

    // malloc(&cluster_assignment_h, N*sizeof(int));

    auto data = getCSV();
    auto centroids = generateCentroids(K, data);

    song *data_h = &data[0];
    song *centroids_h = &centroids[0];

    launcher(centroids_h, data_h, cluster_assignment_h);

    // printf("%f\n",centroids_h[0].danceability);

    std::ofstream output_file("cudaResults.csv");
    output_file << "centroid,danceability,energy,loudness,speechiness,\
    acousticness,instrumental,liveness,valence,tempo\n";
    for (long unsigned int i = 0; i < N; ++i)
    {
        // for (auto s : centroids_h[0])
        // {
        output_file << cluster_assignment_h[i] << "," << data_h[i].toString();
        // output_file <<centroids_h[i].toString() << ",";
    }

    output_file.close();

    return 0;
}