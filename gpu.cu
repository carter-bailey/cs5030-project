// Imports for cuda and reading/writing files
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include "lib/kmeans.hpp"
// This is the size of the centroids so the amount of k
// We had slightly varying results with increasing the value of K because of floating point rounding differences with cuda. 4 was the value that works consistently with the verifier
#define K 4

// The amount of times that centroids are recalculated
#define ROUNDS 20

// Finds the closest centroid to all of the data points and sets its cluster assignment
__global__ void findClosestCentroid(song *data, int *cluster_assignment, song *centroids, int numSongs)
{
    // Get id for datapoint to be updated
    const int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Check the bounds
    if (id >= numSongs)
        return;

    // Find the closest centroid to the datapoint
    float minDistance = INFINITY;
    int closest_centroid = -1;

    for (int c = 0; c < K; ++c)
    {
        float total = 0;
        total += (data[id].danceability - centroids[c].danceability) * (data[id].danceability - centroids[c].danceability);
        total += (data[id].energy - centroids[c].energy) * (data[id].energy - centroids[c].energy);
        total += (data[id].loudness - centroids[c].loudness) * (data[id].loudness - centroids[c].loudness);
        total += (data[id].speechiness - centroids[c].speechiness) * (data[id].speechiness - centroids[c].speechiness);
        total += (data[id].acousticness - centroids[c].acousticness) * (data[id].acousticness - centroids[c].acousticness);
        total += (data[id].instrumental - centroids[c].instrumental) * (data[id].instrumental - centroids[c].instrumental);
        total += (data[id].liveness - centroids[c].liveness) * (data[id].liveness - centroids[c].liveness);
        total += (data[id].valence - centroids[c].valence) * (data[id].valence - centroids[c].valence);
        total += (data[id].tempo - centroids[c].tempo) * (data[id].tempo - centroids[c].tempo);
        float dist = sqrtf(total);

        if (dist < minDistance)
        {
            minDistance = dist;
            closest_centroid = c;
        }
    }

    // set the closest cluster id for this datapoint/threadId
    cluster_assignment[id] = closest_centroid;
}

// Resets the centriods to 0 for all features
__global__ void resetCentroids(song *centroids){

    // get the id
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    // check bounds
    if (id >= K){return;}

    // reset the clusters to 0
    centroids[id].danceability = 0;
    centroids[id].energy =0;
    centroids[id].loudness = 0;
    centroids[id].speechiness = 0;
    centroids[id].acousticness = 0;
    centroids[id].instrumental = 0;
    centroids[id].liveness = 0;
    centroids[id].valence = 0;
    centroids[id].tempo = 0;
}

// Sums up the feautures for all data in a cluster
__global__ void sumCentroids(song *data, int *cluster_assignment, song *centroids, int *cluster_sizes, int numSongs){

    // get the id
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    // check bounds
    if (id >= numSongs){return;}
            
    int cluster_id = cluster_assignment[id];

    // Sum the centroids variables up
    atomicAdd(&centroids[cluster_id].danceability, data[id].danceability);
    atomicAdd(&centroids[cluster_id].energy, data[id].energy);
    atomicAdd(&centroids[cluster_id].loudness, data[id].loudness);
    atomicAdd(&centroids[cluster_id].speechiness, data[id].speechiness);
    atomicAdd(&centroids[cluster_id].acousticness, data[id].acousticness);
    atomicAdd(&centroids[cluster_id].instrumental, data[id].instrumental);
    atomicAdd(&centroids[cluster_id].liveness, data[id].liveness);
    atomicAdd(&centroids[cluster_id].valence, data[id].valence);
    atomicAdd(&centroids[cluster_id].tempo, data[id].tempo);

    // Increase the count of data for this cluster
    atomicAdd(&cluster_sizes[cluster_id], 1);
} 

// Sets the centroid to be the average of the sum of all features for the cluster
__global__ void updateCentroids(song *data, int *cluster_assignment, song *centroids, int *cluster_sizes)
{
    // get the id
    const int id = blockIdx.x * blockDim.x + threadIdx.x;

    // check bounds
    if (id >= K){return;}

    // divide sums by the size
    centroids[id].danceability /= cluster_sizes[id];
    centroids[id].energy /= cluster_sizes[id];
    centroids[id].loudness /= cluster_sizes[id];
    centroids[id].speechiness /= cluster_sizes[id];
    centroids[id].acousticness /= cluster_sizes[id];
    centroids[id].instrumental /= cluster_sizes[id];
    centroids[id].liveness /= cluster_sizes[id];
    centroids[id].valence /= cluster_sizes[id];
    centroids[id].tempo /= cluster_sizes[id];
}

// Launches all of the other device function
void launcher(song *centroids_h, song *data_h, int *cluster_assignment_h, int numSongs)
{
    // Initialize device variables
    song *data_d;
    int *cluster_assignment_d;
    song *centroids_d;
    int *cluster_sizes_h = (int *)malloc(K * sizeof(int));
    int *cluster_sizes_d;
    int *numSongs_d;

    dim3 block(std::ceil(numSongs/64),1,1);
    dim3 grid(64,1,1);

    // Allocate space for the device variables
    cudaMalloc(&data_d, numSongs * sizeof(song));
    cudaMalloc(&cluster_assignment_d, numSongs * sizeof(int));
    cudaMalloc(&centroids_d, K * sizeof(song));
    cudaMalloc(&cluster_sizes_d, K * sizeof(int));
    cudaMalloc(&numSongs_d, sizeof(int));

    // Set the cluster assignemnts and cluster sizes to 0
    cudaMemset(cluster_assignment_d, 0, numSongs * sizeof(int));
    cudaMemset(cluster_sizes_d, 0, K * sizeof(int));

    // Copy host variables to the device
    cudaMemcpy(numSongs_d, &numSongs, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(centroids_d, centroids_h, K * sizeof(song), cudaMemcpyHostToDevice);
    cudaMemcpy(data_d, data_h, numSongs * sizeof(song), cudaMemcpyHostToDevice);

    // Completes the algorithm ROUNDS times
    for(int i = 0; i < ROUNDS; i++)
    {
        // Find the closest centroids
        findClosestCentroid<<<block, grid>>>(data_d, cluster_assignment_d, centroids_d, numSongs);
        resetCentroids<<<block, grid>>>(centroids_d);

        // Synchronize device before summing and updating centroids
        cudaDeviceSynchronize();
        cudaMemset(cluster_sizes_d, 0, K * sizeof(int));

        // Get the sum of the clusters and set centroids to the average
        sumCentroids<<<block, grid>>>(data_d, cluster_assignment_d, centroids_d, cluster_sizes_d, numSongs);
        cudaDeviceSynchronize();
        updateCentroids<<<block, grid>>>(data_d, cluster_assignment_d, centroids_d, cluster_sizes_d);
        cudaDeviceSynchronize();
    }

    // copy our final results over
    cudaMemcpy(data_h, data_d, numSongs * sizeof(song), cudaMemcpyDeviceToHost);
    cudaMemcpy(cluster_assignment_h, cluster_assignment_d, numSongs * sizeof(int), cudaMemcpyDeviceToHost);

    // free all of our memory
    cudaFree(data_d);
    cudaFree(cluster_assignment_d);
    cudaFree(centroids_d);
    cudaFree(cluster_sizes_d);
    free(cluster_sizes_h);
}

// Main function that reads in data passes it to launcher and prints to file
int main()
{
    // Read in data and gnerate initial centroids
    auto data = getCSV();
    auto centroids = generateCentroids(K, data);
    int numSongs = data.size();

    int cluster_assignment_h[numSongs * sizeof(int)];

    song *data_h = &data[0];
    song *centroids_h = &centroids[0];

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float ms;
    cudaEventRecord(startEvent, 0);
    launcher(centroids_h, data_h, cluster_assignment_h, numSongs);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    printf("Time: %f\n", ms);

    // Print the results to the output file
    std::ofstream output_file("results/cudaResults.csv");
    output_file << "centroid,danceability,energy,loudness,speechiness,acousticness,instrumental,liveness,valence,tempo\n";
    for (long unsigned int i = 0; i < numSongs; ++i)
    {
        output_file << cluster_assignment_h[i] << "," << data_h[i].toString();
    }

    output_file.close();

    return 0;
}
