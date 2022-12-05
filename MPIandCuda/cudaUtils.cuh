#pragma once
#include "song.hpp"

__global__ void findClosestCentroid(song *data, int *cluster_assignment, song *centroids, int numSongs, int K)
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

__global__ void resetCentroids(song *centroids, int K){

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
    atomicAdd(&cluster_sizes[cluster_id], 1);
} 

__global__ void updateCentroids(song *data, int *cluster_assignment, song *centroids, int *cluster_sizes, int K)
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