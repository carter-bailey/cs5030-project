__global__ void findClosestCentroid(float *data, int *cluster_assignment, float *centroids, int numSongs, int K)
{
    // Get id for datapoint to be updated
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Check the bounds
    if (id >= numSongs)
        return;

    id *= 9;
    // Find the closest centroid to the datapoint
    float minDistance = INFINITY;
    int closest_centroid = -1;

    for (int c = 0; c < K; c+=9)
    {
        float total = 0;
        total += (data[id] - centroids[c]) * (data[id] - centroids[c]);
        total += (data[id + 1] - centroids[c + 1]) * (data[id + 1] - centroids[c + 1]);
        total += (data[id + 2] - centroids[c + 2]) * (data[id + 2] - centroids[c + 2]);
        total += (data[id + 3] - centroids[c + 3]) * (data[id + 3] - centroids[c + 3]);
        total += (data[id + 4] - centroids[c + 4]) * (data[id + 4] - centroids[c + 4]);
        total += (data[id + 5] - centroids[c + 5]) * (data[id + 5] - centroids[c + 5]);
        total += (data[id + 6] - centroids[c + 6]) * (data[id + 6] - centroids[c + 6]);
        total += (data[id + 7] - centroids[c + 7]) * (data[id + 7] - centroids[c + 7]);
        total += (data[id + 8] - centroids[c + 8]) * (data[id + 8] - centroids[c + 8]);
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

__global__ void resetCentroids(float *centroids, int K){

    // get the id
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // check bounds
    if (id >= K){return;}
    id *= 9;
    // reset the clusters to 0
    centroids[id] = 0;
    centroids[id + 1] =0;
    centroids[id + 2] = 0;
    centroids[id + 3] = 0;
    centroids[id + 4] = 0;
    centroids[id + 5] = 0;
    centroids[id + 6] = 0;
    centroids[id + 7] = 0;
    centroids[id + 8] = 0;
}

__global__ void sumCentroids(float *data, int *cluster_assignment, float *centroids, int *cluster_sizes, int numSongs){

    // get the id
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // check bounds
    if (id >= numSongs){return;}
    id *= 9;
    
    int cluster_id = cluster_assignment[id];

    // Sum the centroids variables up
    atomicAdd(&centroids[cluster_id], data[id]);
    atomicAdd(&centroids[cluster_id + 1], data[id + 1]);
    atomicAdd(&centroids[cluster_id + 2], data[id + 2]);
    atomicAdd(&centroids[cluster_id + 3], data[id + 3]);
    atomicAdd(&centroids[cluster_id + 4], data[id + 4]);
    atomicAdd(&centroids[cluster_id + 5], data[id + 5]);
    atomicAdd(&centroids[cluster_id + 6], data[id + 6]);
    atomicAdd(&centroids[cluster_id + 7], data[id + 7]);
    atomicAdd(&centroids[cluster_id + 8], data[id + 8]);
    atomicAdd(&cluster_sizes[cluster_id], 1);
} 

__global__ void updateCentroids(float *data, int *cluster_assignment, float *centroids, int *cluster_sizes, int K)
{
    // get the id
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // check bounds
    if (id >= K){return;}
    id *= 9;

    // divide sums by the size
    centroids[id] /= cluster_sizes[id];
    centroids[id + 1] /= cluster_sizes[id];
    centroids[id + 2] /= cluster_sizes[id];
    centroids[id + 3] /= cluster_sizes[id];
    centroids[id + 4] /= cluster_sizes[id];
    centroids[id + 5] /= cluster_sizes[id];
    centroids[id + 6] /= cluster_sizes[id];
    centroids[id + 7] /= cluster_sizes[id];
    centroids[id + 8] /= cluster_sizes[id];
}