#pragma once
#include "kmeans.hpp"
#define BLOCKDIM 64

void resetCentroidsExterior(float *centroids, int* K, int numSongs, int blockDim1);
void findClosestCentroidExterior(float *data, int *cluster_assignment, float *centroids, int* numSongs, int* K, int blockDim1, int songsCount);
void cudaMallocate(void **devPtr, size_t size);
void cudaMemorySet(void *devPtr, int value, size_t count);
void cudaMemoryCopy(void *dst, const void *src, size_t count, int kind);
void cudaDeviceSync();
void sumCentroidsExterior(float *data, int *cluster_assignment, float *centroids, int *cluster_sizes, int* numSongs, int blockDim1, int songCount);
void updateCentroidsExterior(float *data, int *cluster_assignment, float *centroids, int *cluster_sizes, int K, int numSongs, int blockDim1, int blockDim2, int blockDim3);


/**
 * @brief Gather all the summed centroids and compute the new centroids
 *
 * @param centroids The current centroids
 * @param centroidCounts The number of songs in each centroid
 */
std::vector<song> averageCentroids(std::vector<song> centroids, int* centroidCounts, int centroidCount){
	for(int i = 0; i < centroidCount; i++){
		float* centroidAttributeSums = (float*)malloc(sizeof(float) * song::NUM_FEATURES);
		// Gather the sum of all the attributes for the centroid
		MPI_Allreduce(centroids[i].toArray(), centroidAttributeSums, song::NUM_FEATURES, MPI_FLOAT, MPI_SUM, MCW);
		int centroidSum;
		MPI_Allreduce(&centroidCounts[i], &centroidSum, 1, MPI_INT, MPI_SUM, MCW);
		// Compute the average of the attributes for the centroid
		for(int j = 0; j < song::NUM_FEATURES; j++){
			if(centroidSum != 0) centroidAttributeSums[j] /= centroidSum;
		}
		centroids[i] = song(centroidAttributeSums);
	}
	return centroids;
}

/**
* @brief Create a ragged array of songs based on the song assignments
*
* @param clusteredSongs The array of songs to be created
* @param songs The songs to be assigned
* @param songAssignments The assignments of each song
*/
void createRaggedSongArray(std::vector<song>* clusteredSongs, std::vector<song> songs, int* songAssignments){
	for(int i = 0; i < songs.size(); i++){
		clusteredSongs[songAssignments[i]].push_back(songs[i]);
	}
}

/**
* @brief Perform the KMeans Algorithm
*
* @param data The data to be clustered
* @param centroids The initial centroids
* @param clusteredSongs The songs that have been clustered
* @param rank The rank of the current process
* @param size The total number of processes
*/
void MPI_KNNWithGPU(std::vector<song> data, std::vector<song> centroids, std::vector<song>* clusteredSongs, int rank, int size, int K){
	// Distribute the data equally among processes
	distributeData(data, rank, size);
	int numSongs = data.size();
	// Distribute the initial centroids to all processes
	distributeCentroids(centroids, rank, size);
	//}

	std::vector<float> songData;
	std::vector<float> centroidData;
	createSongDataArray(songData, data);
	createSongDataArray(centroidData, centroids);

	float* data_h = songData.data(); 
	float *centroids_h = centroidData.data();
	int *cluster_sizes_h = new int[K]();

	float* data_d;
	int *cluster_assignment_d;
	int* cluster_assignment_h = new int[numSongs]();
	float *centroids_d;
	int *cluster_sizes_d;
	int *numSongs_d, *K_d;

	cudaMallocate((void**)&data_d, numSongs * sizeof(float) * song::NUM_FEATURES);
	cudaMallocate((void**)&cluster_assignment_d, numSongs * sizeof(int));
	cudaMallocate((void**)&centroids_d, K * sizeof(float) * song::NUM_FEATURES);
	cudaMallocate((void**)&cluster_sizes_d, K * sizeof(int));
	cudaMallocate((void**)&numSongs_d, sizeof(int));
	cudaMallocate((void**)&K_d, sizeof(int));

	cudaMemorySet(cluster_assignment_d, 0, numSongs * sizeof(int));
	cudaMemorySet(cluster_sizes_d, 0, K * sizeof(int));

	cudaMemoryCopy(numSongs_d, &numSongs, sizeof(int), 0);
	cudaMemoryCopy(K_d, &K, sizeof(int), 0);
	cudaMemoryCopy(data_d, data_h, numSongs * sizeof(float) * song::NUM_FEATURES, 0);


  
	std::vector<int> centroidCounts;
	std::vector<float> newCentroidData;
	std::vector<song> newCentroids;
	for(int i = 0; i < ROUNDS; i++)
	{	
		cudaMemoryCopy(centroids_d, centroids_h, K * sizeof(float) * song::NUM_FEATURES, 0);
		findClosestCentroidExterior(data_d, cluster_assignment_d, centroids_d, numSongs_d, K_d, BLOCKDIM, numSongs);
		cudaDeviceSync();
		resetCentroidsExterior(centroids_d, K_d, numSongs, BLOCKDIM);
		cudaDeviceSync();
		cudaMemorySet(cluster_sizes_d, 0, K * sizeof(int));
		sumCentroidsExterior(data_d, cluster_assignment_d, centroids_d, cluster_sizes_d, numSongs_d, BLOCKDIM, numSongs);
		cudaDeviceSync();
		cudaMemoryCopy(data_h, data_d, numSongs * sizeof(float) * song::NUM_FEATURES, 1);
		cudaMemoryCopy(centroids_h, centroids_d, K * sizeof(float) * song::NUM_FEATURES, 1);
		cudaMemoryCopy(cluster_sizes_h, cluster_sizes_d, K * sizeof(int), 1);
		newCentroids.clear();
		recreateSongs(newCentroids, centroids_h, K);
		newCentroids = averageCentroids(newCentroids, cluster_sizes_h, K);
		newCentroidData.clear();
		createSongDataArray(newCentroidData, newCentroids);

		centroids_h = newCentroidData.data();
	}

	cudaMemoryCopy(cluster_assignment_h, cluster_assignment_d, numSongs * sizeof(int), 1);
	createRaggedSongArray(clusteredSongs, data, cluster_assignment_h);
	// Bring all the data back together for the 0th process to return
	gatherClusteredSongs(clusteredSongs, centroids.size(), rank, size);
}
