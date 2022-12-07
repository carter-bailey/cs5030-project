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


void calculateAttributeSums(song& centroid, std::vector<song> songs)
{
	centroid.reset();
	for (int i = 0; i < songs.size(); i++)
	{
		centroid.danceability += songs[i].danceability;
		centroid.energy += songs[i].energy;
		centroid.loudness += songs[i].loudness;
		centroid.speechiness += songs[i].speechiness;
		centroid.acousticness += songs[i].acousticness;
		centroid.instrumental += songs[i].instrumental;
		centroid.liveness += songs[i].liveness;
		centroid.valence += songs[i].valence;
		centroid.tempo += songs[i].tempo;
	}
}

void recreateSongs(std::vector<song> &songs, float* songNumbers, int songCount){
	// Create the songs from the song data
	for(int i = 0; i < songCount; i++){
		song s;
		s.danceability = songNumbers[i * song::NUM_FEATURES];
		s.energy = songNumbers[i * song::NUM_FEATURES + 1];
		s.loudness = songNumbers[i * song::NUM_FEATURES + 2];
		s.speechiness = songNumbers[i * song::NUM_FEATURES + 3];
		s.acousticness = songNumbers[i * song::NUM_FEATURES + 4];
		s.instrumental = songNumbers[i * song::NUM_FEATURES + 5];
		s.liveness = songNumbers[i * song::NUM_FEATURES + 6];
		s.valence = songNumbers[i * song::NUM_FEATURES + 7];
		s.tempo = songNumbers[i * song::NUM_FEATURES + 8];
		songs.push_back(s);
	}
}

// Create counts and displacements arrays
void createCountsAndDisplacements(std::vector<int>& counts, std::vector<int>& displacements, int songCount, int totalProcesses){
	int offset = 0;
	for(int i = 0; i < totalProcesses; i++){
		counts.push_back(songCount * song::NUM_FEATURES);
		displacements.push_back(offset);
		offset += song::NUM_FEATURES * songCount;
	}
}

// Create the song data array to send to the processes 
void createSongDataArray(std::vector<float>& songData, std::vector<song> songs){
	for(int i = 0; i < songs.size(); i++){
		float* temp = songs[i].toArray();
		for(int j = 0; j < song::NUM_FEATURES; j++){
			songData.push_back(temp[j]);
		}
	}
}

// Send the centroids to the processes
void distributeCentroids(std::vector<song>& centroids, int rank, int totalProcesses){
	std::vector<float> centroidData;
	int centroidCount = centroids.size();
	MPI_Bcast(&centroidCount, 1, MPI_INT, 0, MCW);
	float* centroidNumbers;
	if(rank == 0) {
		createSongDataArray(centroidData, centroids);
		centroidNumbers = centroidData.data();
	}
	else{
		centroidNumbers = (float*)malloc(sizeof(float) * centroidCount * song::NUM_FEATURES);
	}

	MPI_Bcast(centroidNumbers, centroidCount * song::NUM_FEATURES, MPI_FLOAT, 0, MCW);
	if(rank != 0) recreateSongs(centroids, centroidNumbers, centroidCount);
}

void distributeData(std::vector<song>& songs, int rank, int totalProcesses)
{
	std::vector<float> songData;
	int songCount = songs.size() / totalProcesses;
	MPI_Bcast(&songCount, 1, MPI_INT, 0, MCW);
	float* songNumbers = (float*)malloc(sizeof(float) * songCount * song::NUM_FEATURES);
	if (rank == 0)
	{
		std::vector<int> counts;
		std::vector<int> displacements;
		createCountsAndDisplacements(counts, displacements, songCount, totalProcesses);
		createSongDataArray(songData, songs);
		MPI_Scatterv(songData.data(), counts.data(), displacements.data(), MPI_FLOAT, songNumbers, songCount * song::NUM_FEATURES, MPI_FLOAT, 0, MCW);
	}
	else
	{
		MPI_Scatterv(NULL, NULL, NULL, MPI_FLOAT, songNumbers, songCount * song::NUM_FEATURES, MPI_FLOAT, 0, MCW);
	}
	
	// Account for the remaining songs that weren't sent out
	if(rank == 0){
		std::vector<song> temp;
		int remainder = songs.size() % totalProcesses;
		for(int i = 0; i < remainder; i++){
			temp.push_back(songs[songs.size() - i - 1]);
		}
		songs = temp;
	} 
	// Recreate process specific songs
	recreateSongs(songs, songNumbers, songCount);
}



void gatherClusteredSongs(std::vector<song>* songs, int centroidCount, int rank, int size){
	std::vector<int> counts;
	std::vector<int> displacements;
	for(int i = 0; i < centroidCount; i++){
		int* songCountsForThisCentroid = new int[size];
		int songsFromThisRank = songs[i].size();
		
		MPI_Gather(&songsFromThisRank, 1, MPI_INT, songCountsForThisCentroid, 1, MPI_INT, 0, MCW);
		int offset = 0;
		int totalSongsForThisCentroid = 0;
		if(rank == 0){
			for(int j = 0; j < size; j++){
				counts.push_back(songCountsForThisCentroid[j] * song::NUM_FEATURES);
				displacements.push_back(offset);
				offset += songCountsForThisCentroid[j] * song::NUM_FEATURES;
				totalSongsForThisCentroid += songCountsForThisCentroid[j];
			}
		}
		std::vector<float> songData;
		createSongDataArray(songData, songs[i]);
		float* songNumbers = new float[offset];
		MPI_Gatherv(songData.data(), songs[i].size() * song::NUM_FEATURES, MPI_FLOAT, songNumbers, counts.data(), displacements.data(), MPI_FLOAT, 0, MCW);
		counts.clear();
		displacements.clear();
		if(rank == 0){
			std::vector<song> allSongs;
			recreateSongs(allSongs, songNumbers, totalSongsForThisCentroid);
			songs[i] = allSongs;
		}
		delete songNumbers;
	}
}


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

void MPI_KNN(std::vector<song> data, std::vector<song> centroids, std::vector<song>* clusteredSongs, int rank, int size, int K){
/**
* @brief Perform the KMeans Algorithm
*
* @param data The data to be clustered
* @param centroids The initial centroids
* @param clusteredSongs The songs that have been clustered
* @param rank The rank of the current process
* @param size The total number of processes
*/
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
