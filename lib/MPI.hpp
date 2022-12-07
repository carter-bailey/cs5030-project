#pragma once
#include "kmeans.hpp"

/**
 * @brief Calculates the sum of all the attributes of the songs to be averaged later
 *
 * @param centroid The centroid to be summed
 * @param songs The songs to be summed to the centroid
 */
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


/**
 * @brief Recreate the songs from the float data
 *
 * @param songs The songs to be passed back to the main function
 * @param songNumbers The float data to be converted to songs
 * @param songCount The number of songs to be created
 */
void recreateSongs(std::vector<song> &songs, float* songNumbers, int songCount){
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

/**
 * @brief Create counts and displacements arrays for the scatterv and gatherv functions
 *
 * @param counts The counts array to be passed to the scatterv function
 * @param displacements The displacements array to be passed to the scatterv function
 * @param songCount The number of songs to be sent to each process
 * @param totalProcesses The total number of processes
 */
void createCountsAndDisplacements(std::vector<int>& counts, std::vector<int>& displacements, int songCount, int totalProcesses){
	int offset = 0;
	for(int i = 0; i < totalProcesses; i++){
		counts.push_back(songCount * song::NUM_FEATURES);
		displacements.push_back(offset);
		offset += song::NUM_FEATURES * songCount;
	}
}

/**
* @brief Create a float array of the song data to send to each of the processes
*
* @param songData The float values of the song data
* @param songs The songs to be converted to float values
*/
void createSongDataArray(std::vector<float>& songData, std::vector<song> songs){
	for(int i = 0; i < songs.size(); i++){
		float* temp = songs[i].toArray();
		for(int j = 0; j < song::NUM_FEATURES; j++){
			songData.push_back(temp[j]);
		}
	}
}

/**
 * @brief Distribute the centroids to each of the processes initially
 *
 * @param centroids The centroids to be distributed
 * @param rank The rank of the current process
 * @param totalProcesses The total number of processes
 */
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

/**
* @brief Distribute the song data to each of the processes
*
* @param songs The songs to be distributed
* @param rank The rank of the current process
* @param totalProcesses The total number of processes
*/
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


/**
 * @brief Gather the songs from each process and put them into the correct centroid
 *
 * @param songs The songs to be gathered
 * @param centroidCount The number of centroids
 * @param rank The rank of the current process
 * @param size The total number of processes
 */
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
std::vector<song> averageCentroids(std::vector<song> centroids, std::vector<int> centroidCounts){
	for(int i = 0; i < centroids.size(); i++){
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
* @brief Perform the KMeans Algorithm
*
* @param data The data to be clustered
* @param centroids The initial centroids
* @param clusteredSongs The songs that have been clustered
* @param rank The rank of the current process
* @param size The total number of processes
*/
void MPI_KNN(std::vector<song> data, std:: vector<song> centroids, std::vector<song>* clusteredSongs, int rank, int size){
	// Distribute the data equally among processes
	distributeData(data, rank, size);
	// Distribute the initial centroids to all processes
	distributeCentroids(centroids, rank, size);

	std::vector<int> centroidCounts;
	for (int i = 0; i < ROUNDS; i++)
	{
		int centroid;
		for (long unsigned int j = 0; j < data.size(); j++)
		{
			centroid = findClosestCentroid(data[j], centroids);
			clusteredSongs[centroid].push_back(data[j]);
		}
		for (long unsigned int j = 0; j < centroids.size(); j++)
		{
			calculateAttributeSums(centroids[j], clusteredSongs[j]);
			centroidCounts.push_back(clusteredSongs[j].size());
			// don't clear the vectors in the hash on the last round
			if (i < ROUNDS - 1)
			{
				clusteredSongs[j].clear();
			}
		}
		centroids = averageCentroids(centroids, centroidCounts);
		centroidCounts.clear();
	} 

	// Bring all the data back together for the 0th process to return
	gatherClusteredSongs(clusteredSongs, centroids.size(), rank, size);
}
