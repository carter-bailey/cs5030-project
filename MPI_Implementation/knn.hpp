#pragma once
#include "utils.hpp"

#include <unordered_map>
#define ROUNDS 20

/*
 * @brief generates the centroids to be used in the knn
 *
 * @param count - the amount of centroids to create
 * @param s - the songs that the centroids will be created from
 *
 * @return - a vector of songs that will be the centroids for the knn
 */
std::vector<song> generateCentroids(int count, std::vector<song> s)
{
	std::vector<song> centroids;
	// set up the random number generator
	std::default_random_engine eng(0);
	std::uniform_int_distribution<int> distr(0, s.size() - 1);

	int point;
	std::unordered_map<int, bool> hash;

	// make random centroids
	for (int i = 0; i < count; i++)
	{
		point = distr(eng);
		if (hash.find(point) == hash.end())
		{
			centroids.push_back(s[point]);
			hash[point] = true;
		}
	}
	return centroids;
}

/*
 * @brief updates the centroid to be the center of all of it's songs
 *
 * @param centroid - the centroid to update
 * @param songs - the songs that are in the centroids domain
 *
 */
void updateCentroid(song& centroid, std::vector<song> songs)
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
	}
	centroid.danceability /= songs.size();
	centroid.energy /= songs.size();
	centroid.loudness /= songs.size();
	centroid.speechiness /= songs.size();
	centroid.acousticness /= songs.size();
	centroid.instrumental /= songs.size();
	centroid.liveness /= songs.size();
	centroid.valence /= songs.size();
}

/*
 * @brief finds the closest centroid for the song
 *
 * @param s - the song to find the closest centroid for
 * @param centroids - the centroids that we are using
 *
 * @return the index of the centroid that is closest
 */
int findClosestCentroid(song s, std::vector<song> centroids)
{
	float distance, minDistance = std::numeric_limits<float>::max();
	int troid;
	for (int i = 0; i < centroids.size(); i++)
	{
		distance = centroids[i].distance(s);
		if (distance < minDistance)
		{
			troid = i;
			minDistance = distance;
		}
	}
	return troid;
}

std::vector<song> computeAverageCentroids(std::vector<song> allCentroids, int* centroidCounts, int numCentroids){
	std::vector<song> averageCentroids;
	for(int i = 0; i < numCentroids; i++){
		song newCentroid;
		int totalSongsNearThisCentroid = 0;
		for(int j = i; j < allCentroids.size(); j += numCentroids){
			// std::cout << "Computing for " << j << std::endl;
			totalSongsNearThisCentroid += centroidCounts[j];
			newCentroid.danceability += allCentroids[j].danceability * centroidCounts[j];
			newCentroid.energy += allCentroids[j].energy * centroidCounts[j];
			newCentroid.loudness += allCentroids[j].loudness * centroidCounts[j];
			newCentroid.speechiness += allCentroids[j].speechiness * centroidCounts[j];
			newCentroid.acousticness += allCentroids[j].acousticness * centroidCounts[j];
			newCentroid.instrumental += allCentroids[j].instrumental * centroidCounts[j];
			newCentroid.liveness += allCentroids[j].liveness * centroidCounts[j];
			newCentroid.valence += allCentroids[j].valence * centroidCounts[j];
			newCentroid.tempo += allCentroids[j].tempo * centroidCounts[j];
		}
		std::cout << "Total danceablity: " << newCentroid.danceability << " divided by "<< totalSongsNearThisCentroid <<std::endl;
		newCentroid.danceability /= totalSongsNearThisCentroid;
		newCentroid.energy /= totalSongsNearThisCentroid;
		newCentroid.loudness /= totalSongsNearThisCentroid;
		newCentroid.speechiness /= totalSongsNearThisCentroid;
		newCentroid.acousticness /= totalSongsNearThisCentroid;
		newCentroid.instrumental /= totalSongsNearThisCentroid;
		newCentroid.liveness /= totalSongsNearThisCentroid;
		newCentroid.valence /= totalSongsNearThisCentroid;
		newCentroid.tempo /= totalSongsNearThisCentroid;
		averageCentroids.push_back(newCentroid);
	}
	return averageCentroids;
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
	// std::cout << "Process " << rank << " has " << centroids.size() << " centroids" << std::endl;
}

float* distributeData(std::vector<song>& songs, int rank, int totalProcesses)
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


// Gather all the weighted average centroids and compute the new centroids
std::vector<song> averageCentroids(std::vector<song> centroids, std::vector<int> centroidCounts, int numCentroids, int rank, int size){
	std::vector<int> counts;
	std::vector<int> displacements;
	float* centroidData = new float[numCentroids * song::NUM_FEATURES * size];
	int* centroidCountsData = new int[numCentroids * size];
	createCountsAndDisplacements(counts, displacements, numCentroids, size);
	MPI_Gatherv(centroids.data(), numCentroids * song::NUM_FEATURES, MPI_FLOAT, centroidData, counts.data(), displacements.data(), MPI_FLOAT, 0, MCW);
	MPI_Gather(centroidCounts.data(), numCentroids, MPI_INT, centroidCountsData, numCentroids, MPI_INT, 0, MCW);
	std::vector<song> averagedCentroids;
	if(rank == 0){
		std::vector<song> allCentroids;
		recreateSongs(allCentroids, centroidData, size * numCentroids);
		averagedCentroids = computeAverageCentroids(allCentroids, centroidCountsData, numCentroids);
	}
	distributeCentroids(averagedCentroids, rank, size);
	return averagedCentroids;
}

void gatherClusteredSongs(std::vector<song>* songs, int centroidCount, int totalSongs, int rank, int size){
	std::vector<int> counts;
	std::vector<int> displacements;
	// for(int i = 0; i < size; i++){
	// 		std::cout << "Rank " << rank << " has " << songs[i].size() << " songs for centroid " << i << std::endl;
	// }
	for(int i = 0; i < centroidCount; i++){
		int* songCountsForThisCentroid = new int[size];
		int songsFromThisRank = songs[i].size();
		
		MPI_Gather(&songsFromThisRank, 1, MPI_INT, songCountsForThisCentroid, 1, MPI_INT, 0, MCW);
		int offset = 0;
		int totalSongsForThisCentroid = 0;
		if(rank == 0){
			for(int j = 0; j < size; j++){
				// std::cout << "Process " << j << " has " << songCountsForThisCentroid[j] << " songs for centroid " << i << std::endl;
				counts.push_back(songCountsForThisCentroid[j] * song::NUM_FEATURES);
				displacements.push_back(offset);
				offset += songCountsForThisCentroid[j] * song::NUM_FEATURES;
				totalSongsForThisCentroid += songCountsForThisCentroid[j];
			}
			// std::cout << "Total songs for centroid " << i << " is " << totalSongsForThisCentroid << " with offset " << offset << std::endl;
		}
		std::vector<float> songData;
		createSongDataArray(songData, songs[i]);
		float* songNumbers = new float[offset];
		// std::cout << "Rank " << rank << " has " << songs[i].size() << " songs for centroid " << i << std::endl;
		MPI_Gatherv(songData.data(), songs[i].size() * song::NUM_FEATURES, MPI_FLOAT, songNumbers, counts.data(), displacements.data(), MPI_FLOAT, 0, MCW);
		counts.clear();
		displacements.clear();
		if(rank == 0){
			std::vector<song> allSongs;
			recreateSongs(allSongs, songNumbers, totalSongsForThisCentroid);
			songs[i] = allSongs;
			std::cout << "\t\tTotal songs for centroid " << i << " is " << songs[i].size() << std::endl;
		}
		delete songNumbers;
	}
}


void MPI_KNN(std::vector<song> data, std:: vector<song> centroids, std::vector<song>* clusteredSongs, int rank, int size){
	// unordered map has o(n) operations instead of maps o(logn) operations
	// we don't need the map to be ordered so we'll take the speedup
	int totalSongs = 0;
	if(rank == 0) totalSongs = data.size();
	distributeData(data, rank, size);
	distributeCentroids(centroids, rank, size);

	if(rank == 0){
		for(int i = 0; i < centroids.size(); i++){
			std::cout << "Centroid " << i << " is " << centroids[i].toString() << std::endl;
		}
	}

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
			std::cout << "Centrid count for " << j << " is " << clusteredSongs[j].size() << std::endl;
			updateCentroid(centroids[j], clusteredSongs[j]);
			centroidCounts.push_back(clusteredSongs[j].size());
			// don't clear the vectors in the hash on the last round
			if (i < ROUNDS - 1)
			{
				clusteredSongs[j].clear();
			}
		}
		centroids = averageCentroids(centroids, centroidCounts, centroids.size(), rank, size);
		if(rank == 0) {
			for(int i = 0; i < centroids.size(); i++){
				std::cout << "Centroid " << i << " is " << centroids[i].toString() << std::endl;
			}
		}
		centroidCounts.clear();
	} 

	// Bring all the data back together for the 0th process to return
	gatherClusteredSongs(clusteredSongs, centroids.size(), totalSongs, rank, size);
}

/*
 * @brief A serial version of the KNN algorithm
 *
 * @param data - the songs that are being used
 * @param centroids - the centroids that we are using
 *
 * @return a map containing the songs that are closest to each centroid
 */
void serialKMeans(std::vector<song> data, std::vector<song> centroids, std::vector<song>* clusteredSongs)
{
	// unordered map has o(n) operations instead of maps o(logn) operations
	// we don't need the map to be ordered so we'll take the speedup

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
			updateCentroid(centroids[j], clusteredSongs[j]);
			// don't clear the vectors in the hash on the last round
			if (i < ROUNDS - 1)
			{
				clusteredSongs[j].clear();
			}
			for(int i = 0; i < centroids.size(); i++){
				std::cout << "Centroid " << i << " is " << centroids[i].toString() << std::endl;
			}
		}
	}
}
