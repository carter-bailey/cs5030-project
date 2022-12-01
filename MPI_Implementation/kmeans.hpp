#pragma once
#include "utils.hpp"
#include <cmath>
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


// Gather all the weighted average centroids and compute the new centroids
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
