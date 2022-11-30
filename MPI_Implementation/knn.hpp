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

std::vector<song> averageCentroids(std::vector<song> centroids, std::vector<int> centroidCounts){
	


}

void recreateSongs(std::vector<song> &songs, float* songNumbers, int songCount, int rank, int totalProcesses){
	if(rank == 0){
		std::vector<song> temp;
		int remainder = songs.size() % totalProcesses;
		for(int i = 0; i < remainder; i++){
			temp.push_back(songs[songs.size() - i - 1]);
		}
		songs = temp;
	} 
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


float* distributeData(std::vector<song>& songs, std::vector<song>& centroids, int rank, int totalProcesses)
{
	std::vector<float> songData;
	int songCount = songs.size() / totalProcesses;
	MPI_Bcast(&songCount, 1, MPI_INT, 0, MCW);
	float* songNumbers = (float*)malloc(sizeof(float) * songCount * song::NUM_FEATURES);
	if (rank == 0)
	{
		std::vector<int> counts;
		std::vector<int> displacements;
		int offset = 0;
		// Create counts and displacements arrays
		for(int i = 0; i < totalProcesses; i++){
			counts.push_back(songCount * song::NUM_FEATURES);
			displacements.push_back(offset);
			offset += song::NUM_FEATURES * songCount;
		}
		
		// Create the song data array to send to the various processes 
		for(int i = 0; i < songs.size(); i++){
			float* temp = songs[i].toArray();
			for(int j = 0; j < song::NUM_FEATURES; j++){
				songData.push_back(temp[j]);
			}
		}

		MPI_Scatterv(songData.data(), counts.data(), displacements.data(), MPI_FLOAT, songNumbers, songCount * song::NUM_FEATURES, MPI_FLOAT, 0, MCW);
	}
	else
	{
		MPI_Scatterv(NULL, NULL, NULL, MPI_FLOAT, songNumbers, songCount * song::NUM_FEATURES, MPI_FLOAT, 0, MCW);
	}

	recreateSongs(songs, songNumbers, songCount, rank, totalProcesses);
}


std::unordered_map<int, std::vector<song>> MPI_KNN(std::vector<song> data, std:: vector<song> centroids, int rank, int size){
	// unordered map has o(n) operations instead of maps o(logn) operations
	// we don't need the map to be ordered so we'll take the speedup
	std::cout << "Distribution started" << std::endl;
	distributeData(data, centroids, rank, size);
	std::cout << "My rank is " << rank << " and I have " << data.size() << " songs" << std::endl;
	std::unordered_map<int, std::vector<song>> hash;
	// std::vector<int> centroidCounts;
	// for (int i = 0; i < ROUNDS; i++)
	// {
	// 	int centroid;
	// 	for (int j = 0; j < data.size(); j++)
	// 	{
	// 		centroid = findClosestCentroid(data[j], centroids);
	// 		hash[centroid].push_back(data[j]);
	// 	}
	// 	for (int j = 0; j < centroids.size(); j++)
	// 	{
	// 		updateCentroid(centroids[j], hash[j]);
	// 		centroidCounts.push_back(hash[j].count());
	// 		// don't clear the vectors in the hash on the last round
	// 		if (i < ROUNDS - 1)
	// 		{
	// 			hash[j].clear();
	// 		}
	// 	}
	// 	centroids = averageCentroids(centroids, centroidCounts);
	// 	centroidCounts.clear();
	// } 

	// Bring all the data back together for the 0th process to return
	// MPI_Gather();
	return hash;
}

/*
 * @brief A serial version of the KNN algorithm
 *
 * @param data - the songs that are being used
 * @param centroids - the centroids that we are using
 *
 * @return a map containing the songs that are closest to each centroid
 */
std::unordered_map<int, std::vector<song>> serialKNN(std::vector<song> data, std::vector<song> centroids)
{
	// unordered map has o(n) operations instead of maps o(logn) operations
	// we don't need the map to be ordered so we'll take the speedup
	std::unordered_map<int, std::vector<song>> hash;
	for (int i = 0; i < ROUNDS; i++)
	{
		int centroid;
		for (int j = 0; j < data.size(); j++)
		{
			centroid = findClosestCentroid(data[j], centroids);
			hash[centroid].push_back(data[j]);
		}
		for (int j = 0; j < centroids.size(); j++)
		{
			updateCentroid(centroids[j], hash[j]);
			// don't clear the vectors in the hash on the last round
			if (i < ROUNDS - 1)
			{
				hash[j].clear();
			}
		}
	}
	return hash;
}
