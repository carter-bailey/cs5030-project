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


