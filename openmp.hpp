#pragma once
#include "kmeans.hpp"

#include <omp.h>

// Unordered_map is parallel safe to read from but not parallel safe to write too.
// This means that I need to put a mutex when I write to the hash

/*
 * @brief An OPENMP version of the KMeans algorithm
 *
 * @param data - the songs that are being used
 * @param centroids - the centroids that we are using
 * @param clusteredSongs - An array of vectors to contain the final clustered songs
 * @param threadCount - the number of threads to use for parallelizing
 *
 *
 */
void openMPKMean(std::vector<song> data, std::vector<song> centroids, std::vector<song>* clusteredSongs, int threadCount)
{
	// unordered map has o(n) operations instead of maps o(logn) operations
	// we don't need the map to be ordered so we'll take the speedup
	long unsigned int j;
	int centroid;
#pragma omp parallel num_threads(threadCount) default(none) private(j, centroid) shared(clusteredSongs, data, centroids)

	for (int i = 0; i < ROUNDS; i++)
	{
#pragma omp for
		for (j = 0; j < data.size(); j++)
		{
			centroid = findClosestCentroid(data[j], centroids);
#pragma omp critical
			clusteredSongs[centroid].push_back(data[j]);
		}

#pragma omp for
		for (j = 0; j < centroids.size(); j++)
		{
			updateCentroid(centroids[j], clusteredSongs[j]);
			// don't clear the vectors in the hash on the last round
			if (i < ROUNDS - 1)
			{
				clusteredSongs[j].clear();
			}
		}
	}
}
