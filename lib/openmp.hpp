#pragma once
#include "kmeans.hpp"

#include <omp.h>

/*
 * @brief An OPENMP version of the KMeans algorithm
 *
 * @param data - the songs that are being used
 * @param centroids - the centroids that we are using
 * @param clusteredSongs - An array of vectors to contain the final clustered songs
 * @param threadCount - the number of threads to use for parallelizing
 */
void openMPKMean(std::vector<song> data, std::vector<song> centroids, std::vector<song>* clusteredSongs, int threadCount)
{
	long unsigned int j;
	int centroid;
	// initialize the omp threading
#pragma omp parallel num_threads(threadCount) default(none) private(j, centroid) shared(clusteredSongs, data, centroids)

	for (int i = 0; i < ROUNDS; i++)
	{
#pragma omp for
		// for each song we have find it's closest centroid
		for (j = 0; j < data.size(); j++)
		{
			centroid = findClosestCentroid(data[j], centroids);
			// We don't want to try pushing two songs onto the same vector at once so we make this critical
#pragma omp critical
			clusteredSongs[centroid].push_back(data[j]);
		}

#pragma omp for
		// for each centroid we have update it's location based on the songs that are clustered with that centroid
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
