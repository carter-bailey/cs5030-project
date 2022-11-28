#pragma once
#include <omp.h>   
#include "knn.hpp"

// Unordered_map is parallel safe to read from but not parallel safe to write too.
// This means that I need to put a mutex when I write to the hash

/*
 * @brief A serial version of the KNN algorithm
 *
 * @param data - the songs that are being used
 * @param centroids - the centroids that we are using
 *
 * @return a map containing the songs that are closest to each centroid
 */
std::unordered_map<int, std::vector<song>> openMPKNN(std::vector<song> data, std::vector<song> centroids, int thread_count)
{
	// unordered map has o(n) operations instead of maps o(logn) operations
	// we don't need the map to be ordered so we'll take the speedup
	std::unordered_map<int, std::vector<song>> hash;
  int j;
	int centroid;
  #pragma omp parallel num_threads(thread_count) \
    default(none) private(j, centroid) shared(i,hash, data, centroids)

	for (int i = 0; i < ROUNDS; i++)
	{
    #pragma omp for schedule(dynamic)
		for (j = 0; j < data.size(); j++)
		{
			centroid = findClosestCentroid(data[j], centroids);
      #pragma omp critical
			hash[centroid].push_back(data[j]);
		}

    #pragma omp for schedule(dynamic)
		for (j = 0; j < centroids.size(); j++)
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
