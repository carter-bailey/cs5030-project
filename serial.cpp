#include "lib/kmeans.hpp"

#include <chrono>
#define centroidCount 4

int main(int argc, char** argv)
{

	std::cout << "Reading in the data and generating centroids\n";
	auto data = getCSV();
	auto centroids = generateCentroids(centroidCount, data);
	std::vector<song> clusteredSongs[centroidCount];

	std::cout << "Running the serial K Means algorithm\n";
	auto start = std::chrono::high_resolution_clock::now();
	serialKMeans(data, centroids, clusteredSongs);
	auto stop = std::chrono::high_resolution_clock::now();

	auto time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "Time for serial K Means is " << time_taken.count() << " ms\n";

	std::cout << "Writing the results out of the serial K Means algorithm\n";
	writeToCSV(clusteredSongs, centroids, "serialResults.csv");

	return 0;
}
