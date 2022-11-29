#include "kmeans.hpp"
#include "openmp.hpp"

#include <chrono>

int main(int argc, char** argv)
{
	// make sure we have an argument passed in
	if (argc < 2)
	{
		std::cout << "Please provide the amount of centroids you would like made as an argument.\n";
		return 1;
	}

	std::cout << "Reading in the data and generating centroids\n";
	auto data = getCSV();
	int centroidCount = std::stoi(argv[1]);
	auto centroids = generateCentroids(centroidCount, data);
	std::vector<song> clusteredSongs[centroidCount];

	// SERIAL
	std::cout << "Running the serial K Means algorithm\n";
	auto start = std::chrono::high_resolution_clock::now();
	serialKMeans(data, centroids, clusteredSongs);
	auto stop = std::chrono::high_resolution_clock::now();

	auto time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "Time for serial K Means is " << time_taken.count() << " ms\n";

	std::cout << "Writing the results out of the serial K Means algorithm\n";
	writeToCSV(clusteredSongs, centroids, "serialResults.csv");

	// OPENMP
	data = getCSV();
	std::vector<song> clusteredSongsMP[centroidCount];
	centroids = generateCentroids(std::stoi(argv[1]), data);
	std::cout << "Running the OpenMP K Means algorithm\n";
	auto startMP = std::chrono::high_resolution_clock::now();
	openMPKMean(data, centroids, clusteredSongsMP, 12);
	auto stopMP = std::chrono::high_resolution_clock::now();

	time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(stopMP - startMP);
	std::cout << "Time for OpenMP K Means is " << time_taken.count() << " ms\n";
	std::cout << "Writing the results out of the serial K Means algorithm\n";
	writeToCSV(clusteredSongsMP, centroids, "OpenMPResults.csv");
	return 0;
}
