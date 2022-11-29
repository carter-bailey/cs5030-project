#include "knn.hpp"
#include "openmp.hpp"

#include <time.h>

int main(int argc, char** argv)
{
	// make sure we have an argument passed in
	if (argc < 2)
	{
		std::cout << "Please provide the amount of centroids you would like made as an argument.\n";
		return 1;
	}

	clock_t t;
	std::cout << "Reading in the data and generating centroids\n";
	auto data = getCSV();
	auto centroids = generateCentroids(std::stoi(argv[1]), data);

	std::cout << "Running the serial K Means algorithm\n";
	t = clock();
	auto hash = serialKNN(data, centroids);
	t = clock() - t;

	double time_taken = ((double)t) / CLOCKS_PER_SEC; // in seconds
	std::cout << "Time for serial K Means is " << time_taken << "\n";

	std::cout << "Writing the results out of the serial K Means algorithm\n";
	writeToCSV(hash, centroids, "serialResults.csv");
	t = 0;

	data = getCSV();
	centroids = generateCentroids(std::stoi(argv[1]), data);
	std::cout << "Running the OpenMP K Means algorithm\n";
	t = clock();
	hash = openMPKMean(data, centroids, 4);
	t = clock() - t;

	time_taken = ((double)t) / CLOCKS_PER_SEC; // in seconds
	std::cout << "Time for OpenMP K Means is " << time_taken << "\n";
	std::cout << "Writing the results out of the serial K Means algorithm\n";
	writeToCSV(hash, centroids, "serialResults.csv");
	return 0;
}
