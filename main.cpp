#include "knn.hpp"

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
	auto centroids = generateCentroids(std::stoi(argv[1]), data);

	std::cout << "Running the serial KNN algorithm\n";
	auto hash = serialKNN(data, centroids);

	std::cout << "Writing the results out of the serial KNN algorithm\n";
	writeToCSV(hash, centroids);

	return 0;
}
