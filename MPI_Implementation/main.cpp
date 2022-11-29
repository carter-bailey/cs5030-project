#include "knn.hpp"
#include <mpi.h>
#define MCW MPI_COMM_WORLD

int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MCW, &rank);
    MPI_Comm_size(MCW, &size);
	
	// Make sure we have an argument passed in
	if (argc < 2)
	{
		if(rank == 0) std::cout << "Please provide the amount of centroids you would like made as an argument.\n";
		return 1;
	}

	if(rank == 0){
		std::cout << "Reading in the data and generating centroids\n";
		auto data = getCSV();
		auto centroids = generateCentroids(std::stoi(argv[1]), data);
	}
	MPI_Barrier(MCW);
	// Send out all the centroids 
	MPI_Bcast();
	// Probably gonna need to create a unique mpi data type to scatter to all the processes
	MPI_Scatter()
	std::cout << "Running the serial KNN algorithm\n";
	auto hash = serialKNN(data, centroids);

	std::cout << "Writing the results out of the serial KNN algorithm\n";
	writeToCSV(hash, centroids);

    MPI_Finalize();
	return 0;
}
