#include <mpi.h>
#define MCW MPI_COMM_WORLD
#include "knn.hpp"

int main(int argc, char** argv)
{
	int size; 
	int rank;
	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MCW, &rank);
    MPI_Comm_size(MCW, &size);
	
	// Make sure we have an argument passed in
	if (argc < 2)
	{
		if(rank == 0) std::cout << "Please provide the amount of centroids you would like made as an argument.\n";
		return 1;
	}
	std::vector<song> data;
	std::vector<song> centroids;
	int centroidCount = std::stoi(argv[1]);
	if(rank == 0){
		std::cout << "Reading in the data and generating centroids\n";
		data = getCSV();
		centroids = generateCentroids(centroidCount, data);
	}
	MPI_Barrier(MCW);
	
	std::vector<song> clusteredSongsMPI[centroidCount];
	
	if(rank == 0) std::cout << "Running the MPI KNN algorithm\n";
	MPI_KNN(data, centroids, clusteredSongsMPI, rank, size);

	if(rank == 0){
		std::cout << "Writing the results out of the MPI KNN algorithm\n";
		writeToCSV(clusteredSongsMPI, centroids, "MPIResults.csv");
	}

    MPI_Finalize();
	return 0;
}
