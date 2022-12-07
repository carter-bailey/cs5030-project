#include <mpi.h>
#define MCW MPI_COMM_WORLD
#include "lib/MPIandCuda.hpp"
#define centroidCount 4

int main(int argc, char** argv)
{
	int size; 
	int rank;
	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MCW, &rank);
    MPI_Comm_size(MCW, &size);
	
	std::vector<song> data;
	std::vector<song> centroids;
	if(rank == 0){
		std::cout << "Reading in the data and generating centroids\n";
		data = getCSV();
		centroids = generateCentroids(centroidCount, data);
	}
	MPI_Barrier(MCW);
	
	std::vector<song> clusteredSongsMPI[centroidCount];
	
	auto startTime = MPI_Wtime();
	if(rank == 0) std::cout << "Running the MPI KNN algorithm\n";
	MPI_KNN(data, centroids, clusteredSongsMPI, rank, size, centroidCount);
	auto endTime = MPI_Wtime();
	if(rank == 0) std::cout << "MPI KNN took " << endTime - startTime << " seconds to run.\n";

	if(rank == 0){
		std::cout << "Writing the results out of the MPI KNN algorithm\n";
		writeToCSV(clusteredSongsMPI, centroids, "MPIAndCudaResults.csv");
	}

    MPI_Finalize();
	return 0;
}
