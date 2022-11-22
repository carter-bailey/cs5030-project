#include "knn.hpp"

int main(int argv, char** argc)
{
    auto data = getCSV();
    auto centroids = generateCentroids(std::stoi(argc[1]));
    auto hash = serialKNN(data, centroids);
    writeToCSV(hash, centroids);
}
