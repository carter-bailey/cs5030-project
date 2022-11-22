#include "knn.hpp"

int main(int argv, char** argc)
{
    if (argv < 2)
    {
        std::cout << "Please provide the amount of centroids you would like made as an argument.\n";
        return 1;
    }
    srand(0);
    auto data = getCSV();
    auto centroids = generateCentroids(std::stoi(argc[1]));
    for (auto c : centroids)
    {
        std::cout << c.toString();
    }
    auto hash = serialKNN(data, centroids);
    writeToCSV(hash, centroids);
    return 0;
}
