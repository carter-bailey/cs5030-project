#pragma once
#include "utils.hpp"

#include <functional>

std::vector<song> generateCentroids(int count)
{
    std::vector<song> centroids;
    // set up the random number generator
    std::default_random_engine eng(0);
    std::uniform_real_distribution<float> distr(0.0, 1.0);

    float values[9];
    // make random centroids
    for (int i = 0; i < count; i++)
    {

        for (int j = 0; j < 9; j++)
        {
            values[j] = distr(eng);
        }
        centroids.push_back(song(values));
    }
    return centroids;
}

void updateCentroid(song& centroid, std::vector<song> songs)
{
    centroid.reset();
    for (int i = 0; i < songs.size(); i++)
    {
        centroid.danceability += songs[i].danceability;
        centroid.energy += songs[i].energy;
        centroid.loudness += songs[i].loudness;
        centroid.speechiness += songs[i].speechiness;
        centroid.acousticness += songs[i].acousticness;
        centroid.instrumental += songs[i].instrumental;
        centroid.liveness += songs[i].liveness;
        centroid.valence += songs[i].valence;
    }
    centroid.danceability /= songs.size();
    centroid.energy /= songs.size();
    centroid.loudness /= songs.size();
    centroid.speechiness /= songs.size();
    centroid.acousticness /= songs.size();
    centroid.instrumental /= songs.size();
    centroid.liveness /= songs.size();
    centroid.valence /= songs.size();
}

int findClosestCentroid(song s, std::vector<song> centroids)
{
    float distance, minDistance = std::numeric_limits<float>::max();
    int troid;
    for (int i = 0; i < centroids.size(); i++)
    {
        distance = centroids[i].distance(s);
        if (distance < minDistance)
        {
            troid = i;
            minDistance = distance;
        }
    }
    return troid;
}

std::map<int, std::vector<song>> serialKNN(std::vector<song> data, std::vector<song> centroids)
{
    std::map<int, std::vector<song>> hash;
    for (int i = 0; i < 1; i++)
    {
        std::cout << data.size() << "\n";
        int centroid;
        for (int j = 0; j < data.size(); j++)
        {
            centroid = findClosestCentroid(data[j], centroids);
            hash[centroid].push_back(data[j]);
            if (j % 1000 == 0)
            {
                std::cout << hash[centroid].size() << "\n";
            }
        }
        for (int j = 0; j < centroids.size(); j++)
        {
            std::cout << "centroid: " << j << " " << hash[j].size() << "\n";
            updateCentroid(centroids[j], hash[j]);
            if (i < 19)
            {
                hash[j].clear();
            }
        }
    }
    return hash;
}
