#pragma once
#include "utils.hpp"

std::vector<song> generateCentroids(int count)
{
    std::vector<song> centroids;
    // set up the random number generator
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<float> distr(0, 1);

    // make random centroids
    centroids.push_back(song(distr, eng));
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

song findClosestCentroid(song s, std::vector<song> centroids)
{
    float distance, minDistance = std::numeric_limits<float>::max();
    song troid;
    for (song centroid : centroids)
    {
        distance = centroid.distance(s);
        if (distance < minDistance)
        {
            troid = centroid;
            minDistance = distance;
        }
    }
    return troid;
}

std::map<song, std::vector<song>> serialKNN(std::vector<song> data, std::vector<song> centroids)
{
    std::map<song, std::vector<song>> hash;
    for (int i = 0; i < 20; i++)
    {
        for (int j = 0; j < data.size(); j++)
        {
            song centroid = findClosestCentroid(data[j], centroids);
            hash[centroid].push_back(data[j]);
        }
        for (int j = 0; j < centroids.size(); j++)
        {
            updateCentroid(centroids[j], hash[centroids[j]]);
            if (i < 19)
            {
                hash[centroids[j]].clear();
            }
        }
    }
    return hash;
}
