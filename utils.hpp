#pragma once
#include "song.hpp"

#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <vector>

#define DELIMITER ','

struct KeyHash
{
    std::size_t operator()(const song& s) const
    {
        return std::hash<float>()(s.energy) ^ (std::hash<float>()(s.tempo) << 1);
    }
};

struct KeyEqual
{
    bool operator()(const song& lhs, const song& rhs) const
    {
        return lhs.energy == rhs.energy && lhs.tempo == rhs.tempo;
    }
};

// Taken from https://www.delftstack.com/howto/cpp/read-csv-file-in-cpp/
/**
 * @brief Reads a file into a string
 *
 * @param path The file to read
 * @return std::string
 */
std::stringstream readFileIntoStringstream(const std::string& path)
{
    std::stringstream ss = std::stringstream();
    std::ifstream input_file(path);
    if (!input_file.is_open())
    {
        std::cerr << "Couldn't open the file - '" << path << "'" << std::endl;
        exit(EXIT_FAILURE);
    }
    ss << input_file.rdbuf();
    input_file.close();
    return ss;
}

std::vector<song> getCSV()
{
    std::stringstream csv = readFileIntoStringstream("genres_v2.csv");
    std::vector<song> content;
    std::vector<std::string> data;

    // initialize our min and max
    song min;
    song max;

    std::string record;

    // Remove the header of the csv file by reading in the first line
    std::getline(csv, record);

    while (std::getline(csv, record))
    {
        std::stringstream line(record);
        while (std::getline(line, record, DELIMITER))
        {
            data.push_back(record);
        }

        song temp(data);

        // check to see if we have any new mins or maxes.
        min.min(temp);
        max.max(temp);

        content.push_back(temp);
        data.clear();
    }
    for (song s : content)
    {
        s.standardize(min, max);
    }

    return content;
}
void writeToCSV(std::map<song, std::vector<song>> hash, std::vector<song> centroids)
{
    std::ofstream output_file("results.csv");
    output_file << "centroid,danceability,energy,loudness,speechiness,\
    acousticness,instrumental,liveness,valence,tempo\n";
    int counter = 0;
    for (auto c : centroids)
    {
        for (auto s : hash[c])
        {
            output_file << counter << "," << s.toString();
        }
    }
    output_file.close();
}
