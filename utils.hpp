#pragma once
#include "song.hpp"

#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <unordered_map>
#include <vector>

#define DELIMITER ','

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

/**
 * @brief gets the data of the csv and returns it as a vector of songs
 *
 * @return a vector of songs, the data in the csv has been converted to the song class
 */
std::vector<song> getCSV()
{
	std::stringstream csv = readFileIntoStringstream("genres_v2.csv");
	std::vector<song> content;
	std::vector<std::string> data;

	// initialize our min and max
	song min(std::numeric_limits<float>::max());
	song max(std::numeric_limits<float>::min());

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
	// standardize the data in the vector
	for (song& s : content)
	{
		s.standardize(min, max);
	}

	return content;
}

/**
 * @brief writes the centroids and their songs to a csv for visualization purposes
 *
 * @param hash - the map containing all the centroids songs
 * @param centroids - the centroids that we have been using
 */
void writeToCSV(std::unordered_map<int, std::vector<song>> hash, std::vector<song> centroids, std::string name = "results.csv")
{
	std::ofstream output_file(name);
	output_file << "centroid,danceability,energy,loudness,speechiness,\
    acousticness,instrumental,liveness,valence,tempo\n";
	for (long unsigned int i = 0; i < centroids.size(); i++)
	{
		for (auto s : hash[i])
		{
			output_file << i << "," << s.toString();
		}
	}
	output_file.close();
}
