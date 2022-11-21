#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#define DELIMITER ','

struct song
{
    std::string genre;
    float danceability;
    float energy;
    int key;
    float loudness;
    int mode;
    float speechiness;
    float acousticness;
    float intrumental;
    float liveness;
    float valence;
    float tempo;
    float duration;
    int timeSignature;
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
    return ss;
}

std::vector<song> getCSV()
{
    std::stringstream csv = readFileIntoStringstream("genres_v2.csv");
    std::vector<song> content;
    std::vector<std::string> data;

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

        // Assign the data
        song temp;
        temp.danceability = std::stof(data.at(0));
        temp.energy = std::stof(data.at(1));
        temp.key = std::stof(data.at(2));
        temp.loudness = std::stof(data.at(3));
        temp.mode = std::stoi(data.at(4));
        temp.speechiness = std::stof(data.at(5));
        temp.acousticness = std::stof(data.at(6));
        temp.intrumental = std::stof(data.at(7));
        temp.liveness = std::stof(data.at(8));
        temp.valence = std::stof(data.at(9));
        temp.tempo = std::stoi(data.at(10));
        temp.duration = std::stof(data.at(16));
        temp.timeSignature = std::stoi(data.at(17));
        temp.genre = data.at(18);

        content.push_back(temp);
        data.clear();
    }

    return content;
}
