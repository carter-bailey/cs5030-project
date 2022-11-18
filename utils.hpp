#include <vector>
#include <iostream>
#include <fstream>

#define delimiter ','

struct song{
  std::string genre;
  std::string name;
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
std::stringstream readFileIntoString(const std::string& path) {
    auto ss = std::stringstream();
    std::ifstream input_file(path);
    if (!input_file.is_open())
    {
        std::cerr << "Couldn't open the file - '" << path << "'" << std::endl;
        exit(EXIT_FAILURE);
    }
    ss << input_file.rdbuf();
    return ss;
}


std::vector<song> getCSV(){

}
