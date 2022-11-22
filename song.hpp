#pragma once
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <vector>

class song
{
  public:
    float danceability;
    float energy;
    float loudness;
    float speechiness;
    float acousticness;
    float instrumental;
    float liveness;
    float valence;
    float tempo;

    // resets the objects data to 0 so that we can recenter a centroid in the middle of it's data
    void reset()
    {
        speechiness = acousticness = instrumental = liveness = valence = loudness = energy = danceability = 0;
    }

    // base constructor sets everything to zero
    song()
    {
        reset();
    }

    song(std::vector<std::string>);
    // randomly assign all of the variables a random number
    song(std::uniform_real_distribution<float> distr, std::default_random_engine eng);

    float distance(song);
    void min(song);
    void max(song);
    void standardize(song, song);
    std::string toString();
    bool operator<(const song& s) const;
    bool operator==(const song& s) const;
};

// constructor for when a vector of strings is passed in, to be used when reading in csv data
song::song(std::vector<std::string> data) :
    danceability(std::stof(data[0])),
    energy(std::stof(data[1])),
    loudness(std::stof(data[3])),
    speechiness(std::stof(data[5])),
    acousticness(std::stof(data[6])),
    instrumental(std::stof(data[7])),
    liveness(std::stof(data[8])),
    valence(std::stof(data[9])),
    tempo(std::stof(data[10]))
{
}

// constructor that randomly assign all of the variables a random number
song::song(std::uniform_real_distribution<float> distr, std::default_random_engine eng) :
    danceability(distr(eng)),
    energy(distr(eng)),
    loudness(distr(eng)),
    speechiness(distr(eng)),
    acousticness(distr(eng)),
    instrumental(distr(eng)),
    liveness(distr(eng)),
    valence(distr(eng)),
    tempo(distr(eng))
{
}

/**
 * @brief Provides a distance measurement between two songs
 *
 * @param the other song to compare distances with
 */
float song::distance(song s)
{
    float total = 0;
    total += (danceability - s.danceability) * (danceability - s.danceability);
    total += (energy - s.energy) * (energy - s.energy);
    total += (loudness - s.loudness) * (loudness - s.loudness);
    total += (speechiness - s.speechiness) * (speechiness - s.speechiness);
    total += (acousticness - s.acousticness) * (acousticness - s.acousticness);
    total += (instrumental - s.instrumental) * (instrumental - s.instrumental);
    total += (liveness - s.liveness) * (liveness - s.liveness);
    total += (valence - s.valence) * (valence - s.valence);
    total += (tempo - s.tempo) * (tempo - s.tempo);
    return total;
}

/**
 * @brief sets the variables to the min between the two
 *
 * @param the other song to compare values with
 */
void song::min(song s)
{
    this->danceability = this->danceability > s.danceability ? s.danceability : this->danceability;
    this->energy = this->energy > s.energy ? s.energy : this->energy;
    this->loudness = this->loudness > s.loudness ? s.loudness : this->loudness;
    this->speechiness = this->speechiness > s.speechiness ? s.speechiness : this->speechiness;
    this->acousticness = this->acousticness > s.acousticness ? s.acousticness : this->acousticness;
    this->instrumental = this->instrumental > s.instrumental ? s.instrumental : this->instrumental;
    this->liveness = this->liveness > s.liveness ? s.liveness : this->liveness;
    this->tempo = this->tempo > s.tempo ? s.tempo : this->tempo;
}

/**
 * @brief sets the variables to the max between the two
 *
 * @param the other song to compare values with
 */
void song::max(song s)
{
    this->danceability = this->danceability < s.danceability ? s.danceability : this->danceability;
    this->energy = this->energy < s.energy ? s.energy : this->energy;
    this->loudness = this->loudness < s.loudness ? s.loudness : this->loudness;
    this->speechiness = this->speechiness < s.speechiness ? s.speechiness : this->speechiness;
    this->acousticness = this->acousticness < s.acousticness ? s.acousticness : this->acousticness;
    this->instrumental = this->instrumental < s.instrumental ? s.instrumental : this->instrumental;
    this->liveness = this->liveness < s.liveness ? s.liveness : this->liveness;
    this->valence = this->valence < s.valence ? s.valence : this->valence;
    this->tempo = this->tempo < s.tempo ? s.tempo : this->tempo;
}

/**
 * @brief standardizes the object with the min and max song that's passed in
 *
 * @param min: the min values that have been found
 * @param max: the max values that have been found
 */
void song::standardize(song min, song max)
{
    danceability = (danceability - min.danceability) / (max.danceability - min.danceability);
    energy = (energy - min.energy) / (max.energy - min.energy);
    loudness = (loudness - min.loudness) / (max.loudness - min.loudness);
    speechiness = (speechiness - min.speechiness) / (max.speechiness - min.speechiness);
    acousticness = (acousticness - min.acousticness) / (max.acousticness - min.acousticness);
    instrumental = (instrumental - min.instrumental) / (max.instrumental - min.instrumental);
    liveness = (liveness - min.liveness) / (max.liveness - min.liveness);
    valence = (valence - min.valence) / (max.valence - min.valence);
    tempo = (tempo - min.tempo) / (max.tempo - min.tempo);
}

std::string song::toString()
{
    std::stringstream s;
    s << danceability << "," << energy << "," << loudness << "," << speechiness << "," << acousticness << "," << instrumental << "," << instrumental << "," << liveness << "," << valence << "," << tempo << "\n";
    return s.str();
}
bool song::operator<(const song& s) const
{
    return energy < s.energy;
}

bool song::operator==(const song& s) const
{
    return energy == s.energy && tempo == s.tempo;
}
