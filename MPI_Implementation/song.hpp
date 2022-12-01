#pragma once
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <vector>

class song
{
public:
	double danceability;
	double energy;
	double loudness;
	double speechiness;
	double acousticness;
	double instrumental;
	double liveness;
	double valence;
	double tempo;

	static const int NUM_FEATURES = 9;

	// resets the objects data to 0 so that we can recenter a centroid in the middle of it's data
	void reset()
	{
		speechiness = acousticness = instrumental = liveness = valence = loudness = energy = danceability = tempo = 0;
	}

	// base constructor sets everything to zero
	song()
	{
		reset();
	}

	song(double defaultNum);

	song(std::vector<std::string>);
	song(double[]);
	// randomly assign all of the variables a random number

	double distance(song);
	void min(song);
	void max(song);
	void standardize(song, song);
	std::string toString();
	double* toArray();
	bool operator<(const song& s) const;
	bool operator==(const song& s) const;
};

song::song(double defaultNum)
{
	danceability = energy = loudness = speechiness = acousticness = instrumental = liveness = valence = tempo = defaultNum;
}

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

song::song(double* data) :
		danceability(data[0]),
		energy(data[1]),
		loudness(data[2]),
		speechiness(data[3]),
		acousticness(data[4]),
		instrumental(data[5]),
		liveness(data[6]),
		valence(data[7]),
		tempo(data[8])
{
}

/**
 * @brief Provides a distance measurement between two songs
 *
 * @param the other song to compare distances with
 */
double song::distance(song s)
{
	double total = 0;
	total += (danceability - s.danceability) * (danceability - s.danceability);
	total += (energy - s.energy) * (energy - s.energy);
	total += (loudness - s.loudness) * (loudness - s.loudness);
	total += (speechiness - s.speechiness) * (speechiness - s.speechiness);
	total += (acousticness - s.acousticness) * (acousticness - s.acousticness);
	total += (instrumental - s.instrumental) * (instrumental - s.instrumental);
	total += (liveness - s.liveness) * (liveness - s.liveness);
	total += (valence - s.valence) * (valence - s.valence);
	total += (tempo - s.tempo) * (tempo - s.tempo);
	return sqrtf(total);
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
	this->valence = this->valence > s.valence ? s.valence : this->valence;
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
	s << danceability << "," << energy << "," << loudness << "," << speechiness << "," << acousticness << "," << instrumental << "," << liveness << "," << valence << "," << tempo << "\n";
	return s.str();
}

double* song::toArray()
{
	double* arr = (double*)malloc(sizeof(double) * NUM_FEATURES);
	arr[0] = danceability;
	arr[1] = energy;
	arr[2] = loudness;
	arr[3] = speechiness;
	arr[4] = acousticness;
	arr[5] = instrumental;
	arr[6] = liveness;
	arr[7] = valence;
	arr[8] = tempo;
	return arr;
}

// These two operator overloads are required so that that the hashmap works
bool song::operator<(const song& s) const
{
	return energy < s.energy;
}

bool song::operator==(const song& s) const
{
	return energy == s.energy && tempo == s.tempo && loudness == s.loudness && speechiness == s.speechiness &&
				 acousticness == s.acousticness && instrumental == s.instrumental && liveness == s.liveness && valence == s.valence && tempo == s.tempo;
}
