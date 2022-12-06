# CS5030 Final Project - The Genre Reveal Party
## Carter Bailey, Gavin Eddington, Trey Crowther

***

Program to compute the K-mean clustering algorithm to cluster songs using metrics provided by Spotify. Includes a serial and four parallel implementations:

* Serial
* Parallel shared memory CPU
* Parallel shared memory GPU
* Distributed memory CPU
* Distributed memory GPU

## The Problem

K-Means clustering is a type of unsupervised learning where the main goal is to find groups in data. It is an iterative technique where each point is assigned to one of the K groups based on their feature similarity.

### A general outline of the algorithm:

1. Pick K points as the initial centroids (ours is done randomly).
2. Find the distance (we use Euclidean) of each point with the K points (Centroids).

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](readMeImages/latex1.png)

3. Assign each point to the closest Centroid.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](readMeImages/latex2.png)

4. Find new centroids by taking the average of each cluster.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![](readMeImages/latex3.png) 

where S is points in the clusters

5. Repeat steps 2-4.

## Compilation instructions

### **Serial**

Required modules: `$module load cmake`

Compile: `$cmake .` `$make`


Run: `$./serial <amount of centroids>`

### **Shared CPU**
(uses the same cmake file as serial)

Required modules: `$module load cmake`

Compile: `$cmake .` `$make`


Run: `$./openmp <amount of centroids> <amount of threads>`

### **Shared GPU**

Required modules: `module load cuda`

Compile: `nvcc gpu.cu -o main`

Run: `./main`

### **Distributed CPU**

### **Distributed GPU**


## Verification

To verify that the various versions are creating the same centroids run verifier.py. This script checks to make sure that the same amount of songs are assigned to each cluster, using the serial results as the baseline.

This can be done by typing `python3 verifier.py` into the command line.

Note that this file requires pandas which is not on the hpc clusters so you have to copy over the results.csv's onto your own computer to run verifier.py. 

