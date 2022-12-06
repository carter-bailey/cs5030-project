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

<<<<<<< HEAD
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

### **Shared GPU**

Required modules: `module load cuda`

Compile: `nvcc gpu.cu -o main`

Run: `./main`

### **Distributed CPU**

### **Distributed**


