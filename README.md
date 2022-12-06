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

K-Means clustering is a type of unsupervised learning where the main goal is to find groups in data. It is an iterative technique where each point is assigned to one of the $K$ groups based on their feature similarity.

### A general outline of the algorithm:

1. Pick $K$ points as the initial centroids (ours is done randomly).
2. Find the distance (we use Euclidean) of each point with the $K$ points (Centroids).

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$D_{ij}^2 = \sum_{v=1}^n(X_{vi} - X_{vj})^2$

3. Assign each point to the closest Centroid.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$arg \ \underset{c_i\in C}{min} \ dist(c_i,x)^2$

4. Find new centroids by taking the average of each cluster.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$c_i = \frac{1}{|S_i|}\sum_{x_i \in S_i}x_i$ where $S_i$ is points in the $ith$ cluster

5. Repeat steps 2-4.

## Compilation instructions

### **Serial**

### **Shared CPU**

### **Shared GPU**

Required modules: `module load cuda`

Compile: `nvcc gpu.cu -o main`

Run: `./main`

### **Distributed CPU**

### **Distributed**


