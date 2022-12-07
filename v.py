import pandas as pd

def verifier(original, new):
    original.sort_values(["centroid", "danceability", "energy", "loudness", "speechiness", "acousticness", "instrumental", "liveness",  "valence", "tempo"],axis = 0, ascending = True,inplace = True)
    original.reset_index(drop=True,inplace=True)
    new.sort_values(["centroid", "danceability", "energy", "loudness", "speechiness", "acousticness", "instrumental", "liveness",  "valence", "tempo"],axis = 0, ascending = True,inplace = True)
    new.reset_index(drop=True,inplace=True)
    return original.equals(new)

serial = pd.read_csv("results/serialResults.csv")
mp = pd.read_csv("results/OpenMPResults.csv")
mpi = pd.read_csv("results/MPIResults.csv")
gpu = pd.read_csv("results/cudaResults.csv")
gpuMPI = pd.read_csv("results/MPIAndCudaResults.csv")

#serial.sort_values(["centroid", "energy"],axis = 0, ascending = True,inplace = True)
#serial.reset_index(drop=True,inplace=True)
#mp.sort_values(["centroid", "energy"],axis = 0, ascending = True,inplace = True)
#mp.reset_index(drop=True,inplace=True)
#mpi.sort_values(["centroid", "energy"],axis = 0, ascending = True,inplace = True)
#mpi.reset_index(drop=True,inplace=True)
#
#serial.to_csv("serialSorted.csv",index=False)
#mp.to_csv("mpSorted.csv",index=False)
#mpi.to_csv("mpiSorted.csv",index=False)
print(f"openmp is same as serial? {verifier(serial, mp)}")
print(f"openmpi is same as serial? {verifier(serial, mpi)}")
print(f"gpu is same as serial? {verifier(serial, gpu)}")
print(f"gpumpi is same as serial? {verifier(serial, gpuMPI)}")
