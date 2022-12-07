import pandas as pd

def verifier(original, new):
    original.sort_values(["centroid"],axis = 0, ascending = True,inplace = True)
    original.reset_index(drop=True,inplace=True)
    new.sort_values(["centroid"],axis = 0, ascending = True,inplace = True)
    new.reset_index(drop=True,inplace=True)
    return original["centroid"].equals(new["centroid"])

serial = pd.read_csv("results/serialResults.csv")
mp = pd.read_csv("results/OpenMPResults.csv")
mpi = pd.read_csv("results/MPIResults.csv")
gpu = pd.read_csv("results/cudaResults.csv")
gpuMPI = pd.read_csv("results/MPIAndCudaResults.csv")

print(f"openmp is same as serial? {verifier(serial, mp)}")
print(f"openmpi is same as serial? {verifier(serial, mpi)}")
print(f"gpu is same as serial? {verifier(serial, gpu)}")
print(f"gpumpi is same as serial? {verifier(serial, gpuMPI)}")
