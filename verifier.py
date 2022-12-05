import pandas as pd

def verifier(original, new):
    original.sort_values(["centroid"],axis = 0, ascending = True,inplace = True)
    original.reset_index(drop=True,inplace=True)
    new.sort_values(["centroid"],axis = 0, ascending = True,inplace = True)
    new.reset_index(drop=True,inplace=True)
    return original["centroid"].equals(new["centroid"])

serial = pd.read_csv("serialResults.csv")
mp = pd.read_csv("OpenMPResults.csv")
#mpi = pd.read_csv("mpiResults.csv")
gpu = pd.read_csv("cudaResults.csv")

print(f"openmp is same as serial? {verifier(serial, mp)}")
#print(f"openmpi is same as serial? {verifier(serial, mpi)}")
print(f"gpu is same as serial? {verifier(serial, gpu)}")
