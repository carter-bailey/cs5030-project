import pandas as pd

def verifier(original, new):
    original.sort_values(["centroid", "energy"],axis = 0, ascending = True,inplace = True)
    original.reset_index(drop=True,inplace=True)
    new.sort_values(["centroid", "energy"],axis = 0, ascending = True,inplace = True)
    new.reset_index(drop=True,inplace=True)
    return original.equals(new)

serial = pd.read_csv("serialResults.csv")
mp = pd.read_csv("OpenMPResults.csv")
#mpi = pd.read_csv("mpiResults.csv")

print(f"openmp is same as serial? {verifier(serial, mp)}")
#print(f"openmpi is same as serial? {verifier(serial, mpi)}")
