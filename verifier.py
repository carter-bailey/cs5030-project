import pandas as pd

def verifier(original, new):
centroid,danceability,energy,loudness,speechiness,    acousticness,instrumental,liveness,valence,tempo
    original.sort_values(["centroid", "danceability", "energy", "loudness", "speechiness", "acousticness", "instrumental", "liveness",  "valence", "tempo"],axis = 0, ascending = True,inplace = True)
    original.reset_index(drop=True,inplace=True)
    new.sort_values([["centroid", "danceability", "energy", "loudness", "speechiness", "acousticness", "instrumental", "liveness",  "valence", "tempo"]],axis = 0, ascending = True,inplace = True)
    new.reset_index(drop=True,inplace=True)
    return original.equals(new)

serial = pd.read_csv("serialResults.csv")
mp = pd.read_csv("OpenMPResults.csv")
#mpi = pd.read_csv("mpiResults.csv")

serial.sort_values(["centroid", "energy"],axis = 0, ascending = True,inplace = True)
serial.reset_index(drop=True,inplace=True)
mp.sort_values(["centroid", "energy"],axis = 0, ascending = True,inplace = True)
mp.reset_index(drop=True,inplace=True)
#mpi.sort_values(["centroid", "energy"],axis = 0, ascending = True,inplace = True)
#mpi.reset_index(drop=True,inplace=True)

serial.to_csv("serialSorted.csv",index=False)
mp.to_csv("mpSorted.csv",index=False)
#mpi.to_csv("mpiSorted.csv",index=False)
print(f"openmp is same as serial? {verifier(serial, mp)}")
#print(f"openmpi is same as serial? {verifier(serial, mpi)}")
