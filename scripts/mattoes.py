import loris
import numpy as np
import matplotlib.pyplot as plt 
import os
import fnmatch
from tqdm import tqdm
import numpy as np
import scipy.io as sio

events = []
FILENAME = "res_500x500_noise_500000_signal_0"
mat = sio.loadmat('/media/sam/Samsung_T51/PhD/Code/orbital_localisation/test_files/' + FILENAME + '.mat')

events = mat['e']
print(events[0][0]["x"].shape)

matX  =  events[0][0]["x"]
matY  =  events[0][0]["y"]
matP  =  events[0][0]["p"]
matTs =  events[0][0]["t"]

# nEvents = events[0][0]["x"].shape[1]
nEvents = matX.shape[0]
x  = matX.reshape((nEvents, 1))
y  = matY.reshape((nEvents, 1))
p  = matP.reshape((nEvents, 1))
ts = matTs.reshape((nEvents, 1))

events = np.zeros((nEvents,4))

events = np.concatenate((ts,x, y, p),axis=1).reshape((nEvents,4))

finalArray = np.asarray(events)
print(finalArray)
# finalArray[:,0] -= finalArray[0,0]

ordering = "txyp"
loris.write_events_to_file(finalArray, "/media/sam/Samsung_T51/PhD/Code/orbital_localisation/test_files/" + FILENAME + ".es",ordering)
print("File: " + FILENAME + "converted to .es")
