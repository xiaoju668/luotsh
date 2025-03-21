import numpy as np

energy1 = np.float64(0.0)
energy2 = np.float64(0.9/27.2113863)
num_vib = 3
time = 2500
traj = 200
dt = np.float64(0.2 * 100.0 / 2.4189)   # mean 0.2fs
omiga = np.array( [0.126,0.074,0.118] , dtype=np.float64 ) / 27.2113863
kai1 = np.array( [0.037,-0.105,0] , dtype=np.float64 ) / 27.2113863
kai2 = np.array( [-0.254,0.149,0] , dtype=np.float64 ) / 27.2113863
lamda = np.float64(0.262 / 27.2113863)
wf = np.array( [ [0] , [1] ] )
state = 'state2'