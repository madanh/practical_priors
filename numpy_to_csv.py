import numpy as np
import sys

#fname = sys.argv[1]
fname = '20170710_0_data.npy'
d = np.load(fname)
pass
d = d/1000
ofname = '_data.csv'
np.savetxt(ofname,d)

#fname = sys.argv[1]
fname = '20170710_0_truth.npy'
d = np.load(fname)
pass
d = d['x'][:,None]/1000
ofname = '_truth.csv'
np.savetxt(ofname,d)
