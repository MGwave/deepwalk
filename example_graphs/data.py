import numpy as np
import pickle as pkl
import scipy.sparse as sp
with open('polblogsDataObj_0','rb') as f:
    objs = pkl.load(f, encoding='latin1')
adj = objs[0]
labels = sp.csc_matrix(objs[2])
sio.savemat('polblogs.mat',{'network':adj, 'group':labels})
sio.loadmat("polblogs.mat")