import scipy.io
import numpy as np

def read_mat(mat_file):
    labels_dict = scipy.io.loadmat(mat_file)
    label = labels_dict['labels']
    labels = np.squeeze(label - 1)
    np.save('labels.npy',labels)

def read_npy(npy_file):
	return np.load(npy_file)

