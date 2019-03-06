import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class BalancedImbalancedDataset(Dataset):
	def __init__(self, path):
		self.X, self.y = self._load_data(path)
		self.where_zero, self.where_one = self._get_subset_indices()

	def _load_data(self, path):
		X, y = [], []
		with open(path) as f:
			for line in f.readlines():
				values = line.split(' ')
				features = [float(x) for x in values[:2]]
				X.append(features)

				label = [float(values[-1].strip('\n'))]
				y.append(label)

		X = torch.Tensor(X)
		y = torch.Tensor(y)
		return X, y

	def _get_subset_indices(self):
		where_zero = np.where(self.y==0.0)[0]
		where_one = np.where(self.y==1.0)[0]
		return where_zero, where_one

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		if idx % 2 == 0:
			zero_sample_index = np.random.choice(self.where_zero)
			assert(self.y[zero_sample_index] == 0.0)
			return self.X[zero_sample_index], self.y[zero_sample_index]
		else:
			one_sample_index = np.random.choice(self.where_one)
			assert(self.y[one_sample_index] == 1.0)
			return self.X[one_sample_index], self.y[one_sample_index]

	def plot_data(self):
		X0 = np.array(self.X[self.where_zero])
		X1 = np.array(self.X[self.where_one])
		plt.plot(X0[0], X0[1])
		plt.plot(X1[0], X1[1])
		plt.xlabel("Feature 0")
		plt.ylabel("Feature 1")
		plt.show()

class ImbalancedDataset(Dataset):
	def __init__(self, path):
		self.X, self.y = self._load_data(path)
		self.where_zero, self.where_one = self._get_subset_indices()

	def _load_data(self, path):
		X, y = [], []
		with open(path) as f:
			for line in f.readlines():
				values = line.split(' ')
				features = [float(x) for x in values[:2]]
				X.append(features)

				label = [float(values[-1].strip('\n'))]
				y.append(label)

		X = torch.Tensor(X)
		y = torch.Tensor(y)
		return X, y

	def _get_subset_indices(self):
		where_zero = np.where(self.y==0.0)[0]
		where_one = np.where(self.y==1.0)[0]
		return where_zero, where_one

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]

	def plot_data(self):
		X0 = np.array(self.X[self.where_zero])
		X1 = np.array(self.X[self.where_one])
		plt.plot(X0[0], X0[1])
		plt.plot(X1[0], X1[1])
		plt.xlabel("Feature 0")
		plt.ylabel("Feature 1")
		plt.show()