import os
import os.path as osp
import torch
from torch_geometric.data import Data, Dataset
import torch_geometric.transforms as T
import pandas as pd
import esm
from torch_geometric.utils import to_networkx, to_undirected, degree
import networkx as nx
import matplotlib.pyplot as plt
import math
class RBDDataset_train(Dataset):
	def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
		super().__init__(root, transform, pre_transform, pre_filter)
	
	@property
	def raw_file_names(self):
		return ['file.pt']

	@property
	def processed_file_names(self):
		return ['file.pt']
    
	def process(self):
		idx = 0
		#for raw_path in self.raw_paths:
		for files in os.listdir('train_r1_test_r2/train/raw/'):
			# Read data from `raw_path`.
			if osp.isfile(osp.join(self.processed_dir, f'data_{idx}.pt'))==False:
				data = torch.load('train_r1_test_r2/train/raw/'+files)
				if self.pre_filter is not None and not self.pre_filter(data):
					continue
				if self.pre_transform is not None:
					data = self.pre_transform(data)
				torch.save([data,files], osp.join(self.processed_dir, f'data_{idx}.pt'))
			idx += 1

	def len(self):
		ctr=0
		for files in os.listdir('train_r1_test_r2/train/raw/'):
			ctr+=1
		return ctr

	def get(self, idx):
		# for files in os.listdir('sample_graphs/processed/'):
		# 	cur_ind = int(files.split('_')[-1].replace('.pt',''))
		# 	if cur_ind == int(idx):
		# 		data = torch.load(osp.join(self.processed_dir, files))
		# 		break
		data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
		return data
class RBDDataset_val(Dataset):
	def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
		super().__init__(root, transform, pre_transform, pre_filter)
	
	@property
	def raw_file_names(self):
		return ['file.pt']

	@property
	def processed_file_names(self):
		return ['file.pt']
    
	def process(self):
		idx = 0
		#for raw_path in self.raw_paths:
		for files in os.listdir('train_r1_test_r2/val/raw/'):
			# Read data from `raw_path`.
			if osp.isfile(osp.join(self.processed_dir, f'data_{idx}.pt'))==False:
				data = torch.load('train_r1_test_r2/val/raw/'+files)
				if self.pre_filter is not None and not self.pre_filter(data):
					continue
				if self.pre_transform is not None:
					data = self.pre_transform(data)
				torch.save([data,files], osp.join(self.processed_dir, f'data_{idx}.pt'))
			idx += 1

	def len(self):
		ctr=0
		for files in os.listdir('train_r1_test_r2/val/raw/'):
			ctr+=1
		return ctr

	def get(self, idx):
		# for files in os.listdir('sample_graphs/processed/'):
		# 	cur_ind = int(files.split('_')[-1].replace('.pt',''))
		# 	if cur_ind == int(idx):
		# 		data = torch.load(osp.join(self.processed_dir, files))
		# 		break
		data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
		return data
class RBDDataset_test(Dataset):
	def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
		super().__init__(root, transform, pre_transform, pre_filter)
	
	@property
	def raw_file_names(self):
		return ['file.pt']

	@property
	def processed_file_names(self):
		return ['file.pt']
    
	def process(self):
		idx = 0
		#for raw_path in self.raw_paths:
		for files in os.listdir('train_r1_test_r2/test/raw/'):
			# Read data from `raw_path`.
			if osp.isfile(osp.join(self.processed_dir, f'data_{idx}.pt'))==False:
				data = torch.load('train_r1_test_r2/test/raw/'+files)
				if self.pre_filter is not None and not self.pre_filter(data):
					continue
				if self.pre_transform is not None:
					data = self.pre_transform(data)
				torch.save([data,files], osp.join(self.processed_dir, f'data_{idx}.pt'))
			idx += 1

	def len(self):
		ctr=0
		for files in os.listdir('train_r1_test_r2/test/raw/'):
			ctr+=1
		return ctr

	def get(self, idx):
		# for files in os.listdir('sample_graphs/processed/'):
		# 	cur_ind = int(files.split('_')[-1].replace('.pt',''))
		# 	if cur_ind == int(idx):
		# 		data = torch.load(osp.join(self.processed_dir, files))
		# 		break
		data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
		return data

class RBDDataset_train_combo(Dataset):
	def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
		super().__init__(root, transform, pre_transform, pre_filter)
	
	@property
	def raw_file_names(self):
		return ['file.pt']

	@property
	def processed_file_names(self):
		return ['file.pt']
    
	def process(self):
		idx = 0
		#for raw_path in self.raw_paths:
		for files in os.listdir('train_r1_r2/train/raw/'):
			# Read data from `raw_path`.
			if osp.isfile(osp.join(self.processed_dir, f'data_{idx}.pt'))==False:
				data = torch.load('train_r1_r2/train/raw/'+files)
				if self.pre_filter is not None and not self.pre_filter(data):
					continue
				if self.pre_transform is not None:
					data = self.pre_transform(data)
				torch.save([data,files], osp.join(self.processed_dir, f'data_{idx}.pt'))
			idx += 1

	def len(self):
		ctr=0
		for files in os.listdir('train_r1_r2/train/raw/'):
			ctr+=1
		return ctr

	def get(self, idx):
		# for files in os.listdir('sample_graphs/processed/'):
		# 	cur_ind = int(files.split('_')[-1].replace('.pt',''))
		# 	if cur_ind == int(idx):
		# 		data = torch.load(osp.join(self.processed_dir, files))
		# 		break
		data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
		return data

class RBDDataset_val_combo(Dataset):
	def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
		super().__init__(root, transform, pre_transform, pre_filter)
	
	@property
	def raw_file_names(self):
		return ['file.pt']

	@property
	def processed_file_names(self):
		return ['file.pt']
    
	def process(self):
		idx = 0
		#for raw_path in self.raw_paths:
		for files in os.listdir('train_r1_r2/val/raw/'):
			# Read data from `raw_path`.
			if osp.isfile(osp.join(self.processed_dir, f'data_{idx}.pt'))==False:
				data = torch.load('train_r1_r2/val/raw/'+files)
				if self.pre_filter is not None and not self.pre_filter(data):
					continue
				if self.pre_transform is not None:
					data = self.pre_transform(data)
				torch.save([data,files], osp.join(self.processed_dir, f'data_{idx}.pt'))
			idx += 1

	def len(self):
		ctr=0
		for files in os.listdir('train_r1_r2/val/raw/'):
			ctr+=1
		return ctr

	def get(self, idx):
		# for files in os.listdir('sample_graphs/processed/'):
		# 	cur_ind = int(files.split('_')[-1].replace('.pt',''))
		# 	if cur_ind == int(idx):
		# 		data = torch.load(osp.join(self.processed_dir, files))
		# 		break
		data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
		return data


set_train=RBDDataset_train(root='train_r1_test_r2/train/')
set_train.process()
set_val=RBDDataset_val(root='train_r1_test_r2/val/')
set_val.process()
set_test=RBDDataset_test(root='train_r1_test_r2/test/')
set_test.process()

# set_train_c=RBDDataset_train_combo(root='train_r1_r2/train/')
# set_train_c.process()
# set_val_c=RBDDataset_val_combo(root='train_r1_r2/val/')
# set_val_c.process()