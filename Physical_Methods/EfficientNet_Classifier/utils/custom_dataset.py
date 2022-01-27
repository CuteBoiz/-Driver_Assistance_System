"""
Preprocess dataset from txt files.

author: phatnt
date: 2021-08-10
"""

import os 
import numpy as np
import cv2
import torch

class CustomDataset(torch.utils.data.Dataset):
	"""
	Customize your own Dataset to Pytorch-type dataset.
	"""
	def __init__(self, txt_file, transform=None):
		"""
		Args:
			- txt_file: path to txt_file ("image_path", id)
			- transform: Transforms type(Resize, Normalize, To_Tensor)
		"""
		assert os.path.isfile(txt_file)
		self.image_paths = []
		self.targets = []
		self.transform = transform

		with open(txt_file, 'r') as f:
			lines = f.readlines()
		for line in lines:
			line = line.strip()
			image_path, target = line.split("\"")[1], int(line.split("\"")[2].strip())
			self.image_paths.append(image_path)
			self.targets.append(target)
	
	def __len__(self):
		return len(self.image_paths)
	
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		image = cv2.imread(self.image_paths[idx])
		target = int(self.targets[idx])
		path = self.image_paths[idx]
		sample = {'image': image, 'path': path, 'target': target}
		if self.transform:
			sample = self.transform(sample)
		return sample

class Resize(object):
	def __init__(self, output_size):
		assert isinstance(output_size, int)
		self.output_size = output_size
		
	def __call__(self, sample):
		image, path, target = sample['image'], sample['path'], sample['target']
		resized = cv2.resize(image, (self.output_size, self.output_size), interpolation = cv2.INTER_AREA)
		return {'image': resized, 'path': path, 'target': target}
	
class Normalize(object):
	def __call__(self, sample):
		image, path, target = sample['image'], sample['path'], sample['target']
		image = np.float32(image)
		image = image*(1/255)
		mean = [0.485, 0.456, 0.406]
		std = [0.229, 0.224, 0.225]
		image = (image - mean) / std
		return {'image': image, 'path': path, 'target': target}

class ToTensor(object):
	def __call__(self, sample):
		image, path, target = sample['image'], sample['path'], sample['target']
		image = image.transpose((2, 0, 1))
		image = np.asarray(image).astype(np.float32)
		return {'image': torch.from_numpy(image),
				'path': path,
				'target': target}

	


	

