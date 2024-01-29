# -------------------
# Script for (raw) Data GENERATION 
# ADD CAREFUL EXPLANATION
#
import torch
from torch import pi
from math import sqrt


def simple_augment (single_series):
	'''
	Given a collection of measurements, add the time component
	'''
	data = single_series.reshape(-1,)
	n_times = data.shape[0]
	result = torch.zeros(n_times, 2)
	result[:, 0] = torch.linspace(0., 1., n_times)
	result[:, 1] = data
	return result
#-

def remove_some(data, perc=10):
	'''
	Remove perc% of the points, default 10%
	'''
	data_len = len(data)
	# remove the perc% of the data
	new_len = int(data_len / 100 * (100-perc))
	# Be sure of always having at least two data points
	if new_len <= 2:
		new_len = 2
	shuffled_indeces = torch.randperm(new_len)
	preserved_indeces = shuffled_indeces[:new_len]
	tensor_indeces = preserved_indeces.sort()[0]
	new_data = data[tensor_indeces]
	return new_data
#---
	
def add_noise(data, noise_std=0.01):
	'''
	add a noise on a given one-dimensional time series.
	'''
	noises = torch.normal(0., noise_std, (data.shape))
	return data + noises
#---


	
def segment_dataset (n_samples, max_nodes):
	'''
	Generate the dataset for random segments.
	'''
	x_raw = []			# list of arrays of diff. lengths
	y_full = torch.zeros(n_samples) # labels

	for nth in range(n_samples):
		# determine the ending point of the segment
		end_pt = -1. + 2. * torch.rand((1,))
		# if positive, the label is 1. Otherwise stays 0.
		if end_pt > 0:
			y_full[nth] = 1
		# Determine on how many nodes the curve is generated
		num_nodes = torch.randint(max_nodes - 2, (1,)).item() + 2
		# Append the segment
		tmp = torch.linspace(0., 1., num_nodes) * end_pt
		# Remove 10% of nodes and add noise with std 0.01
		tmp2 = add_noise(tmp, 0.02)
		x_raw.append(tmp2)
	print(f"Raw dataset: {n_samples} samples, max nodes {max_nodes}")
	print(f"Balancing: {y_full.mean() * 100 : .1f}%")
	return x_raw, y_full, "SEG"
#-

def sinusoid_dataset (n_samples, max_nodes):
	'''
	Generate the dataset for sinusoids with random frequency.
	'''
	x_raw = []			# list of arrays of diff. lengths
	y_full = torch.zeros(n_samples) # labels
	for nth in range(n_samples):
		# sin (2 pi freq)
		# Random frequency between 1, 2, 3 and 4
		freq = torch.randint(4, (1,)) + 1
		# if higher than 2, label as 1. Otherwise stays 0.
		if freq > 2:
			y_full[nth] = 1
		# Determine on how many nodes the curve is generated
		nodes = torch.randint(max_nodes - 2, (1,)).item() + 2
		tmp = torch.sin(2.*pi*torch.linspace(0.,1,nodes)*freq)
		tmp2 = add_noise(tmp, 0.02)
		x_raw.append(tmp2)
	print(f"Raw dataset: {n_samples} samples, max nodes {max_nodes}")
	print(f"Balancing: {y_full.mean() * 100 : .1f}%")
	return x_raw, y_full, "SIN"
#-


def gaussian_shape(time, mu, sigma = 0.01):
	denumerator = 1.
	numerator = - ((time - mu) ** 2) / (sigma ** 2)
	partial = (torch.exp(numerator) / denumerator)	
	return partial
#---

def impulse_dataset (n_samples, max_nodes):
	'''
	Generate the dataset of impulses with random activation.
	'''
	x_raw = []			# list of arrays of diff. lengths
	y_full = torch.zeros(n_samples) # labels

	for nth in range(n_samples):
		# determine the activation point, between [0.2 and 0.8]
		act = torch.rand((1,)) * 0.6 + 0.2
		# if higher than half, the label is 1. Otherwise stays 0.
		if act > 0.5:
			y_full[nth] = 1
		# Determine on how many nodes the curve is generated
		nodes = torch.randint(max_nodes - 2, (1,)).item() + 2
		# generate the curve
		tmp = gaussian_shape(torch.linspace(0.,1.,nodes), act)
		tmp2 = add_noise(tmp, 0.02)
		x_raw.append(tmp2)
	print(f"Raw dataset: {n_samples} samples, max nodes {max_nodes}")
	print(f"Balancing: {y_full.mean() * 100 : .1f}%")
	return x_raw, y_full, "IMP"
#-


# Other utilities
def accuracy (y_true, y_given):
	# Determine how many labels of y_given matches y_true
	diff = torch.abs(y_true - y_given)
	# The number of 0 in diff says how many are the same
	# so, the number of 1 gives us the inaccuracy:
	inaccuracy = diff.mean()
	return (1. - inaccuracy) * 100.
#---


def distance_matrix (data):
	n = len(data)
	mtx = torch.zeros(n * (n-1) - 2)
	counter = -1
	for i in range(n):
		for j in range(i + 1, n):
			counter += 1
			mtx[counter] = torch.norm(data[i] - data[j])
	if (counter == int(n*(n-1)/2 - 1)):
		print("--- ok distance matrix counter ---")
#	print(f"Counter {counter}, should be {int(n*(n-1)/2 - 1}")	
	return mtx
#---

def mtx_err (og_distances, new_distances):
	avg1 = og_distances.mean()
	avg2 = new_distances.mean()
	print(f"Distances: {avg1 :.1f} -> {avg2 :.1f}")
	err = torch.abs(avg1 - avg2) / avg1 * 100.
	print(f"Error of {err :.1f} %")
	return err
#---
