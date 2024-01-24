'''
Utils for the AlgoRadius project.

	get_radius (input_data, quantile = 0.95):
		returns a float indicating the radius to contain data


	to_two_dimension (input_data):
		returns transformed dataset, old, new distance mtx norms	

'''
import torch
import matplotlib.pyplot as plt
import signatory as sg
import importlib
import math
from sklearn.manifold import MDS
import numpy as np

def get_radius (input_data, quantile = 0.95):
	'''
	Given multiple multidimensional points, centered,
	compute the minimal radius capable of containing quantile% of them.
	'''
	# Verify correct dimensionality, (n_samples, m)
	assert (quantile > 0 and quantile <= 1)
	assert (len(input_data.shape) == 2)
	n_samples = input_data.shape[0]
	m = input_data.shape[1]
	# Center the data
	tmp = input_data.clone().detach()
	avg = torch.mean(tmp, dim=0)
	x_data = tmp - avg
	# Store all the norms: the radius will be selected among them
	all_norms = torch.zeros(n_samples)
	for nth in range(n_samples):
		all_norms[nth] = torch.norm(x_data[nth])
	# Set initially the radius as the max possible value
	radius = torch.max(all_norms)
	backup_radius = radius.clone()
	to_contain = math.ceil(n_samples * quantile)
	# Decrease the radius as long as it contains STRICTLY too much points
	while (all_norms <= radius).sum() > to_contain:
		backup_radius = radius.clone()
		radius *= 0.99	
#		print(f"Radius: {radius : .1f}")
	radius = backup_radius.clone()
	# Check that the radius contains the desired quantile of samples
	est_quantile = (all_norms <= radius).sum() / n_samples
	print(f"(radius {radius:.2f} has {est_quantile*100:.1f}% of points)")
	assert (est_quantile >= quantile and est_quantile <= 1.)
	return radius
#---


def test_get_radius():
	x = torch.normal(10., 30., (1000, 4))
	print(get_radius(x))
	y = torch.rand((2,1))
	print(y)
	print(get_radius(y))
	z = torch.rand((100000,1))
	print(get_radius(z))
	return 1
#---


def norm_distmatrix (idata):
	'''
	Simply get the rescaled norm of the distance matrix of given data
	'''
	assert (len(idata.shape) == 2)
	n_points = idata.shape[0]
	# if a single point is given, the distance matrix is simply zero
	if n_points == 1:
		return 0.
	# otherwise we compute stuff as expected
	n_entries = int(n_points * (n_points + 1) / 2) - n_points
	upper_matrix = torch.zeros(n_entries)
	counter = 0
	for i in range(n_points):
		for j in range(i + 1, n_points):
			upper_matrix[counter] = torch.norm(idata[i] - idata[j])
#			upper_matrix[counter] /= math.sqrt(idata.shape[1])
			counter += 1
	assert (counter == n_entries)
	return torch.norm(upper_matrix) / math.sqrt(n_entries)
#---


def test_norm_distmatrix ():
	print(f"Testing norm of distance matrix")
	x = torch.normal(0., 1., (100, 1))
	print(norm_distmatrix(x))
	x = torch.zeros(10, 1)
	assert(norm_distmatrix(x) < 1e-4)
	x = torch.ones(1, 1)
	print(norm_distmatrix(x))	
	x = torch.ones(2, 1)
	print(norm_distmatrix(x))
	return 1
#---
	

def to_two_dimension (input_data):
	assert (len(input_data.shape) == 2)
	n_samples = input_data.shape[0]
	m = input_data.shape[1]
	# Center the data
	tmp = input_data.clone().detach()
	avg = torch.mean(tmp, dim=0)
	x_data = tmp - avg
	# Compute the distance matrix, before transform
	start_geometry = norm_distmatrix(x_data)
	# Transform data using Multidimensional Scaler
	embedding = MDS(n_components=2, normalized_stress = "auto")
	tmp_transformed = embedding.fit_transform(x_data.numpy())
	x_transformed = torch.from_numpy(tmp_transformed)
	# Compute the distance matrix, after the transform
	end_geometry = norm_distmatrix(x_transformed)
	# (comparing them says about the reliability of the 2d data)
	tmp_err = torch.abs(end_geometry - start_geometry) * 100.
	rel_err = tmp_err / torch.abs(start_geometry)
	print(f"(norms: {start_geometry:.1f} -> {end_geometry:.1f})")
	print(f"(error of {rel_err:.1f}%)")
	return x_transformed
#---


def test_to_two_dimension():
	print(f"Testing Multidimensional Scaling")
	x = torch.normal(0., 3., (20, 2))
	new = to_two_dimension(x)

	x = torch.normal(0., 3., (20, 5))
	new = to_two_dimension(x)
	return 1
#---


def check_quantiles ():
	LIM = 10
	for nth in range(LIM):
		x = torch.normal(4., 10., (100, 5))
		r1 = get_radius(x)
		y = to_two_dimension(x)
		r2 = get_radius(y)
		print(f"{r1:.1f} VS {r2:.1f} ?")
	return 1
#---

def add_zero_3d (idata):
	'''
	Receive a dataset composed by multidimensional paths,
	each of a fixed dimension:
	(n_paths, n_times, n_dimension)
	for instance (2, 10, 3) means:
	two paths, each three dimensional, with 10 time observations.
	The purpose of this function is ti add a zero at the beginning of
	each observation.
	'''
	assert (len(idata.shape) == 3)
	n_paths = idata.shape[0]
	n_times = idata.shape[1]
	n_dim = idata.shape[2]
	tmp = torch.zeros(n_paths, n_times + 1, n_dim)
	for kth in range(n_paths):
		for nth in range(n_dim):
			tmp[kth][1:, nth] = idata[kth][:, nth]
	return tmp	
#---


def augment_2d (idata):
	assert (len(idata.shape) == 2)
	dim = idata.shape[1]
	n_times = idata.shape[0]
	tmp = torch.zeros(n_times, dim + 1)
	tmp[:, 0] = torch.linspace(0., 1., n_times)
	for nth in range(1, dim + 1):
		tmp[:, nth] = idata[:, nth - 1]
	return tmp
#---


def augment_3d (idata):
	assert (len(idata.shape) == 3)
	n_paths = idata.shape[0]
	n_times = idata.shape[1]
	n_dim = idata.shape[2]
	tmp = torch.zeros(n_paths, n_times, n_dim + 1)
	for nth in range(n_paths):
		tmp[nth] = augment_2d(idata[nth])
	return tmp
#---


def path_len (og_path):
	assert (len(og_path.shape) <= 2)
	if (len(og_path.shape) == 1):
		path = og_path.reshape(-1, 1)
	elif (len(og_path.shape) == 2):
		path = og_path
	tot = 0.
	n_times = path.shape[0]
	for nth in range(1, n_times):
		tot += torch.norm(path[nth] - path[nth - 1])
	return tot	
#---		

def test_path_len():
	# check on one dimensionals
	x = torch.ones(10)
	assert(path_len(x) == 0)
	# check on multidimensional
	y = torch.ones(10, 5)
	assert(path_len(y) == 0)
	# check that augmenting increases the length of at most 1
	for nth in range(10):
		z = torch.normal(0., 10., (10, 1))
		len1 = path_len(z)
		z2 = augment_2d(z)
		len2 = path_len(z2)
		assert (len2 <= len1 + 1)
		print(f"({len2:.3f} <= {len1:.3f} + 1)")
	return 1
#---	
	

if __name__ == "__main__":
	test_get_radius()
	test_norm_distmatrix()
	test_to_two_dimension()
	test_path_len()
