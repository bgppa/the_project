'''
This library included all the utilities for working with the signature
	and visualizing the corresponding results.
'''
import torch
import signatory as sg
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.manifold import MDS


###########################################################################
####	Signature: computation, error estimation, norms
###########################################################################

def my_signature(database, depth):
	'''
	Wrap the signatory signature function so to add a possible
	customization.
	'''
	if (len(database.shape) == 2):
		my_database = database.unsqueeze(0)
	elif (len(database.shape) == 3):
		my_database = database
	else:
		input(f"ERR: my_signature: invalid size for the database")
		return 0.
	return sg.signature(my_database, depth)
#---

def test_my_signature():
	z = torch.normal(0., 1., (10, 3, 4))
	s1 = sg.signature(z, 4)
	s2 = my_signature(z, 4)
	assert torch.norm(s1 - s2) < 1e-3
	z = torch.normal(0., 1., (10, 3))
	s1 = sg.signature(z.unsqueeze(0), 3)
	s2 = my_signature(z, 3)
	assert (torch.norm(s1 - s2) < 1e-3)
	return 1
#---

def get_sig_levels (curve_dim, depth):
	'''
	Given the dimension of the curve and the truncation depth,
	returns the indices separating each level of the signature.
	Useful for computing the tensor norm or plotting the signature
	in a more precise way.
	'''
	if (curve_dim < 1 or depth < 1):
		input(f"ERR: get_sig_levels: invalid parameters")
		return 0
	indices = []
	level_start = 0
	for nth in range(depth):
		offset = curve_dim ** (nth + 1)
		print(f"({nth+1}: [{level_start}, {level_start + offset}) )")
		indices.append([level_start, level_start + offset])
		level_start += offset
	return indices
#---

def test_get_sig_levels ():
	# Curve of dimension 1, depth 2
	l1 = get_sig_levels(1, 2)	
	assert (len(l1) == 2)
	# first level starts in 0 and ends before 1...
	assert l1[0] == [0, 1]
	# second level starts in 1 end ends before 2
	assert l1[1] == [1, 2]
	
	# Curve of dimension 2, depth 3
	dim2 = get_sig_levels(2, 3)
	assert (len(dim2) == 3)
	assert dim2[0] == [0, 2]
	assert dim2[1] == [2, 6]
	assert dim2[2] == [6, 14]

	# Curve of dimension 4, depth 1
	dim4 = get_sig_levels(4, 1)
	assert (len(dim4) == 1)
	assert dim4[0] == [0, 4]
	return 1
#---

def factorial (n):
	if n <= 1:
		return 1
	else:
		return n * factorial(n - 1)
#---

def test_factorial():
	assert (factorial(0) == 1)
	assert (factorial(5) == 120)
	assert (factorial(-3) == 1)
	return 1
#---

def sig_err (depth, curve_len):
	'''
	Estimate the Signature TRUNCATION ERROR when working with a curve
	of certain 1-variation (curve_len).
	Assume to truncate until depth, included.
	'''
	if (depth < 0 or curve_len < 0):
		input(f"ERR: sig_err: invalid parameters. Returning 0.")
		return 0
	sm = 0.
	for nth in range(depth + 1):
		# nth from 0 to depth, included
		sm += (curve_len ** nth) / factorial(nth)
	return torch.exp(torch.tensor(curve_len)) - sm
#---

def test_sig_err():
	'''
	Testing on values analytical predictable.
	'''
	e1 = sig_err(0, 1)
	assert torch.abs(e1 - torch.e + 1.) < 1e-6
	e2 = sig_err(1, 2)
	assert (e2 <= (torch.e ** 2. - 2.))
	return 1
#---

def sig_norm (input_signature, curve_dim, depth):
	'''
	Compute the TENSOR NORM of a signature, obtained by summing the
	euclidean norms of each level composing it.
	'''
	levels = get_sig_levels(curve_dim, depth)
	the_sig = input_signature.reshape(-1)
	tot_norm = 0.
	for nth in range(depth):
		curr_lvl = nth + 1
		vals = the_sig[levels[nth][0]:levels[nth][1]]
		tot_norm += torch.norm(vals)
		print(f"({depth}-signature of norm {tot_norm:.3f})")
	return tot_norm
#---

def test_sig_norm ():
	# it must be the same on signatures containing only one level
	for n in range(10):
		tmp = torch.normal(0., 1., (10, 2))
		s = my_signature(tmp, 1)
		euclidean = torch.norm(s)
		tensor_norm = sig_norm(s, 2, 1)
		assert torch.norm(euclidean - tensor_norm) < 1e-6

	# otherwise its square is always bigger than the square of the eucl.
	for n in range(10):
		tmp = torch.normal(0., 1., (10, 3))
		s = my_signature(tmp, 4)
		euclidean2 = torch.norm(s) ** 2.
		tensored2 = sig_norm(s, 3, 4) ** 2.
		assert (tensored2 >= euclidean2)

	return 1
#---


def find_depth (curve_len, desired_err):
	'''
	Check how much should I truncate to reach some desired error.
	'''
	# initial_err corresponds to truncating at level 0, a possibility
	# not supported by signatory since by contruction the level 0
	# is _always_ simply the constant 1
	initial_err = torch.exp(curve_len) - 1.
	curr_err = initial_err
	count = 0
	max_iter = 20
	print(f"(Starting error: {initial_err :.3f})")
	while (curr_err > desired_err) and (count < max_iter):
		count += 1
		curr_err = sig_err (count, curve_len)
		print(f"(Depth {count} produces error {curr_err :.3f})")
	if count >= max_iter:
		print(f"Desired error too small!")
	else:
		print(f"(Desired error of {desired_err:.3f} at depth {count})")
	return count
#---

def test_find_depth ():
	random_values = torch.randint(8, (10, 2)) + 1
	for (curve_len, og_depth) in random_values:
		desired_err = sig_err(og_depth, curve_len)
		suggested_depth = find_depth(curve_len, desired_err)
		assert (og_depth == suggested_depth)
	return 1
#---

def check_decay (isig, depth, curve_length):
	'''
	Check that all the levels of the given signature satisfy
	the bounds predicted by the theory.
	'''
	print(f"TO IMPLEMENT")
	pass
#---

def test_check_decay():
	pass

############################################################################
####	Signature: visualization and dimension reduction
############################################################################

def plot_signature (input_signature, curve_dim, depth):
	'''
	We produce a more elaborated plot of the signature,
	including the decay of level norms and every sig level.
	'''
	levels = get_sig_levels (curve_dim, depth)
	the_sig = input_signature.reshape(-1)
	for nth in range(depth):
		curr_lvl = nth + 1
		vals = the_sig[levels[nth][0]:levels[nth][1]]
		curr_norm = torch.norm(vals)
		x_coordinate = levels[nth][1]-1
		plt.axvline(x=x_coordinate, color="teal",linestyle="dashdot",
		label=f"[lv{curr_lvl} {curr_norm:.1e}]")
	plt.legend()
	plt.plot(the_sig, color="orange")
	plt.grid()
	plt.title(f"Signature until level {depth}")
	plt.show()
	return 1
#---

def test_plot_signature():
	tmp = torch.normal(0., 1., (10, 3))
	s = my_signature(tmp, 6)
	plot_signature(s, 3, 6)
	return 1
#---

def sigdist_fullmtx(idata, curve_dim, depth):
	'''
	Get the FULL distance matrix for a set of signatures,
	given in input as idata (input data).
	Norms are TENSOR NORMS.
	'''
	assert (len(idata.shape) == 2)
	n_points = idata.shape[0]
	# if a single point is given, the distance matrix is simply zero
	if n_points == 1:
		return 0.
	matrix = torch.zeros(n_points, n_points)
	# otherwise we compute stuff as expected
	counter = 0
	for i in range(n_points):
		for j in range(n_points):
			pt = idata[i] - idata[j]
			matrix[i][j] = sig_norm(pt, curve_dim, depth)
	return matrix
#---

def sigdist_udmtx(idata, curve_dim, depth):
	'''
	Get just the UPPER DIAGONAL of the distance matrix for a set 
	of signatures, given in input as idata (input data).
	Norms are TENSOR NORMS.
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
			pt = idata[i] - idata[j]
			upper_matrix[counter] = sig_norm(pt, curve_dim, depth)
			counter += 1
	assert (counter == n_entries)
	return upper_matrix
#---

def test_sigdist_matrix():
	curve_dim = 5
	depth = 6
	for k in range(10):
		tmp = torch.normal(0., 1., (20, 10, curve_dim))
		s = my_signature(tmp, depth)
		d1 = sigdist_fullmtx(s, curve_dim, depth)
		# get the upper diagonal
		norm_d1 = torch.norm(torch.triu(d1, diagonal=1))
		d2 = sigdist_udmtx (s, curve_dim, depth)
		norm_d2 = torch.norm(d2)
		assert (torch.abs(norm_d1 - norm_d2) < 1e-1)
		print(f"-----SIGDISTANCE MATRIX----------")
		print(d1)
		print(d2)
		print(f"--- they must be the same ---")
	return 1
#---	

def eucldist_fullmtx(idata):
	'''
	Get the FULL distance matrix for a set of data,
	measured in the ordinary Euclidean norm.
	'''
	assert (len(idata.shape) == 2)
	n_points = idata.shape[0]
	# if a single point is given, the distance matrix is simply zero
	if n_points == 1:
		return 0.
	matrix = torch.zeros(n_points, n_points)
	# otherwise we compute stuff as expected
	counter = 0
	for i in range(n_points):
		for j in range(n_points):
			pt = idata[i] - idata[j]
			matrix[i][j] = torch.norm(pt)
	return matrix
#---

def test_eucldist_fullmtx():
	# Generate datasets of equal points
	x = torch.ones((100, 4))
	# The distance matrix must be just zero
	assert (torch.norm(eucldist_fullmtx(x)) < 1e-6)
	return 1
#---

def sigs_to_dimtwo (input_sigs, curve_dim, depth):
	'''
	Given some SIGNATURES in input, convert them
	isometrically in dimension 2.
	'''
	assert (len(input_sigs.shape) == 2)
	n_samples = input_sigs.shape[0]
	m = input_sigs.shape[1]
	# Center the data
	tmp = input_sigs.clone().detach()
	avg = torch.mean(tmp, dim=0)
	x_data = tmp - avg
	# Compute the distance matrix, before transform
	start_geometry = sigdist_fullmtx(x_data, curve_dim, depth)
	# Transform data using Multidimensional Scaler
	embedding = MDS(n_components=2, normalized_stress="auto",
						dissimilarity="precomputed")
	tmp_transformed = embedding.fit_transform(start_geometry.numpy())
	x_transformed = torch.from_numpy(tmp_transformed)
	# Compute the distance matrix, after the transform
	end_geometry = eucldist_fullmtx(x_transformed)
	# (comparing them says about the reliability of the 2d data)
	start_norm = torch.norm(start_geometry)
	end_norm = torch.norm(end_geometry)
	tmp_err = torch.abs(end_norm - start_norm) * 100.
	rel_err = tmp_err / torch.abs(start_norm)
	print(f"(norms: {start_norm:.1f} -> {end_norm:.1f})")
	print(f"(error of {rel_err:.1f}%)")
	return x_transformed, rel_err
#---

def test_sigs_to_dimtwo():
	n_samples = 3
	curve_dim = 5
	depth = 6
	tmp = torch.normal(0., 1., (n_samples, 10, curve_dim))
	signatures = my_signature(tmp, depth)
	reduced, err = sigs_to_dimtwo(signatures, curve_dim, depth)
	# Since we have three points, they should be perfect in dimension 2
	assert (err < 1.)
	# Will be deleted
	for nth in range(n_samples):
		plt.scatter(reduced[nth][0], reduced[nth][1])
	plt.grid()
	plt.title("Test for the reduction in dimension 2")
	plt.show()
	return 1
#---	

def get_qradius (input_data, quantile = 0.95):
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
	# Decrease the radius as long as it contains STRICTLY too many points
	while (all_norms <= radius).sum() > to_contain:
		backup_radius = radius.clone()
		radius *= 0.99
#		print(f"Radius: {radius : .1f}")
	radius = backup_radius.clone()
	# Check that the radius contains the desired quantile of samples
	est_quantile = (all_norms <= radius).sum() / n_samples
	print(f"(q-radius {radius:.2f} has {est_quantile*100:.1f}% of points)")
	assert (est_quantile >= quantile and est_quantile <= 1.)
	return radius
#---

def test_get_qradius():
	x = torch.normal(10., 30., (1000, 4))
	print(get_qradius(x))
	y = torch.rand((2,1))
	print(y)
	print(get_qradius(y))
	z = torch.rand((100000,1))
	print(get_qradius(z))
	return 1
#---




############################################################################
####	Just the main part now, to run all possible tests
############################################################################
if __name__ == "__main__":
	expected = 11
	success = 0
	success += test_my_signature()
	success += test_get_sig_levels()
	success += test_factorial()
	success += test_sig_err()
	success += test_find_depth()
	success += test_sig_norm()
	success += test_plot_signature()
	success += test_sigdist_matrix()
	success += test_eucldist_fullmtx()
	success += test_sigs_to_dimtwo()
	success += test_get_qradius()

	print(f"PASSED: {success}/{expected}")
