'''
This library included all the utilities for working with the signature
	and visualizing the corresponding results.
'''
import torch
import signatory as sg
import matplotlib.pyplot as plt


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


############################################################################
####	Signature: Visualization
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


############################################################################
####	Just the main part now, to run all possible tests
############################################################################
if __name__ == "__main__":
	success = 0.
	success += test_my_signature()
	success += test_get_sig_levels()
	success += test_factorial()
	success += test_sig_err()
	success += test_find_depth()
	success += test_sig_norm()
	success += test_plot_signature()

	print(f"{success} test passed out of .")
