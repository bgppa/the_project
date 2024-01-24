'''
Collection of auxiliary functions for the SigWizard main script.
Separated and stored here so to improve code readability and testing routines.
'''
import torch
import matplotlib.pyplot as plt
import signatory as sg
import torch.optim as optim
import torch.nn as nn
import math

DEBUG_MODE = 0

def manage_error():
	'''
	Here I decide how my functions will behave when encountering error
	By construction, it must at least return a 0.
	Even better if I am 
	'''
	if DEBUG_MODE:
		return 0
	else:
		quit()
#---

def isonedim(a_tensor):
	if (len(a_tensor.shape) > 1):
		if (a_tensor.shape[1] != 1):
			print(f"WRN: shape is {a_tensor.shape}")
			print(f"the expected dimension is 1")
			return 0
	return 1
#---
def test_isonedim():
	r = torch.zeros(10,)
	assert (isonedim(r) == 1)
	r = torch.zeros(10,1)
	assert (isonedim(r) == 1)
	r = torch.zeros(10, 2)
	assert (isonedim(r) == 0)
	return 1
#---



###########################################################################
####	Utilities to preprocess a time series
###########################################################################

def strict_positive(ts):
	path = ts.reshape(-1,)
	for elm in path:
		if elm <= 0.:
			return 0
	return 1
#-------

def test_strict_positive():
	one = torch.normal(0., 10, (10000,))
	assert (strict_positive(one) == 0)
	two = torch.randint(1, 1000, (300, 56))
	assert (strict_positive(two) == 1)
	return 1
#-------

def logret(ts):
	'''
	Convert the 1-dimensional time series in input, to log returns.
	Return the converted and its original starting value,
	useful later to recover the orginal data.
	'''
	if (isonedim(ts) == 0):
		print("ERR: logret: tensor not 1-dimensional: {ts.shape}")
		return manage_error()
	loc_ts = ts.reshape(-1,)
	if (strict_positive(loc_ts) == 0):
		print("ERR: logret: negative values")
		return manage_error()
	# if we have a one dimensional, positive array
	stval = loc_ts[0]
	result = torch.log(loc_ts / stval)
	return result, stval
#-------

def test_logret():
	print("Testing logret...")
	one = torch.ones(19,)
	one_lg, one_st = logret(one)
	assert(torch.norm(one_lg) < 1e-6)
	assert(one_st == 1.)
	print("The following ERR is produced on purpose:")
	two = torch.normal(0., 10, (100,1))
	two_lg = logret(two)
	assert (two_lg == 0)
	return 1
#-------

def logback(logvalues, their_start):
	'''
	We are given an array of numbers each coming from a logreturn transform.
	But, each number can potentially come from a different time series,
	having its original starting value (store in their_start).
	If all the values are from the same transformed time series,
	giving their_start as an array of equal elements as the start values,
	actually inverts the log transform.
	'''
	if (isonedim(logvalues) == 0) or (isonedim(their_start) == 0):	
		print(f"ERR: logback: expected shape is (-1,), but: ")
		print(f"logvalues: {logvalues.shape}")
		print(f"their_start: {their_start.shape}")
		return manage_error()
	array1 = torch.exp(logvalues.reshape(-1,))
	array2 = their_start.reshape(-1,)
	n_samples = len(array1)
	if (n_samples != len(array2)):
		print("ERR: logback: arrays of different length")
		print(f"{n_samples} != {len(array2)}")
		return manage_error()
	result = torch.zeros(n_samples)
	for nth in range(n_samples):
		result[nth] = array1[nth] * array2[nth]
	return result
#-------

def test_logback():
	print("Testing logback...")
	one = torch.normal(100., 0.1, (20,))
	one_lg, one_st = logret(one)
	starting_values = torch.ones(20,) * one_st
	res = logback(one_lg, starting_values)
	assert (torch.norm(one - res) < 1e-4)
	return 1
#-------

def augment_1d(onedim_ts):
	'''
	Given a one dimensional time series, add the time coordinate.
	'''
	if (isonedim(onedim_ts) == 0):
		print(f"ERR: augment_1d: shape {onedim_ts.shape}")
		print(f"only accept 1-dim paths.")
		return manage_error()
	path = onedim_ts.reshape(-1,)
	n_times = path.shape[0]
	aug = torch.zeros(n_times, 2)
	aug[:, 0] = torch.linspace(0., 1., n_times)
	aug[:, 1] = path
	return aug
#-------

def test_augment_1d():
	one = torch.normal(3., 10, (30, 2))
	print("The following ERR is risen on purpose:")
	res = augment_1d(one)
	assert (res == 0)
	one = torch.normal(0., 10., (10, 1))
	two = one.reshape(10,)
	res1 = augment_1d(one)
	res2 = augment_1d(two)
	assert (torch.norm(res1 - res2) < 1e-4)
	assert res1.shape[0] == 10
	assert res1.shape[1] == 2
	assert res2.shape[0] == 10
	assert res2.shape[1] == 2
	return 1
#---

def isdecreasing(sequence):
	'''
	Returns 1 if the sequence is strictly decreasing
	'''
	if isonedim(sequence):
		sequence = sequence.reshape(-1)
		for nth in range(1, len(sequence)):
			if sequence[nth - 1] <= sequence[nth]:
				return 0
		return 1
	else:
		print(f"isdecreasing: not a onedime sequence!")
		return manage_error()
#-------

def test_isdecreasing():
	a = torch.ones(10)
	assert(isdecreasing(a) == 0)
	b = torch.tensor([0., 1., 2., 10])
	assert(isdecreasing(b) == 0)
	c = torch.tensor([1., -1., 10, 2])
	assert(isdecreasing(c) == 0)
	d = torch.tensor([10., 5., 1])
	assert(isdecreasing(d) == 1)
	return 1
#-------
	
def curve_length_1d (onedim_curve):
	if (isonedim(onedim_curve)):
		res = 0.
		curve = onedim_curve.reshape(-1)
		for nth in range(1, len(curve)):
			res += torch.abs(curve[nth] - curve[nth-1])
		return res
	else:
		print("Err: curve_length on a curve NOT one dimensional")
		return -1
#---

def test_curve_length_1d():
	a = torch.ones(100)
	assert (curve_length_1d(a) == 0.)
	a = torch.zeros(10, 3)
	assert(curve_length_1d(a) == -1)
	a = torch.tensor([0., 1., 10.])
	assert(curve_length_1d(a) == 10.)
	a = torch.tensor([3., 1.])
	assert(curve_length_1d(a) == 2.)
	return 1
#---

def curve_length_Nd (multidim_curve):
	'''
	Cumpute an upper bound for the lenght of a multidimensional curve		'''
	assert (len(multidim_curve.shape) == 2)
	n_curves = multidim_curve.shape[1]
	upper_bound = 0.
	for nth in range(n_curves):
		upper_bound += curve_length_1d(multidim_curve[:,nth])
	return upper_bound
#---

def test_curve_length_Nd():
	x = torch.ones(10, 2)
	assert (curve_length_Nd(x) == 0.)
	x = torch.ones(10, 1)
	assert (curve_length_Nd(x) == 0.)
	x = torch.normal(0., 5., (20, 1))
	res1 = curve_length_1d(x)
	res2 = curve_length_Nd(x)
	assert (torch.abs(res1 - res2) < 1e-6)
	return 1
#---
	
def add_zero_1d (onedim_curve):
	if (isonedim(onedim_curve)):
		curve = onedim_curve.reshape(-1)
		tmp = torch.zeros(len(curve)+1)
		tmp[1:] = curve
		return tmp
	else:
		print(f"add_zero: curve is not one dimensional")
		return manage_error()
#---

def test_add_zero_1d():
	a = torch.normal(10., 1., (100,))
	a = add_zero_1d(a)
	assert(len(a) == 101)
	assert(a[0] == 0.)
	a = torch.ones(1)
	a = add_zero_1d(a)
	assert(len(a) == 2)
	assert(a[0] == 0.)
	a = torch.normal(0., 1., (100, 2))
	a = add_zero_1d(a) # must print an error
	return 1
#---


############################################################################
####	Utilities for training a model
############################################################################


def linear_training(dim, e_units, x_train, y_train, x_val, y_val, lr):
	'''
	This is a classic training routine for Pytorch, in our case we are fine
	with a simple linear model, mseloss and a learning rate as parameter.
	'''
	print("Starting training...")	
	model = nn.Linear(dim, 1, bias = False)
	optimizer = optim.Adam(model.parameters(), lr=lr)
	loss_fn = nn.MSELoss()
	# I will check that the losses are following a decreasing behavior
	monitor_size = 5
	last_losses = torch.zeros(monitor_size)
	# Monitoring the loss evolution during training
	n_epochs = 10_000 * e_units
	hist_loss_train = torch.ones(n_epochs)
	hist_loss_val = torch.ones(n_epochs)
	# Classic training loop
	for nth in range(n_epochs):
		optimizer.zero_grad()
		y_train_pred = model(x_train)
		loss_train = loss_fn(y_train_pred, y_train)
		with torch.no_grad():
			y_val_pred = model(x_val)
			loss_val = loss_fn(y_val_pred, y_val)
			if (nth % 1000 == 0):
				print(f"{nth+1}/{n_epochs}")
				print(f"t: {loss_train.item():.3e} ", end=' ')
				print(f"v:{loss_val.item():.3e}")
			hist_loss_train[nth] = loss_train.item()
			hist_loss_val[nth] = loss_val.item()
			if nth < monitor_size:
				last_losses[nth] = loss_val.item()
			else:
				tmp = torch.zeros(monitor_size)
				tmp[:-1] = last_losses[1:]
				tmp[-1] = loss_val.item()
				last_losses = tmp
			if (nth > 0) and (nth % 1000 == 0):
		#		print("CHECKING")
		#		print(last_losses)
				if not isdecreasing(last_losses):
					print("Last monitored losses: ")
					print(last_losses)
					ch = input("Stop straining?")
					if len(ch) > 0:
						ch = ch.upper()[0]
						print(f"{ch.upper()}")
						if ch == 'Y':
							break
			loss_train.backward()
			optimizer.step()
			
	# endfor
	print(f"Training ended!")
	plt.plot(hist_loss_train[1000:nth], label='train')
	plt.plot(hist_loss_val[1000:nth], label='val')
	plt.grid()
	plt.legend()
	plt.title("Loss function evolution")
	plt.show()

	coefficients = list(model.parameters())[0][0].detach()
	plt.plot(range(1, dim+1), coefficients)
	plt.axhline(y=0, color="black", linestyle="dashdot")
	plt.grid()
	plt.title(f"The {dim} parameters of the linear model")
	plt.show()


	return model
#---

def test_linear_training():
	'''
	This test has to be written
	'''
	return 0
#---


def diagonal_test (x_train, y_train, x_val, y_val, model):
	'''
	Plot true values VS predicted one in a diagonal shape easy to read.
	'''
	train_pred = model(x_train).detach()
	val_pred = model(x_val).detach()

	y_min1 = torch.min(train_pred)
	y_min2 = torch.min(val_pred)
	y_min = torch.min(y_min1, y_min2)

	y_max1 = torch.max(train_pred)
	y_max2 = torch.max(val_pred)
	y_max = torch.max(y_max1, y_max2)

	plt.scatter(y_train, train_pred[:, 0], color="blue", label="train")
	plt.scatter(y_val, val_pred[:, 0], color="red", label="val")
	plt.plot(torch.linspace(y_min, y_max, 10),
			torch.linspace(y_min, y_max, 10),
			label="diagonal target",
			linestyle="dashed", color="green")
	plt.title("Diagonal test")
	plt.legend()
	plt.grid()
	plt.show()
#---



#############################################################################
####	Utilities for the Signature Transform
#############################################################################


def my_signature(database, signature_depth):
	if (len(database.shape) == 2):
		my_database = database.unsqueeze(0)
	elif (len(database.shape == 3):
		my_database = database
	else:
		print(f"my_signature: invalid size for the database")
		msg()
	return sg.signature(my_database, depth = signature_depth)
#---

def get_sig_levels (curve_dim, depth):
	indices = []
	level_start = 0
	for nth in range(depth):
		offset = curve_dim ** (nth + 1)
		print(f"{nth+1}: [{level_start}, {level_start + offset})")
		indices.append([level_start, level_start + offset])
		level_start += offset
	return indices
#---


def sig_err (depth, curve_len):
	'''
	Estimate the Signature norm error for a curve of certain
	1-variation and truncation depth.
	'''
	sm = 0.
	for nth in range(depth + 1):
		# nth from 0 to depth, included
		sm += (curve_len ** nth) / factorial(nth)
	return torch.exp(torch.tensor(curve_len)) - sm
#---


def sig_norm (input_signature, curve_dim, depth):
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
		plt.axvline(x=x_coordinate, color="orange",linestyle="dashdot",
			label=f"[lv{curr_lvl} {curr_norm:.1e}]")
	plt.legend()
	plt.plot(the_sig, color="teal")
	plt.grid()
	plt.title(f"Signature until level {depth}")
	plt.show()
	return 1
#---

def check_signature_decay(input_curve, curve_dim, target_depth):
	'''
	Check the curve has a signature decaying after the predicted level
	'''
	# From that level on, the norms of the levels must decay
	upper_level = math.ceil(curve_length_Nd(input_curve))
	checks = 5
	# Compute the signature until 'checks' level after the minimum
	depth = checks + upper_level
	the_sig = my_signature(input_curve, depth).reshape(-1)
	levels = get_sig_levels(curve_dim, depth)
	curr_val = the_sig[levels[upper_level-1][0]:levels[upper_level-1][1]]
	curr_norm = torch.norm(curr_val)
	for nth in range(upper_level, checks + upper_level):
		new_vals = the_sig[levels[nth][0]:levels[nth][1]]
		new_norm = torch.norm(new_vals)
		if (new_norm > curr_norm):
			print(f"is NOT decaying")
			return 0
		else:
			curr_norm = new_norm.clone()
	return 1
#-------
	

#############################################################################
#############################################################################
#############################################################################

if __name__ == "__main__":
	DEBUG_MODE = 1
	print(f"IMPORTANT: DEBUG mode is {DEBUG_MODE}")
	print(f"{'*' * 50}")
	if (DEBUG_MODE):
		print("Error will return 0 instead of exiting program.")
	else:
		print(f"Errors will exit the program insted of returning 0.")
	print(f"{'*' * 50}")
	expected = 8
	success = 0
#	success += test_isonedim()
#	success += test_logback()
	success += test_augment_1d()
	success += test_strict_positive()
#	success += test_logret()	
	success += test_curve_length_1d()
	success += test_add_zero_1d()
#	success += test_isdecreasing()
	success += test_curve_length_Nd()
	print(f"{success}/{expected} test passed (errors are on purpose).")
