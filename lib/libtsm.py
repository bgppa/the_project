'''
Time Series Manipulation library, suitable to manipulare time series
before applying on them the signature transform.
We have functions like log-return transforms, timestamp augmentation
and zero-start injection.

INFORMATION:
a "flexible_time_series" is a tensor that can be of maximum shape 3.

IF the input is of shape (a, b, c), is interpreted as:
a = number of time series
b = number of time observations for each time series
c = amount of numbers registered for each observation
Basicall is a batch of multidimensional time series.

IF a tensor of dimension 2 is given, written as (x, y), it is interpreted as:
x = number of time points
y = amount of numbers registered for each observation
In other words, it is a time series of length x in dimension y

IF a tensor of dimension 1 or 0 is given, it is interpreted
as a one-dimensional time series composed by the given values.

LIST OF USER FUNCTIONS:
[see above for the definition of "flexible_time_series"]
augment 		(flexible_time_series)
add_zero		(flexible_time_series)
get_1var		(flexible_time_series)
'''
import torch


##########################################################################
###	Easy checks like dimensionality, positivity
##########################################################################

def is_onedim(a_tensor):
	if (len(a_tensor.shape) > 1):
		if (a_tensor.shape[1] != 1):
			print(f"WRN: shape is {a_tensor.shape}")
			print(f"the expected dimension is 1")
			return 0
	return 1
#---

def test_is_onedim():
	r = torch.zeros(10,)
	assert (is_onedim(r) == 1)
	r = torch.zeros(10,1)
	assert (is_onedim(r) == 1)
	r = torch.zeros(10, 2)
	assert (is_onedim(r) == 0)
	return 1
#---

def is_spositive(ts):
	'''
	Return 1 if the given tensor only contains strictly positive values.
	'''
	path = ts.reshape(-1,)
	for elm in path:
		if elm <= 0.:
			return 0
	return 1
#-------

def test_is_spositive():
	one = torch.normal(0., 10, (10000,))
	assert (is_spositive(one) == 0)
	two = torch.randint(1, 1000, (300, 56))
	assert (is_spositive(two) == 1)
	return 1
#-------

##########################################################################
###	Computation of the 1-variation of a multidimensional curve
##########################################################################

def curve_length_1d (onedim_curve):
	'''
	Return the piecewise length of a one dimensional curve
	'''	
	if (is_onedim(onedim_curve)):
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
	a = torch.tensor([0., 1., 10.])
	assert(curve_length_1d(a) == 10.)
	a = torch.tensor([3., 1.])
	assert(curve_length_1d(a) == 2.)
	return 1
#---

def curve_length_Nd (curve):
	'''
	Cumpute the piecewise lenght of a multidimensional curve
	'''
	assert (len(curve.shape) == 2)
	n_times = curve.shape[0]
	curr = 0.
	for nth in range(1, n_times):
		curr += torch.norm(curve[nth] - curve[nth - 1])
	return curr
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


def get_1var (icurve):
	'''
	Compute the 1-variation of a n-dimensional curve
	'''
	assert (len(icurve.shape) <= 3)
	if (len(icurve.shape) == 1):
		return curve_length_1d(icurve)
	elif (len(icurve.shape) == 2):
		return curve_length_Nd(icurve)
	else:
		database = torch.zeros(icurve.shape[0], 1)
		for nth in range(icurve.shape[0]):
			database[nth] = curve_length_Nd(icurve[nth])
		return database
#---


def test_get_1var():
	a = torch.ones(100)
	assert (get_1var(a) == 0.)
	a = torch.zeros(10, 3)
	assert(get_1var(a) == 0.)
	a = torch.tensor([0., 1., 10.])
	assert(get_1var(a) == 10.)
	a = torch.tensor([3., 1.])
	assert(get_1var(a) == 2.)
	x = torch.ones(10, 2)
	assert (get_1var(x) == 0.)
	x = torch.ones(10, 1)
	assert (get_1var(x) == 0.)
	x = torch.normal(0., 5., (20, 1))
	res1 = get_1var(x)
	res2 = get_1var(x)
	assert (torch.abs(res1 - res2) < 1e-6)
	a = torch.normal(0., 0.01, (3, 4, 2))
	all_1var = get_1var(a)
	assert (torch.norm(all_1var) < 10)
	return 1
#---

##############################################################################
####	Zero-insertion for a time series: ADD TESTS
##############################################################################


def add_zero_1d (curve):
	'''
	Add a zero at the beginning of a one-dimensional time series
	'''
	if len(curve.shape) == 1:
		n_times = len(curve)
		result = torch.zeros(n_times + 1)
		for nth in range(n_times):
			result[nth + 1] = curve[nth]
		return result
	elif len(curve.shape) == 2 and curve.shape[1] == 1:
		n_times = curve.shape[0]
		result = torch.zeros(n_times + 1, 1)
		for nth in range(n_times):
			result[nth + 1][0] = curve[nth][0]
		return result
	else:
		print(f"add_zero_1d: shape not supported. Returning 0.")
		return 0.
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
	assert(add_zero_1d(a) == 0)
	return 1
#---


def add_zero_2d (multidim_curve):
	assert (len(multidim_curve.shape) == 2)
	n_times = multidim_curve.shape[0]
	dim_measurements = multidim_curve.shape[1]
	res = torch.zeros(n_times + 1, dim_measurements)
	for nth in range(n_times):
		res[nth + 1] = multidim_curve[nth]
	return res
#---
		

def test_add_zero_2d():
	pass


def add_zero_3d (dataset):
	assert (len(dataset.shape) == 3)
	n_samples = dataset.shape[0]
	n_times = dataset.shape[1]
	dim_measurements = dataset.shape[2]
	res = torch.zeros(n_samples, n_times + 1, dim_measurements)
	for k in range(n_samples):
			res[k] = add_zero_2d(dataset[k])
	return res
#---


def test_add_zero():
	pass


def add_zero (generic_curve):
	tensor_dim = len(generic_curve.shape)
	if tensor_dim == 1:
		return add_zero_1d(generic_curve)
	elif tensor_dim == 2:
		return add_zero_2d(generic_curve)
	else:
		return add_zero_3d(generic_curve)
#---


def test_add_zero():
	pass

##############################################################################
####	Augmentation: NEED TO ADD THE TESTS
##############################################################################

def augment_1d(onedim_ts):
	'''
	Given a one dimensional time series, add the time coordinate.
	'''
	if (is_onedim(onedim_ts) == 0):
		print(f"ERR: augment_1d: shape {onedim_ts.shape}")
		print(f"only accept 1-dim paths.")
		return 0.
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

def augment_2d (idata):
	'''
	Augment a multivariate time series
	'''
	assert (len(idata.shape) == 2)
	dim = idata.shape[1]
	n_times = idata.shape[0]
	tmp = torch.zeros(n_times, dim + 1)
	tmp[:, 0] = torch.linspace(0., 1., n_times)
	for nth in range(1, dim + 1):
		tmp[:, nth] = idata[:, nth - 1]
	return tmp
#---


def test_augment_2d():
	# TO DO
	pass

def augment_3d (idata):
	'''
	Augment a batch of multivariate time series
	'''
	assert (len(idata.shape) == 3)
	n_paths = idata.shape[0]
	n_times = idata.shape[1]
	n_dim = idata.shape[2]
	tmp = torch.zeros(n_paths, n_times, n_dim + 1)
	for nth in range(n_paths):
		tmp[nth] = augment_2d(idata[nth])
	return tmp
#---

def test_augment_3d():
	# TO DO
	pass

def augment (idata):
	'''
	Use the three routines above for a simpler user interface
	'''
	tensor_shape = len(idata.shape)
	assert (tensor_shape <= 3)
	if tensor_shape == 1:
		return augment_1d(idata)
	elif tensor_shape == 2:
		return augment_2d(idata)
	else:
		return augment_3d(idata)
#---

def test_augment():
	# TO DO
	pass


##############################################################################


if __name__ == "__main__":
	expected = 3
	success = 0
	success += test_curve_length_1d()
	success += test_curve_length_Nd()
	success += test_get_1var()
	success += test_is_onedim()
	success += test_add_zero_1d()

	print(f"Passed: {success} / {expected}.")	
