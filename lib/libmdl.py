'''
Just a library with simple utilities for training models.
TO REVISE BETTER
'''

import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt


def mix_and_split(full_x, full_y):
	'''
	Mix the dataset and return train and validation data
	'''
	n_samples = full_x.shape[0]
	half = int(n_samples / 2)
	random_indeces = torch.randperm(n_samples)
	train_indeces = random_indeces[:half]
	val_indeces = random_indeces[half:]
	x_train = full_x[train_indeces]
	y_train = full_y[train_indeces]
	x_val = full_x[val_indeces]
	y_val = full_y[val_indeces]
	return (x_train, y_train, x_val, y_val, train_indeces, val_indeces)
#---


def linear_classifier (x_train, y_train, x_val, y_val, tot_epochs = 10_000,
		lr = 1e-3):
	'''
	ADD INFORMATION HERE
	'''
	# simple linear regression
	assert (len(x_train.shape) == 2)
	dim_samples = x_train.shape[1]
	model = nn.Sequential(nn.Linear(dim_samples, 2))
	optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
	loss_fn = nn.CrossEntropyLoss()

	# Keep track of loss on training and validation data
	lt_hist = torch.zeros(tot_epochs)
	lv_hist = torch.zeros(tot_epochs)
	# Train the model
	for nth in range(tot_epochs):
		optimizer.zero_grad()
		train_pred = model(x_train)
		loss = loss_fn(train_pred, y_train)
		with torch.no_grad():
			val_pred = model(x_val)
			val_loss = loss_fn(val_pred, y_val)
			lv_hist[nth] = val_loss.item()
			lt_hist[nth] = loss.item()
			print(f"{nth+1}/{tot_epochs} t:{loss.item():.3e}")
			print(f"\t\tv:{val_loss.item():.3e}")
		loss.backward()
		optimizer.step()

	# Plot the loss history
	plt.plot(range(tot_epochs), lt_hist, label='train loss', color="teal")
	plt.plot(range(tot_epochs), lv_hist, label='val loss', color="orange")
	plt.grid()
	plt.legend()
	plt.title("Trained linear model: loss evolution")
	plt.show()
	return model
#---


def tuples_to_labels(raw_predictions):
	'''
	Given an array of tuples, convert each into the integer
	corresponding to the position with highest value.
	'''
	length = raw_predictions.shape[0]
	labels = torch.zeros(length)
	for nth in range(length):
		labels[nth] = torch.max(raw_predictions[nth], dim=0)[1]
	return labels
#---

def accuracy(y1_arg, y2_arg):
	'''
	Given y1 and y2 tensors of labels 0 and 1, return the percentage
	of labels with the same values
	'''
	# Convert into tensor float just in case
	y1 = y1_arg.type(torch.float)
	y2 = y2_arg.type(torch.float)
	sm = (y1 - y2) ** 2 
	# The number of 1's in sm equals the number of _wrong_ predictions
	sm = (sm - 1) ** 2
	# Now the number of 1 in sm equals the number of RIGHT predictions
	acc = torch.mean(sm)
	# Convert into percentage and return
	return (acc * 100.).item()
#---

##########################################################################
####	THE FOLLOWING SECTION MUST BE CHECKED AGAIN
##########################################################################

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
