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

