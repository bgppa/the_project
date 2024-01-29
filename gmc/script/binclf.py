# -----------------------------------------------------------------------
# This is the experiment concerning BINARY CLASSIFICATION with curves
# of different sample size. 
# -----------------------------------------------------------------------
import torch
import signatory as sg
import matplotlib.pyplot as plt
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import MDS
from math import sqrt
from datagn import segment_dataset, sinusoid_dataset, impulse_dataset
from datagn import remove_some, simple_augment
from datagn import accuracy, distance_matrix, mtx_err


# Get from the user two parameters: the random seed, and the data type
data_list = ["SEG", "SIN", "IMP"]
STATUS = True
if (len(sys.argv) != 3):
	print("Error: you must specify SEED and DATATYPE")
	print(f"SEED: a positive integer.")
	print(f"DATATYPE in {data_list}")
	print(f"example: $python3 {sys.argv[0]} 0 SEG")
	STATUS = 0
else:
	if int(sys.argv[1]) < 0:
		print("Fatal: SEED must be positive (given: {sys.argv[1]}).")
		STATUS = 0
	if sys.argv[2] not in data_list:
		print(f"Fatal: {sys.argv[2]} is not in {data_list}")
		STATUS = 0
if STATUS == 0:
	quit()
# Otherwise the parameters are correct and we can proceed with the simulation

my_seed = int(sys.argv[1])
data_type = sys.argv[2]
torch.manual_seed(my_seed)
NO_GUI = True
sig_depth = 4


# Simply generate data
n_samples = 400
max_nodes = 100

if data_type == "SEG":
	x_raw, y_full, data_type = segment_dataset(n_samples, max_nodes)
elif data_type == "SIN":
	x_raw, y_full, data_type = sinusoid_dataset(n_samples, max_nodes)
else:
	x_raw, y_full, data_type = impulse_dataset(n_samples, max_nodes)
print(f"Working with {data_type} data, random seed {my_seed}.")
print(f"List of {n_samples} data")
# while y_full is a proper torch tensor

# Print some random paths
num_display = 3
chosen_to_show = torch.randint(n_samples, (num_display,))
for idx in chosen_to_show:
	print(f"Plotting path {idx+1}...")
	mycolor = "orange" #"red"
	if y_full[idx] == 1:
		mycolor = "teal" #"green"
	plt.plot(x_raw[idx], color = mycolor, label=f"path {idx}")
plt.legend()
plt.grid()
#plt.title(f"Displaying {num_display} randomly chosen paths...")
if NO_GUI:
	plt.savefig(f"plots/{data_type}-paths-seed{my_seed}.png")
	plt.clf()
else:
	plt.show()
#---

#quit()
# Now, proceed with data preprocessing
# (1) Transform my list into a list with augmented tensors
x_aug = []
for nth in range(n_samples):
	# augment the time series by adding the time coordinate
	x_auxiliary = simple_augment(x_raw[nth])
	# remove some random points from it, modeling lost in measurements
	x_aug.append(remove_some(x_auxiliary, 10))

#input("DEBUG - proceed?")
# Now my list of tensors can be transformed into signatures
sig_len = 2 ** (sig_depth + 1) - 2
x_full = torch.zeros(n_samples, sig_len)
for nth in range(n_samples):
	x_full[nth] = sg.signature(x_aug[nth].unsqueeze(0), depth=sig_depth)

# Plot the signatures
for idx in chosen_to_show:
	print(f"Plotting sig-path {idx+1}...")
	mycolor = "orange" #"red"
	if y_full[idx] == 1:
		mycolor = "teal" #"green"
	plt.plot(x_full[idx], color = mycolor, label=f"sig of path {idx}")
plt.grid()
plt.legend()
#plt.title(f"...and their {sig_depth}-signatures")
if NO_GUI:
	plt.savefig(f"plots/{data_type}-signatures-seed{my_seed}.png")
	plt.clf()
else:
	plt.show()
#---
#quit()

# Create train and validation data
half = int(n_samples / 2)
x_train = x_full[:half]
y_train = y_full[:half]
x_val = x_full[half:]
y_val = y_full[half:]
print(f"Train data: {y_train.mean()*100 : .1f}%, {len(x_train)} samples.")
print(f"Val data: {y_val.mean()*100 : .1f}%, {len(x_val)} samples.")

# (3) Classification using KNN
my_neigh = int(sqrt(half))
knn_model = KNeighborsClassifier(n_neighbors=my_neigh)
knn_model.fit(x_train, y_train)
train_predict = knn_model.predict(x_train)
val_predict = knn_model.predict(x_val)
full_predict = knn_model.predict(x_full)

train_accuracy = accuracy(y_train, train_predict)
val_accuracy = accuracy(y_val, val_predict)
full_accuracy = accuracy(y_full, full_predict)

print("--------------------------------------------")
print(f"KNN model on {my_neigh} neighbors")
print(f"Train ACC: {train_accuracy : .1f} %")
print(f"Val ACC: {val_accuracy: .1f} %")
print(f"Overall ACC: {full_accuracy: .1f} %")
print("--------------------------------------------")

print("Generating the visualization plot in 2d...")
print("--------------------------------------------")
embedding = MDS(n_components = 2)

og_distances = distance_matrix(x_full)
reduced = torch.tensor(embedding.fit_transform(x_full))
new_distances = distance_matrix(reduced)
embedding_error = mtx_err(og_distances, new_distances)
print(f"Embedding error of {embedding_error :.1f} %")

wrong_classified = 0
for nth in range(n_samples):
	my_color = "teal" #"green"
	my_marker = ","
	my_alpha = 0.2
	# if we have a correctly classified point
	if y_full[nth] == full_predict[nth]:
		if y_full[nth] == 0:
			my_color = "orange" #"red"
	else:
	# if we have a mis-classified point, print in black
		wrong_classified += 1
		my_color = "black"
		my_alpha = 1.
		my_marker = "*"
#	print(f"point {nth}, marker {my_marker}, alpha {my_alpha}")
	plt.scatter(reduced[nth][0], reduced[nth][1],
		color=my_color, marker=my_marker, alpha=my_alpha)
plt.grid()
print(f"({wrong_classified}/{n_samples} wrongly classified)")
fdlty = 100. - embedding_error
#plt.title(f"Isometric[{fdlty :.1f}%] 2dEmbedding [acc: {full_accuracy:.1f}%])")
if NO_GUI:
	plt.savefig(f"plots/{data_type}-isometry-seed{my_seed}.png")
	plt.clf()
else:
	plt.show()
