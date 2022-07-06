# script to generate matrix for sdm figure in paper
# for early version of paper.  Will probably not be used.

import numpy as np

def clean_str(arr):
	# From string representation of numpy array
	output = np.array_str(arr).replace("]\n [","\n")
	if arr.max() == 1 and arr.min() == 0:
		# assume binary, remove spaces between numbers
		output = output.replace(" ","")
	return output


def make_address(nrows=12, ncols=16, nact=2):
	np.random.seed(seed=78639)
	address_matrix = np.random.randint(0,high=2,size=(nrows, ncols), dtype=np.int8)
	counter_matrix = np.zeros((nrows, ncols), dtype=np.int8)
	address = np.random.randint(0,high=2,size=ncols, dtype=np.int8)
	data = np.random.randint(0,high=2,size=ncols, dtype=np.int8)
	xor = np.logical_xor(address_matrix, address)
	hamming_distances = np.count_nonzero(xor, axis=1)
	idx = np.argpartition(hamming_distances, nact)   # get index of nact smallest hamming distances first
	counter_matrix[idx[0:nact]] += data * 2 - 1
	padr = np.array_str(address_matrix).replace(" ","").replace("[","").replace("]","")
	print("Address matrix=\n%s" % clean_str(address_matrix))
	print("Address=%s" % clean_str(address))
	print("Data=%s" % clean_str(data))	
	print("Hamming_distances=%s" % clean_str(hamming_distances))
	print("counter_matrix=%s" % clean_str(counter_matrix))

make_address()


