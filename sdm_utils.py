
import numpy as np
import sys
import random
from random import randint
from random import randrange
# import pprint
# pp = pprint.PrettyPrinter(indent=4)
import gmpy2
from gmpy2 import xmpz


byteorder = "big" # sys.byteorder



def initialize_binary_matrix(nrows, ncols, debug=False):
	# create binary matrix with each row having binary random number stored as an integer (long)
	bm = np.random.randint(2, size=(nrows, ncols), dtype=np.int8)
	# byteorder = "big" # sys.byteorder
	# convert numpy binary array to python integers
	bmp = np.packbits(bm, axis=-1)
	bm = []
	for b in bmp:
		intval = int.from_bytes(b.tobytes(), byteorder)
		if debug:
			print("bm intval=%s" % bin(intval))
		bm.append(intval)
	return bm

def pseudo_random_choice(istart, iend, n, seed):
	# returns list of n values that are in the range(istart, iend).  seed is an integer
	# random number seed which should be > iend.
	selected_vals = []
	available_vals = list(range(istart, iend))
	while len(selected_vals) < n:
		index = seed % len(available_vals)
		selected_vals.append(available_vals.pop(index))
	return selected_vals

def select_top_matches_with_possible_ties(matches, nret, b, debug=False):
	# check for multiple items having same hamming distance as that at nret
	# if so, select randomly between them
	# find number of items that have same hamming distance as last one to make the cut
	j = 0
	last_hamming = matches[nret -1][1]
	while nret+j < len(matches) and matches[nret + j][1] == last_hamming:
		j += 1
	if j > 0:
		# search for previous matches with same hamming distance
		k = -2
		while nret + k >=0 and matches[nret + k][1] == last_hamming:
			k = k -1
		istart = k+nret+1
		iend = j+nret
		num_needed = nret - istart
		# print("num_needed=%s of %s" % (num_needed, iend - istart))
		selected_vals = pseudo_random_choice(istart, iend, num_needed, b)
		selected_vals.sort()
		if debug:
			print("istart = %s, iend=%s, num_needed=%s, selected_values=%s" % (istart, iend,
				num_needed, selected_vals))
		top_matches = matches[0:istart]
		for i in selected_vals:
			top_matches.append(matches[i])
	else:
		if debug:
			print("No block at end")
		top_matches = matches[0:nret]
	return top_matches

def test_select_top_matches_with_possible_ties():
	matches = [(0, 3), (1, 3), (2, 8), (3, 8), (4, 8), (5, 8), (6, 9), (7, 9), (8, 10) ]
	print("matches = %s" % matches)
	seed = randrange(1000000000)
	for nret in range(1, len(matches)+1):
		print("nret=%s"% nret)
		top_matches = select_top_matches_with_possible_ties(matches, nret, seed, debug=True)
		print(top_matches)

def sdm_random_addresses(sdm_num_rows, b, nret, debug):
	# return list of random address associated with b, returning the same list for each b
	if not hasattr(sdm_random_addresses, "row_cache"):
		sdm_random_addresses.row_cache = {}  # it doesn't exist yet, so initialize it
	if b not in sdm_random_addresses.row_cache:
		sdm_random_addresses.row_cache[b] = np.random.choice(sdm_num_rows, size=nret, replace=False)
	return sdm_random_addresses.row_cache[b]

def find_matches(m, b, nret, index_only=False, debug=False, include_stats=False, distribute_ties=False,
		sdm_address_method=0):
	# m is 2-d array of binary values, first dimension is value index, second is binary number
	# b is binary number to match
	# nret is number of top matches to return
	# returns sorted tuple (i, c) where i is index and c is number of non-matching bits (0 is perfect match)
	# if index_only is True, only return the indices, not the c
	# if distribute_ties is True, check for multiple items with same hamming distance, and select between them
	# in a pseudorandom way.  This is used when matching to hard location addresses.
	# If sdm_addres_method is 1, then use randomlly selected coordinates rather than hamming distance match to address
	# This is used to debug possible problem with hamming distance method
	if sdm_address_method == 1:
		sdm_num_rows = len(m)
		assert index_only
		assert not include_stats
		return sdm_random_addresses(sdm_num_rows, b, nret, debug)
	# distribute_ties=False
	matches = []
	for i in range(len(m)):
		try:
			ndiff = gmpy2.popcount(m[i]^b)
		except Exception as e:
			print("failed, len(m)=%s, type(b)=%s" % (len(m), type(b)))
			import pdb; pdb.set_trace()		
		matches.append( (i, ndiff) )
	if debug:
		print("matches =\n%s" % matches)
	# find top nret matches
	matches.sort(key = lambda y: (y[1], y[0]))
	if distribute_ties:
		top_matches = select_top_matches_with_possible_ties(matches, nret, b)
	else:
		top_matches = matches[0:nret]
	if index_only:
		top_matches = [x[0] for x in top_matches]
	if include_stats:
		# # find number of items that have same hamming distance, print that to see if this is a problem
		# j = 0
		# while matches[nret + j][1] == matches[nret -1][1]:
		# 	j += 1
		# if j > 0:
		# 	# search for previous matches with same hamming distance
		# 	k = -1
		# 	while nert + k >=0 and matches[nret + k][1] == matches[nret -1][1]:
		# 		k = k -1
		# 	print("Found %s matches to hamming distance %s" % (j, matches[nret -1]))
		dhds = [x[1] for x in matches[1:]]
		mean_dhd = statistics.mean(dhds)	# distractor hamming distance
		stdev_dhd = statistics.stdev(dhds)
		return (top_matches, mean_dhd, stdev_dhd)
	return top_matches

def rotate_left(n, width, d = 1):
	# n is a binary number (large integer)
	# width is the word_length (number of bits) in the integer
	# d is number of bits to shift
	# adapted from:
	# https://stackoverflow.com/questions/63759207/circular-shift-of-a-bit-in-python-equivalent-of-fortrans-ishftc
    return ((n << d) % (1 << width)) | (n >> (width - d))

def rotate_right(n, width, d = 1):
	# n is a binary number (large integer)
	# width is the word_length (number of bits) in the integer
	# d is number of bits to shift
    # adapted from:
    # https://stackoverflow.com/questions/27176317/bitwise-rotate-right
    return (2**width-1)&(n>>d|n<<(width-d))

def add_noise(iar, noise_percent):
	# add noise by flipping sign of elements of iar (integer array)
	# iar must be an numpy int array.  noise_percent is the percent noise, 0 to 100
	if noise_percent == 0:
		return
	word_length = len(iar)
	num_to_flip = int(word_length * noise_percent / 100)
	rng = np.random.default_rng()
	idx = rng.choice(word_length, size=num_to_flip, replace=False, shuffle=False)
	iar[idx] *= -1

def add_noise_works(iar, noise_percent):
	# add noise by flipping sign of elements of iar (integer array)
	# iar must be an numpy int array.  noise_percent is the percent noise, 0 to 100
	if noise_percent == 0:
		return
	word_length = len(iar)
	num_to_flip = int(word_length * noise_percent / 100)
	# rng = np.random.default_rng()
	# idx = rng.choice(word_length, size=num_to_flip)
	# iar[idx] *= -1
	indicies = np.random.choice(word_length, size=num_to_flip, replace=False)
	iar[indicies] *= -1

def int2iar(n, width):
	# convert n (integer) to int array with integer bits 0, 1 mapped to -1, +1 in the array
	# width is the word_length (number of bits) in the integer
	# returns numpy int8 array with: where n bit is zero, -1, where n bit is one, +1
	bool_vals = list(xmpz(n).iter_bits(stop=width))
	s = np.where(bool_vals, 1, -1)
	s = np.flipud(s)   # flip so most significant bit is first
	return s


def iar2int(iarray):
	# convert numpy int array to binary by thresholding > 0 then to int
	bbytes = np.where(iarray>0, ord('1'), ord('0')).astype(np.int8).tobytes()  # makes byte string, e.g. '0110101'
	intval = int(bbytes, 2)
	return intval

def pop_random_element(L):
	# pop random element from a list
	# from https://stackoverflow.com/questions/10048069/what-is-the-most-pythonic-way-to-pop-a-random-element-from-a-list
	i = random.randrange(len(L)) # get random index
	L[i], L[-1] = L[-1], L[i]    # swap with the last element
	x = L.pop()                  # pop last element O(1)
	return x