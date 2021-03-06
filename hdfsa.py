# script to store finite-state automaton using high-dimensional vectors with:
# 1) bundling and 2) sparse distributed memory

import numpy as np
import argparse
import sys
import random
from random import randint
from random import randrange
# import time
# import hashlib
# import json
import pprint
pp = pprint.PrettyPrinter(indent=4)
# from bitarray import bitarray
import gmpy2
from gmpy2 import xmpz
import statistics
import scipy.stats
import os.path
import math
from datetime import datetime
import matplotlib.pyplot as plt

byteorder = "big" # sys.byteorder

class Env:
	# stores environment settings and data arrays

	# command line arguments
	parms = [
		{ "name":"num_states", "kw":{"help":"Number of states in finite-state automaton", "type":int},
	 	  "flag":"s", "required_init":"i", "default":100 },
		{ "name":"num_actions", "kw":{"help":"Number of actions in finite-state automaton", "type":int},
	 	  "flag":"r", "required_init":"i", "default":10 },
	 	{ "name":"num_choices", "kw":{"help":"Number of actions per state in finite-state automaton", "type":int},
	 	  "flag":"c", "required_init":"i", "default":10 },
	 	{ "name":"num_trials", "kw":{"help":"Number of trials to run (used when generating table)", "type":int},
	 	  "flag":"t", "required_init":"i", "default":3 },
	 	{ "name":"verbosity", "kw":{"help":"Verbosity of output, 0-minimum, 1-show fsa, 2-show errors", "type":int},
		  "flag":"v", "required_init":"i", "default":0 },
		{ "name":"sdm_word_length", "kw":{"help":"Word length for SDM memory, 0 to disable", "type":int},
	 	  "flag":"w", "required_init":"i", "default":512 },
		{ "name":"sdm_method", "kw":{"help":"0-normal SDM, 1-bind SDM, 2-both (combo)", "type":int},
	 	  "flag":"o", "required_init":"i", "default":0 },
	 	{ "name":"sdm_address_method", "kw":{"help":"0-hamming (normal), 1-np.random_choice", "type":int},
	 	  "flag":"k", "required_init":"i", "default":0 },
	 	{ "name":"bind_word_length", "kw":{"help":"Word length for binding memory, 0 to disable", "type":int},
	 	  "flag":"b", "required_init":"i", "default":512 },
	 	{ "name":"num_rows", "kw":{"help":"Number rows in sdm memory","type":int},
	 	  "flag":"m", "required_init":"i", "default":2048 },
		{ "name":"activation_count", "kw":{"help":"Number memory rows to activate for each address (sdm)","type":int},
		  "flag":"a", "required_init":"m", "default":20},
		{ "name":"counter_magnitude", "kw":{"help":"Max magnitude of SDM counter; zero if no limit","type":int},
		  "flag":"M", "required_init":"m", "default":0},
		{ "name":"counter_zero_ok", "kw":{"help":"Allow counter value to be zero, 1-yes, 0-no","type":int,
		  "choices":[0, 1]}, "flag":"Z", "required_init":"m", "default":1},
		{ "name":"noise_percent", "kw":{"help":"Percent of bits to change in memory to test noise resiliency",
		  "type":float}, "flag":"n", "required_init":"m", "default":0.0},
		{ "name":"debug", "kw":{"help":"Debug mode","type":int, "choices":[0, 1]},
		  "flag":"d", "required_init":"", "default":0},
		{ "name":"save_sdm_counter_stats", "kw":{"help":"Save sdm counter statistics and show histogram","type":int,
		  "choices":[0, 1]}, "flag":"u", "required_init":"", "default":0},
		{ "name":"generate_error_vs_storage_table", "kw":{"help":"Generate table to make plots (error vs storage)","type":int, "choices":[0, 1]},
		  "flag":"g", "required_init":"", "default":0},
		{ "name":"generate_error_vs_bitflips_table", "kw":{"help":"Generate table to make plots (error vs bithlips)","type":int, "choices":[0, 1]},
		  "flag":"f", "required_init":"", "default":0},
		{ "name":"error_vs_bitflips_table_start_at_equal_storage", "kw":{"help":"Start at equal storage for error vs bithlips table","type":int, "choices":[0, 1]},
		  "flag":"j", "required_init":"", "default":0},
		{ "name":"table_start", "kw":{"help":"Start value for table generation (storage or bit flip percent)","type":int},
		  "flag":"x", "required_init":"", "default":100000},
		{ "name":"table_step", "kw":{"help":"Step value for table generation","type":int},
		  "flag":"y", "required_init":"", "default":100000},
		{ "name":"table_stop", "kw":{"help":"Stop value for table generation","type":int},
		  "flag":"z", "required_init":"", "default":1000000},
		{ "name":"exclude_item_memory", "kw":{"help":"Exclude item memory and address sizes when calculating storage requirements",
			"type":int, "choices":[0, 1]}, "flag":"q", "required_init":"", "default":0},
		# { "name":"format", "kw":{"help":"Format used to store items and hard addresses, choices: "
		#    "int8, np.packbits, bitarray, gmpy2, gmpy2pure, colsum"},
		#   "flag":"f", "required_init":"i", "default":"int8", "choices":["int8", "np.packbits", "bitarray", "gmpy2",
		#   "gmpy2pure"]},
		# { "name":"num_items", "kw":{"help":"Number items in item memory","type":int},
		#   "flag":"i", "required_init":"m", "default":10},
		# { "name":"num_reps", "kw":{"help":"Number repepitions though itmes to match hard locations","type":int},
		#   "flag":"t", "required_init":"m", "default":50},
		{ "name":"seed", "kw":{"help":"Random number seed","type":int},
		  "flag":"e", "required_init":"i", "default":2021},
		  ]

	def __init__(self):
		self.parse_command_arguments()
		self.display_settings()
		self.initialize()

	def parse_command_arguments(self):
		parser = argparse.ArgumentParser(description='Test formats for storing hard locations for a SDM.',
			formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		# also make parser for interactive updating parameters (does not include defaults)
		iparse = argparse.ArgumentParser(description='Update sdm parameters.') # exit_on_error=False)
		for p in self.parms:
			parser.add_argument("-"+p["flag"], "--"+p["name"], **p["kw"], default=p["default"])
			iparse.add_argument("-"+p["flag"], "--"+p["name"], **p["kw"])  # default not used for interactive update
		self.iparse = iparse # save for later parsing interactive input
		args = parser.parse_args()
		self.pvals = {p["name"]: getattr(args, p["name"]) for p in self.parms}

	def initialize(self):
		# initialize sdm, char_map and merge
		print("Initializing.")
		np.random.seed(self.pvals["seed"])


	def display_settings(self):
		print(self.get_settings())
		# print("Settings:")
		# for p in self.parms:
		# 	print("%s %s: %s" % (p["flag"], p["name"], self.pvals[p["name"]]))

	def get_settings(self):
		msg = "Started at: %s\n" % datetime.now()
		# msg += "Comment: after fixing sdm size to include address memory"
		msg += "Arguments:\n"
		msg += " ".join(sys.argv)
		msg += "\nSettings:\n"
		for p in self.parms:
			msg += "%s %s: %s\n" % (p["flag"], p["name"], self.pvals[p["name"]])
		return msg


	# def update_settings(self, line):
	# 	instructions = ("Update settings using 'u' followed by KEY VALUE pair(s), where keys are:\n" +
	# 		'; '.join(["-"+p["flag"] + " --"+p["name"] for p in self.parms]))
	# 	#	" -w --world_length; -r --num_rows; -a --activation_count; -cmf --char_match_fraction; -s --string_to_store"
	# 	if len(line) < 4:
	# 		self.display_settings()
	# 		print(instructions)
	# 		return
	# 	try:
	# 		args = self.iparse.parse_args(shlex.split(line))
	# 	except Exception as e:
	# 	# except argparse.ArgumentError:
	# 		print('Invalid entry, try again:\n%s' % s)
	# 		return
	# 	updated = []
	# 	for p in self.parms:
	# 		name = p["name"]
	# 		val = getattr(args, name)
	# 		if val is not None:
	# 			if self.pvals[name] == val:
	# 				print("%s unchanged (is already %s)" % (name, val))
	# 			else:
	# 				self.pvals[name] = val
	# 				updated.append("%s=%s" % (name, val))
	# 				if p["required_init"]:
	# 					self.required_init += p["required_init"]
	# 	if updated:
	# 		print("Updated: %s" % ", ".join(updated))
	# 		self.display_settings()
	# 		print("required_init=%s" % self.required_init)
	# 	else:
	# 		print("Nothing updated")


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


# scratch_code = """

class Memory_Usage:
	# for storing and displaying memory usage of different storage techniques

	def __init__(self, name):
		self.name = name  # name of memory storage technique, e.g. Overman SDM

	def set_usage(self, address_memory, storage_memory, item_memory):
		# values specified in bytes
		self.address_memory = address_memory
		self.storage_memory = storage_memory
		self.item_memory = item_memory

	def get_total(self):
		return self.address_memory + self.storage_memory + self.item_memory

	def __repr__(self):
		return '%s- address:%s, storage:%s, item:%s, total:%s' % (self.name,
			self.address_memory, self.storage_memory, self.item_memory, self.get_total())



class Memory:
	# abstract class for memory of any type (binding and various types of SDM)

	def __init__(self, pvals):
		self.pvals = pvals
		self.debug = pvals["debug"]
		self.verbosity = pvals["verbosity"]
		self.initialize_storage()

	# folling must be overidden

	def initialize_im(self, item_count):
		# initialize item memory
		assert False, "must be ovridden"

	def initialize_storage(self):
		# initialize array used to store data, e.g. a 2-D array (for SDM) or 1 1-D array for binding
		assert False, "must be ovridden"

	def store(self, address, data):
		# using address save data
		assert False, "must be ovridden"

	def finalize_storage(self):
		pass

	def recall(address):
		# recall data stored at address
		assert False, "must be ovridden"

	def get_storage_requirements(self):
		# return requirements for storage as a text string
		assert False, "must be ovridden"


class SdmA(Memory):
	# abstracted version of SDM

	def initialize_storage(self):
		self.word_length = self.pvals["sdm_word_length"]
		# self.address_length = self.word_length
		self.num_rows = self.pvals["num_rows"]
		self.nact = self.pvals["activation_count"]
		self.storage = np.zeros((num_rows, word_length), dtype=np.int16)
		self.addresses = initialize_binary_matrix(num_rows, word_length)
		self.fmt = "0%sb" % word_length

# """

class Sdm:
	# implements a sparse distributed memory
	def __init__(self, address_length=128, word_length=128, num_rows=512, nact=5, noise_percent=0,
		sdm_address_method=0, counter_magnitude=0, counter_zero_ok=1, debug=False):
		# nact - number of active addresses used (top matches) for reading or writing
		self.address_length = address_length
		self.word_length = word_length
		self.num_rows = num_rows
		self.nact = nact
		self.noise_percent = noise_percent
		self.data_array = np.zeros((num_rows, word_length), dtype=np.int16)
		self.addresses = initialize_binary_matrix(num_rows, word_length)
		self.debug = debug
		self.sdm_address_method = sdm_address_method
		self.counter_magnitude = counter_magnitude  # maximum possible magnitute of counter when finalized
		self.counter_zero_ok = counter_zero_ok  # Allow counter to be zero.  1-yes, 0-no.
												# Used to force binary counters (-1/+1 without zero)
		self.fmt = "0%sb" % word_length
		self.hits = np.zeros((num_rows,), dtype=np.int16)
		self.stored = {} # stores number of times value stored and read [<data>, <store_count>, <read_count>]

	def store(self, address, data):
		# store binary word data at top nact addresses matching address
		top_matches = find_matches(self.addresses, address, self.nact, index_only = True, distribute_ties=True,
			sdm_address_method=self.sdm_address_method)
		d = int2iar(data, self.word_length)
		for i in top_matches:
			self.data_array[i] += d
			# check here for magnitude value greater than 128
			overflow = np.any(np.abs(self.data_array[i] > 127))
			if overflow and False:
				print("Found overflow in SDM counter")
				sys.exit("Found overflow in SDM counter")
				np.clip(self.data_array[i], -127, 127, out=self.data_array[i])
			self.hits[i] += 1
		if self.debug:
			print("store\n addr=%s\n data=%s" % (format(address, self.fmt), format(data, self.fmt)))
		self.save_stored(address, data, action="store")


	def save_stored(self, address, data, action):
		assert action in ("store", "read")
		if address not in self.stored:
			if action == "read":
				print("attempt to read value that is not stored")
				return
			self.stored[address] = [data, 1, 0]
			return
		if action == "store":
			print("storing multiple times at same address")
			return
		# reading at address
		self.stored[address][2] += 1

	def show_stored(self):
		not_read_count = 0
		for b in self.stored.keys():
			if self.stored[b][2] == 0:
				not_read_count += 1
		print("Numer of items stored = %s, not_read_count=%s" % (len(self.stored), not_read_count))


	def add_noise(self):
		# add noise to counters to test noise resiliency
		if self.noise_percent == 0:
			return
		for i in range(self.num_rows):
			add_noise(self.data_array[i], self.noise_percent)

	def truncate_counters(self):
		# truncate counters to magniture specified
		if self.counter_magnitude == 0:
			return
		self.data_array[self.data_array > self.counter_magnitude] = self.counter_magnitude
		self.data_array[self.data_array < -self.counter_magnitude] = -self.counter_magnitude
		if self.counter_zero_ok == 0:
			# replace zero counter values with random +1 or -1
			random_plus_or_minus_one = np.random.randint(0,high=2,size=(self.num_rows, self.word_length), dtype=np.int8) * 2 -1
			# there might be a fast way to do this in numpy without making a new copy of the array, but I couldn't find a way
			for i in range(self.num_rows):
				for j in range(self.word_length):
					if self.data_array[i,j] == 0:
						self.data_array[i,j] = random_plus_or_minus_one[i,j]
			# self.data_array = numpy.where(self.data_array == 0, [x, y, ]/)
			# self.data_array[self.data_array == 0] = random_plus_or_minus_one

	def show_hits(self):
		# display histogram of overlapping hits
		midpoint = int(self.num_rows / 2)
		first_half_count = np.sum(self.hits[0:midpoint])
		second_half_count = np.sum(self.hits[midpoint:])
		print("first_half_count: %s, second_half_count %s" % (first_half_count, second_half_count))
		mean_hits = np.mean(self.hits)
		stdev_hits = np.std(self.hits)
		predicted_mean = np.sum(self.hits) / self.num_rows
		predicted_stdev = math.sqrt(predicted_mean)
		print("hits mean=%s, predicted_mean=%s, stdev=%s, predicted_stdev=%s, sum=%s, num_rows=%s" % (
			mean_hits, predicted_mean, stdev_hits, predicted_stdev, np.sum(self.hits), self.num_rows))
		values, counts = np.unique(self.hits, return_counts=True)
		vc = [(values[i], counts[i]) for i in range(len(values))]
		vc.sort(key = lambda y: (y[0], y[1]))
		print("hits - counts:")
		pp.pprint(vc)

	def read(self, address):
		top_matches = find_matches(self.addresses, address, self.nact, index_only = True,
			distribute_ties=True, sdm_address_method=self.sdm_address_method)
		i = top_matches[0]
		isum = np.int32(self.data_array[i].copy())  # np.int32 is to convert to int32 to have range for sum
		for i in top_matches[1:]:
			isum += self.data_array[i]
		isum2 = iar2int(isum)
		if self.debug:
			print("read\n addr=%s\n top_matches=%s\n data=%s\nisum=%s," % (format(address, self.fmt), top_matches,
				format(isum2, self.fmt), isum))
		# self.save_stored(address, None, action="read")
		return isum2

	def test():
		# test sdm
		word_length = 512
		num_rows = 10
		nact = 5
		debug = True
		sdm = Sdm(address_length=word_length, word_length=word_length, num_rows=num_rows, nact=nact, debug=debug)
		fmt = sdm.fmt
		a1 = random.getrandbits(word_length)
		a2 = random.getrandbits(word_length)
		s1 = random.getrandbits(word_length)
		s2 = random.getrandbits(word_length)
		sr = random.getrandbits(word_length)
		sdm.store(a1, s1)
		sdm.store(a2, s2)
		r1 = sdm.read(a1)
		r2 = sdm.read(a2)
		print("a1 = %s" % format(a1, fmt))
		print("s1 = %s" % format(s1, fmt))
		print("r1 = %s, diff=%s" % (format(r1, fmt), gmpy2.popcount(s1^r1)))
		print("a2 = %s" % format(a2, fmt))
		print("s2 = %s" % format(s2, fmt))
		print("r2 = %s, diff=%s" % (format(r2, fmt), gmpy2.popcount(s2^r2)))
		print("random distance: %s, %s" % (gmpy2.popcount(sr^s1), gmpy2.popcount(sr^s2)))

class Sdm_counter_usage():
	# record and plot usage of counters (magnitute).  Used to estimate efficiency

	def __init__(self):
		self.counter_usage = []  # stores: (size, nrows, stdev)

	def save_size(self, size):
		# save total size of storage for include in record routine
		self.size = size

	def record(self, sdm):
		# record counter usage for sdm
		# size is number of bytes for sdm
		self.plot_hist(sdm)
		mean = np.mean(sdm.data_array)
		std = np.std(sdm.data_array)
		nrows = sdm.data_array.shape[0]
		usage = (self.size, nrows, mean, std)
		self.counter_usage.append((usage))

	def save_to_file(self, env, file_name="sdm_counter_usage"):
		fname = get_file_name(file_name)
		fp = open(fname,'w')
		fp.write(env.get_settings())
		fp.write("# following has sdm: (size, nrows, mean, stdev), for sdm counters\n")
		fp.write(pprint.pformat(self.counter_usage, indent=4))
		fp.close()

	def plot_hist(self, sdm):
		# x = np.histogram(sdm.data_array, bins="auto")
		# the histogram of the data
		counts = sdm.data_array.flatten()
		nbins = self.get_nbins(counts)
		n, bins, patches = plt.hist(counts, nbins, density=True, facecolor='g', alpha=0.75)
		plt.xlabel('Counter value')
		plt.ylabel('Probability')
		size_title = "10^6 bytes" if self.size == 1000000 else "%s kB" % (int(self.size / 1000))
		plt.title('Histogram of SDM counter values for size %s' % size_title)
		# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
		# plt.xlim(40, 160)
		# plt.ylim(0, 0.03)
		plt.grid(True)
		plt.show()

	def get_nbins(self, xv):
		# calculate number of bins to use in histogram for values in xv
		xv_range = max(xv) - min(xv)
		if xv_range == 0:
			nbins = 3
		else:
			nbins = int(xv_range) + 1
		return nbins


class Bundle:
	# bundle in hd vector

	def __init__(self, word_length, noise_percent=0, debug=False):
		self.word_length = word_length
		self.noise_percent = noise_percent
		self.debug = debug
		self.fmt = "0%sb" % word_length
		self.bundle = np.zeros((self.word_length, ), dtype=np.int16)

	# def make_summand(self, n):
	# 	return int2iar(n, self.word_length)
	# 	# n is an integer to add to counter, binary bits
	# 	# returns numpy int8 array with: where n bit is zero, -1, where n bit is one, +1
	# 	# first element of array is set to most significant bit, to match bin() representation
	# 	bool_vals = list(xmpz(n).iter_bits(stop=self.word_length))
	# 	s = np.where(bool_vals, 1, -1)
	# 	# flip for most significant bit is first
	# 	s = np.flipud(s)
	# 	return s

	def add(self, v):
		# add binary vector v to bundle
		d = int2iar(v, self.word_length)
		self.bundle += d
		if self.debug:
			print("add  %s" % format(v, self.fmt))
			print("bundle=%s" % self.bundle)

	def binarize(self):
		# convert from bundle to binary then to int
		if self.noise_percent > 0:
			add_noise(self.bundle, self.noise_percent)
		return iar2int(self.bundle)

	def test():
		word_length = 1000
		bun = Bundle(word_length, debug=True)
		fmt = bun.fmt
		a1 = random.getrandbits(word_length)
		a2 = random.getrandbits(word_length)
		s1 = random.getrandbits(word_length)
		s2 = random.getrandbits(word_length)
		sr = random.getrandbits(word_length)
		print("a1 = %s" % format(a1, fmt))
		print("s1 = %s" % format(s1, fmt))
		print("a1^s=%s" % format(a1^s1, fmt))
		bun.add(a1^s1)
		bun.add(a2^s2)
		b = bun.binarize()
		print("binb=%s" % format(b, fmt))
		print("recalling from bundle:")
		print("a1^b=%s" % format(a1^b, fmt))
		print("  s1=%s, diff=%s" % (format(s1, fmt), gmpy2.popcount(a1^b^s1)))
		print("a2^b=%s" % format(a2^b, fmt))
		print("  s2=%s, diff=%s" % (format(s2, fmt), gmpy2.popcount(a2^b^s2)))
		print("random distance: %s, %s" % (gmpy2.popcount(a1^b^sr), gmpy2.popcount(a2^b^sr)))


class FSA:
	# finite-state automaton

	def __init__(self, num_states, num_actions, num_choices, debug=False):
		# num_choices is number of actions per state
		self.num_states = num_states
		self.num_actions = num_actions
		self.num_choices = num_choices
		self.num_transitions = num_states * num_choices # number items stored in sdm or the bundle
		self.debug = debug
		fsa = []
		for i in range(num_states):
			possible_actions = list(range(self.num_actions))
			possible_next_states = list(range(self.num_states))
			nas = []
			for j in range(num_choices):
				nas.append( ( pop_random_element(possible_actions), pop_random_element(possible_next_states) ) )
			fsa.append(nas)
		self.fsa = fsa

	def display(self):
		# display the fsa
		for state_num in range(self.num_states):
			state_name = "s%s" % state_num
			ns = self.fsa[state_num]
			next_states = ', '.join(["a%s->s%s" % (ns[i][0], ns[i][1]) for i in range(len(ns))])
			print("%s: %s" % (state_name, next_states))

	def initialize_item_memory(self, word_length):
		self.word_length = word_length
		self.states_im = initialize_binary_matrix(self.num_states, word_length, self.debug)
		self.actions_im = initialize_binary_matrix(self.num_actions, word_length, self.debug)
		self.bytes_required = int((self.num_states + self.num_actions) * word_length / 8)


	def get_storage_requirements_for_im(self):
		return "im: %s bytes" % int((self.num_states + self.num_actions) * self.word_length / 8)


class FSA_store:
	# abstract class for storing FSA.  Must subclass to implement different storage methods

	def __init__(self, fsa, word_length, debug, pvals, custor=None):
		# custor is Sdm_counter_usage object if saving counter_usage (only used for sdm)
		if word_length == 0:
			# don't process if wordlength is zero
			return
		self.fsa = fsa
		self.word_length = word_length
		self.debug = debug
		self.pvals = pvals
		self.custor = custor
		self.fsa.initialize_item_memory(word_length)
		print("Recall from %s" % self.__class__.__name__)
		self.initialize()
		self.store()
		self.recall()

	def store(self):
		# store the fsa
		for state_num in range(self.fsa.num_states):
			state_v = self.fsa.states_im[state_num]
			action_next_state_list = self.fsa.fsa[state_num]
			for action_nexts in action_next_state_list:
				action_num, next_state_num = action_nexts
				action_v = self.fsa.actions_im[action_num]
				next_state_v = self.fsa.states_im[next_state_num]
				self.save_transition(state_v, action_v, next_state_v)
		self.finalize_store()

	def recall(self):
		# recall the fsa
		num_errors = 0
		nret = 3
		bit_error_counts = []
		hdiff_difs = []
		mean_dhds = []
		stdev_dhds = []
		item_count = 0
		vr = random.getrandbits(self.word_length)
		for state_num in range(self.fsa.num_states):
			state_v = self.fsa.states_im[state_num]
			action_next_state_list = self.fsa.fsa[state_num]
			for action_nexts in action_next_state_list:
				item_count += 1
				action_num, next_state_num = action_nexts
				action_v = self.fsa.actions_im[action_num]
				next_state_v = self.fsa.states_im[next_state_num]
				found_v = self.recall_transition(state_v, action_v)
				# found_v = rotate_left(state_v ^ action_v ^ self.bundle, self.word_length)
				if self.debug:
					print("s%s: a%s, hdist=%s, random=" %(state_num, action_num, gmpy2.popcount(found_v ^ next_state_v)),
						gmpy2.popcount(found_v ^ vr))
				im_matches, mean_dhd, stdev_dhd = find_matches(self.fsa.states_im, found_v, nret, 
					debug=self.debug, include_stats=True)
				mean_dhds.append(mean_dhd)
				stdev_dhds.append(stdev_dhd)
				if self.debug:
					print("find_matches returned: %s" % im_matches)
				found_i = im_matches[0][0]
				bit_error_count = im_matches[0][1]  # hamming distance between recalled vector and closest match
				hdiff_dif = im_matches[1][1] - im_matches[0][1]
				if found_i != next_state_num:
					# since this is in error, hdiff_dif needs to be calculated based on hdif to next_state_v
					bit_error_count = gmpy2.popcount(found_v ^ next_state_v)
					hdiff_dif = im_matches[0][1] - bit_error_count
					num_errors += 1
					if self.pvals["verbosity"] > 1:
						print("error, expected state=s%s, found_state=s%s, found_hdif=%s, hdif_dif=%s, im_matches=%s" % (
							next_state_num, 
							found_i, gmpy2.popcount(self.fsa.states_im[found_i] ^ found_v), hdiff_dif, im_matches))
				else:
					assert gmpy2.popcount(found_v ^ next_state_v) == bit_error_count
				hdiff_difs.append(hdiff_dif)
				bit_error_counts.append(bit_error_count)
		mean_hdd = statistics.mean(hdiff_difs)
		stdev_hdd = statistics.stdev(hdiff_difs)
		mean_dhd = statistics.mean(mean_dhds)  # distractor hamming distance (not just closest one)
		stdev_dhd = statistics.mean(stdev_dhds)
		probability_of_error = scipy.stats.norm(mean_hdd, stdev_hdd).cdf(0.0)
		probability_correct = 1.0 - probability_of_error
		actual_fraction_error = num_errors / item_count
		actual_fraction_correct = 1.0 - actual_fraction_error
		# print("bit_error_counts=%s" % bit_error_counts)
		mean_bit_error_count = statistics.mean(bit_error_counts)
		stdev_bit_error_count = statistics.stdev(bit_error_counts)	
		print("num_errors=%s/%s, hdiff avg=%0.1f, std=%0.1f, prob error=%.2e" % (num_errors, item_count,
			mean_hdd, stdev_hdd, scipy.stats.norm(mean_hdd, stdev_hdd).cdf(0.0)))
		# ? expected_mean, expected_stdev = get_expected_mean_and_stdev(item_count, )
		# print("Expected: error=%.2e, correct=%.2e; actual: error=%.2e, correct=%.2e" % (
		# 	probability_of_error, probability_correct, actual_fraction_error,
		# 	actual_fraction_correct))
		print("error expected=%s actual=%s;  correct expected=%s actual=%s" % (
			probability_of_error, actual_fraction_error, probability_correct,
			actual_fraction_correct))
		print("bit_error_count mean=%s, stdev=%s" % (mean_bit_error_count, stdev_bit_error_count))
		if self.pvals["exclude_item_memory"]:
			total_storage_required = self.bytes_required
		else:
			total_storage_required = self.fsa.bytes_required + self.bytes_required
		print("Storage required: %s, %s, total: %.3e" % (self.fsa.get_storage_requirements_for_im(),
			self.get_storage_requirements(), total_storage_required))

		# save info for retrieval if generating table
		self.sinfo = {"item_count": item_count, "num_errors": num_errors,
			"actual_fraction_error": actual_fraction_error,
			"actual_fraction_correct": actual_fraction_correct,
			"probability_of_error": probability_of_error,
			"probability_correct": probability_correct,
			"total_storage_required": total_storage_required,
			"mean_bit_error_count": mean_bit_error_count,
			"stdev_bit_error_count": stdev_bit_error_count,
			"mean_dhd": mean_dhd,
			"stdev_dhd": stdev_dhd}

	# following classes must or can be overridden by subclasses

	def initialize(self):
		# subclass must be overridden
		sys.exit("initialize must be overridden")

	def save_transition(self, state_v, action_v, next_state_v):
		sys.exit("save_transition must be overridden")

	def recall_transition(self, state_v, action_v):
		# recall next_state_v from state_v and action_v
		sys.exit("recall_transition must be overridden")

	def finalize_store(self):
		pass

	def get_storage_requirements(self):
		assert False, "get_storage_requirements must be overridden"


class FSA_bind_store(FSA_store):
	# store FSA using binding into a single vector

	def initialize(self):
		self.bundle = Bundle(self.word_length, noise_percent=self.pvals["noise_percent"])
		self.bytes_required = int(self.word_length / 8)

	def save_transition(self, state_v, action_v, next_state_v):
		add_v = state_v ^ action_v ^ rotate_right(next_state_v, self.word_length)
		self.bundle.add(add_v)

	def finalize_store(self):
		# convert from counter to binary
		self.bundle = self.bundle.binarize()

	def recall_transition(self, state_v, action_v):
		# recall next_state_v from state_v and action_v
		found_v = rotate_left(state_v ^ action_v ^ self.bundle, self.word_length)
		return found_v

	def get_storage_requirements(self):
		return ("bundle: %s bytes" % int(self.word_length / 8))


class FSA_sdm_store(FSA_store):
	# store FSA using SDM (sparse distributed memory)

	def initialize(self):
		self.sdm = Sdm(address_length=self.word_length, word_length=self.word_length,
			num_rows=self.pvals["num_rows"], nact=self.pvals["activation_count"],
			noise_percent=self.pvals["noise_percent"], sdm_address_method=self.pvals["sdm_address_method"],
			counter_magnitude=self.pvals["counter_magnitude"], counter_zero_ok=self.pvals["counter_zero_ok"],
			debug=self.debug)
		self.bytes_required = (self.word_length + int(self.word_length / 8)) * self.pvals["num_rows"]

	def save_transition(self, state_v, action_v, next_state_v):
		# self.sdm.store(state_v ^ action_v, next_state_v)
		# store binding with state and action so all next_states stored are unique
		# self.sdm.store(state_v ^ action_v, state_v ^ action_v ^ next_state_v)
		self.sdm.store(state_v ^ action_v, state_v ^ action_v ^ rotate_right(next_state_v, self.word_length))

	def finalize_store(self):
		if self.pvals["noise_percent"] > 0:
			self.sdm.add_noise()
		if self.custor is not None:
			self.custor.record(self.sdm)  # save counter statistics
		if self.pvals["counter_magnitude"] > 0:
			self.sdm.truncate_counters()
		# self.sdm.show_hits()

	def recall_transition(self, state_v, action_v):
		# recall next_state_v from state_v and action_v
		# return self.sdm.read(state_v ^ action_v)
		# found_v = self.sdm.read(state_v ^ action_v)
		# unbind with state and action
		recalled_v = self.sdm.read(state_v ^ action_v)
		found_v = rotate_left(state_v ^ action_v ^ recalled_v, self.word_length)
		# return state_v ^ action_v ^ found_v
		return found_v

	def get_storage_requirements(self):
		return ("sdm counter: %s bytes" % (self.word_length * self.pvals["num_rows"]))


class FSA_combo_store(FSA_store):
	# store FSA using SDM (sparse distributed memory) and binding

	def initialize(self):
		self.sdm = Sdm(address_length=self.word_length, word_length=self.word_length,
			num_rows=self.pvals["num_rows"], nact=self.pvals["activation_count"],
			noise_percent=self.pvals["noise_percent"], sdm_address_method=self.pvals["sdm_address_method"],
			debug=self.debug)
		self.bytes_required = self.word_length * self.pvals["num_rows"]

	def save_transition(self, state_v, action_v, next_state_v):
		self.sdm.store(state_v, action_v ^ next_state_v)

	def finalize_store(self):
		if self.pvals["noise_percent"] > 0:
			self.sdm.add_noise()

	def recall_transition(self, state_v, action_v):
		# recall next_state_v from state_v and action_v
		return self.sdm.read(state_v) ^ action_v

	def get_storage_requirements(self):
		return ("sdm counter: %s bytes" % (self.word_length * self.pvals["num_rows"]))

def get_file_name(base_name):
		# return name of file that does not yet exist
		count = 0
		full_name = "%s.txt" % base_name
		while os.path.isfile(full_name):
			count += 1
			full_name = "%s_%s.txt" % (base_name, count)
		return full_name

class Table_Generator_error_vs_storage():
	# generate table of error vs storage

	def __init__(self, env, fsa):
		self.env = env
		self.pvals = env.pvals
		self.fsa = fsa
		self.num_items = fsa.num_states + fsa.num_actions
		table_start = self.pvals["table_start"]
		table_step = self.pvals["table_step"]
		table_stop = self.pvals["table_stop"]
		assert table_start < table_stop, "table_start (%s) must be less than table_stop (%s)" % (
			table_start, table_stop)
		assert table_step > 0, "table_step must be > 0, is %s" % table_step
		num_steps = int((table_stop - table_start) / table_step)
		assert num_steps > 3, "number of steps must be > 3, is %s" % num_steps
		print("table generation start=%s, step=%s, stop=%s, num_steps=%s" % (table_start, table_step,
			table_stop, num_steps))
		self.storage_min = table_start  # min amount of storage
		self.storage_max = table_stop # max amount of storage
		self.storage_step = table_step # step size
		# assert self.pvals["num_states"] == 100
		# assert self.pvals["num_actions"] == 10
		# assert self.pvals["num_choices"] == 10
		# assert self.pvals["sdm_word_length"] == 512
		self.generate_table()

	def format_info(self, rid, storage, mtype, mlen, sinfo):
		# create row in output table from sinfo
		# self.sinfo = {"item_count": item_count, "num_errors": num_errors,
		# 	"actual_fraction_error": actual_fraction_error,
		# 	"actual_fraction_correct": actual_fraction_correct,
		# 	"probability_of_error": probability_of_error,
		# 	"probability_correct": probability_correct.pvals,
		# 	"total_storage_required": total_storage_required}
		# assert sinfo["item_count"] == 1000
		row = "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (rid, storage, mtype, mlen, sinfo["num_errors"],
			sinfo["actual_fraction_error"], sinfo["probability_of_error"], 
			sinfo["mean_bit_error_count"],sinfo["stdev_bit_error_count"],
			sinfo["mean_dhd"], sinfo["stdev_dhd"],
			sinfo["total_storage_required"])
		return row

	def get_sdm_activation_count(self, m, k):
		# compute number of rows in sdm to be active
		# m is number of rows in sdm.  k is number of items being stored (1000)
		# nact = round(sdm_num_rows / 100)  # originally used in hdfsa.py
		nact = round( m/((2*m*k)**(1/3)) )
		return nact

	def generate_table(self):
		file_name = get_file_name("sdata")
		fp = open(file_name,'w')
		fp.write(self.env.get_settings())
		fp.write("\n-----Data starts here-----\n")
		fp.write("rid\tstorage\tmtype\tmem_len\terror_count\tfraction_error\tprobability_of_error\t"
			"mean_bit_error_count\tstdev_bit_error_count\tmean_dhd\tstdev_dhd\ttotal_storage_required\n")
		print("storage\tbind_len\tsdm_len\tparameters")
		sid = 0
		scu = Sdm_counter_usage() if self.env.pvals["save_sdm_counter_stats"] == 1 else None
		for storage in range(self.storage_min, self.storage_max+1, self.storage_step):
			sid += 1
			tid = 0
			for trial in range(self.pvals["num_trials"]):
				tid += 1
				bind_l = self.bind_len(storage)
				sdm_num_rows = self.sdm_len(storage)
				# sdm_activation_count = round(sdm_num_rows / 100)
				sdm_activation_count = self.get_sdm_activation_count(sdm_num_rows, self.fsa.num_transitions)
				parameters = "-b %s -m %s -a %s" % (bind_l, sdm_num_rows, sdm_activation_count)
				self.pvals["num_rows"] = sdm_num_rows
				self.pvals["activation_count"] = sdm_activation_count
				print("%s\t%s\t%s\t%s" % (storage, bind_l, sdm_num_rows, parameters))
				fbs = FSA_bind_store(self.fsa, bind_l, self.pvals["debug"], self.pvals)
				rid = "%s.%s" % (sid, tid)
				fp.write(self.format_info(rid, storage, "bind", bind_l, fbs.sinfo))
				if tid == 1 and scu is not None:
					# save size and sdm counter_usage
					scu.save_size(storage)
					custor = scu
				else:
					custor = None
				fss = FSA_sdm_store(self.fsa, self.pvals["sdm_word_length"], self.pvals["debug"], self.pvals, custor)
				fp.write(self.format_info(rid, storage, "sdm", sdm_num_rows, fss.sinfo))
				# fp.flush()
		fp.close()
		if scu is not None:
			scu.save_to_file(self.env)  # save sdm counter usage to file

	def bind_len(self, storage):
		num_items_needing_memory = self.num_items if self.pvals["exclude_item_memory"] == 0 else 0
		# given storage in bytes, compute vector length to use that storage
		length = round((storage * 8) / (num_items_needing_memory + 1))
		return length

	def sdm_len(self, storage):
		# given storage in bytes, compute # rows in SDM memory to use that storage
		sdm_word_length = self.pvals["sdm_word_length"] # num bites in sdm address and memory
		num_items_needing_memory = self.num_items if self.pvals["exclude_item_memory"] == 0 else 0
		bytes_for_item_memory = round(num_items_needing_memory * sdm_word_length / 8)
		bytes_per_row = sdm_word_length + sdm_word_length / 8  # bytes needed for counters and address
		num_rows = round((storage - bytes_for_item_memory) / bytes_per_row)
		return num_rows


class Table_Generator_error_vs_bitflips():
	# generate table of error vs bitflips

	def __init__(self, env, fsa):
		self.env = env
		self.pvals = env.pvals
		self.fsa = fsa
		self.num_items = fsa.num_states + fsa.num_actions
		self.pflip_min = 0.0  # min percent bit flips
		self.pflip_max = 50.0 # max percent bit flips
		self.pflip_step = 5 # step percent bit flips
		assert self.pvals["num_states"] == 100
		assert self.pvals["num_actions"] == 10
		assert self.pvals["num_choices"] == 10
		assert self.pvals["sdm_word_length"] == 512
		if not self.pvals["error_vs_bitflips_table_start_at_equal_storage"]:
			# folloging give about the same level of erreo, about 1.0e-6
			# python hdfsa.py -s 100 -a 10 -c 10 -w 512 -m 1300 -a 13 -b 93000
			assert self.pvals["num_rows"] == 1300
			assert self.pvals["activation_count"] == 13
			assert self.pvals["bind_word_length"] == 93000
		else:
			# this should be made an argument
			storage = 1000000
			if storage == 800000:
				# following settings for 800000 bytes storage for both
				# 8.1     800000  bind    57658   0       0.0     0.0005658820455930389   800004
				# 8.1     800000  sdm     1549    0       0.0     1.14064404735783e-07    800128
				# python hdfsa.py -s 100 -a 10 -c 10 -w 512 -m 1549 -a 15 -b 57658 -j 1 -f 1
				assert self.pvals["num_rows"] == 1377
				assert self.pvals["activation_count"] == 10
				assert self.pvals["bind_word_length"] == 57658
			elif storage == 1000000:
				# following settings for 1000000 bytes storage for both
				# 10.6    1000000 bind    72072   0       0.0     4.0137960109522544e-05  999999
				# 10.6    1000000 sdm     1939    0       0.0     5.200091764358587e-09   999808
				# python hdfsa.py -s 100 -a 10 -c 10 -w 512 -m 1939 -a 19 -b 72072 -j 1 -f 1 -t 10
				assert self.pvals["num_rows"] == 1724  # reduced from 1939 for address space
				assert self.pvals["activation_count"] == 11
				assert self.pvals["bind_word_length"] == 72072
			else:
				sys.exit("Invalid storage size for bit flip table")
		self.generate_table()

	def format_info(self, rid, pflip, mtype, mlen, sinfo):
		# create row in output table from sinfo
		# self.sinfo = {"item_count": item_count, "num_errors": num_errors,
		# 	"actual_fraction_error": actual_fraction_error,
		# 	"actual_fraction_correct": actual_fraction_correct,
		# 	"probability_of_error": probability_of_error,
		# 	"probability_correct": probability_correct.pvals,
		# 	"total_storage_required": total_storage_required}
		assert sinfo["item_count"] == 1000
		row = "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (rid, pflip, mtype, mlen, sinfo["num_errors"],
			sinfo["actual_fraction_error"], sinfo["probability_of_error"], 
			sinfo["mean_bit_error_count"],sinfo["stdev_bit_error_count"],
			sinfo["mean_dhd"], sinfo["stdev_dhd"],
			sinfo["total_storage_required"])
		return row


		return row

	def generate_table(self):
		file_name = get_file_name("fdata")
		fp = open(file_name,'w')
		fp.write(self.env.get_settings())
		fp.write("\n-----Data starts here-----\n")
		fp.write("rid\t%s\tmtype\tmem_len\terror_count\tfraction_error\tprobability_of_error\t"
		"mean_bit_error_count\tstdev_bit_error_count\tmean_dhd\tstdev_dhd\ttotal_storage_required\n" % "pflip")

		print("pflip\tparameters")
		fid = 0
		pflip = self.pflip_min
		while pflip <= self.pflip_max:
			fid += 1
			tid = 0
			for trial in range(self.pvals["num_trials"]):
				tid += 1
				parameters = "-n %s" % pflip
				rid = "%s.%s" % (fid, tid)
				self.pvals["noise_percent"] = pflip
				print("%s\t%s\t%s" % (rid, pflip, parameters))
				fbs = FSA_bind_store(self.fsa, self.pvals["bind_word_length"], self.pvals["debug"], self.pvals)
				fp.write(self.format_info(rid, pflip, "bind", self.pvals["bind_word_length"], fbs.sinfo))
				fss = FSA_sdm_store(self.fsa, self.pvals["sdm_word_length"], self.pvals["debug"], self.pvals)
				fp.write(self.format_info(rid, pflip, "sdm", self.pvals["num_rows"], fss.sinfo))
			pflip += self.pflip_step
		fp.close()


def main_show_table():
	# if len(sys.argv) != 2:
	# 	sys.exit("Usage %s <storage (in bytes)>" % sys.argv[0])
	# storage = int(sys.argv[1])
	print("storage\tbind_len\tsdm_len\tparameters")
	for i in range(1, 11):
		storage = i * 100000
		bind_l = bind_len(storage)
		sdm_l = sdm_len(storage)
		parameters = "-b %s -m %s -a %s" % (bind_l, sdm_l, round(sdm_l / 100))
		print("%s\t%s\t%s" % (storage, bind_len(storage), sdm_len(storage)), parameters)


def main():
	env = Env()
	fsa = FSA(env.pvals["num_states"], env.pvals["num_actions"], env.pvals["num_choices"])
	if env.pvals["verbosity"] > 0:
		fsa.display()
	if env.pvals["generate_error_vs_storage_table"] == 1:
		Table_Generator_error_vs_storage(env, fsa)
		return
	if env.pvals["generate_error_vs_bitflips_table"] == 1:
		Table_Generator_error_vs_bitflips(env, fsa)
		return
	FSA_bind_store(fsa, env.pvals["bind_word_length"], env.pvals["debug"], env.pvals)
	if env.pvals["sdm_method"] in (0, 2):
		FSA_sdm_store(fsa, env.pvals["sdm_word_length"], env.pvals["debug"], env.pvals)
	if env.pvals["sdm_method"] in (1, 2):
		FSA_combo_store(fsa, env.pvals["sdm_word_length"], env.pvals["debug"], env.pvals)

main()
# test_select_top_matches_with_possible_ties()
# Bundle.test()
# Sdm.test()


