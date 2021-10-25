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


byteorder = "big" # sys.byteorder

class Env:
	# stores environment settings and data arrays

	# command line arguments
	parms = [
		{ "name":"num_states", "kw":{"help":"Number of states in finite-state automaton", "type":int},
	 	  "flag":"s", "required_init":"i", "default":20 },
		{ "name":"num_actions", "kw":{"help":"Number of actions in finite-state automaton", "type":int},
	 	  "flag":"r", "required_init":"i", "default":10 },
	 	{ "name":"num_choices", "kw":{"help":"Number of actions per state in finite-state automaton", "type":int},
	 	  "flag":"c", "required_init":"i", "default":3 },
	 	{ "name":"num_trials", "kw":{"help":"Number of trials to run (used when generating table)", "type":int},
	 	  "flag":"t", "required_init":"i", "default":3 },
	 	{ "name":"verbosity", "kw":{"help":"Verbosity of output, 0-minimum, 1-show fsa, 2-show errors", "type":int},
		  "flag":"v", "required_init":"i", "default":0 },
		{ "name":"sdm_word_length", "kw":{"help":"Word length for SDM memory, 0 to disable", "type":int},
	 	  "flag":"w", "required_init":"i", "default":512 },
		{ "name":"sdm_method", "kw":{"help":"0-normal SDM, 1-bind SDM, 2-both (combo)", "type":int},
	 	  "flag":"o", "required_init":"i", "default":0 },	 	  
	 	{ "name":"bind_word_length", "kw":{"help":"Word length for binding memory, 0 to disable", "type":int},
	 	  "flag":"b", "required_init":"i", "default":512 },
	 	{ "name":"num_rows", "kw":{"help":"Number rows in memory","type":int},
	 	  "flag":"m", "required_init":"i", "default":2048 },
		{ "name":"activation_count", "kw":{"help":"Number memory rows to activate for each address","type":int},
		  "flag":"a", "required_init":"m", "default":20},
		{ "name":"noise_percent", "kw":{"help":"Percent of bits to change in memory to test noise resiliency",
		  "type":float}, "flag":"n", "required_init":"m", "default":0.0},
		{ "name":"debug", "kw":{"help":"Debug mode","type":int, "choices":[0, 1]},
		  "flag":"d", "required_init":"", "default":0},
		{ "name":"generate_table", "kw":{"help":"Generate table to make plots","type":int, "choices":[0, 1]},
		  "flag":"g", "required_init":"", "default":0},
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
		print("Current settings:")
		for p in self.parms:
			print("%s %s: %s" % (p["flag"], p["name"], self.pvals[p["name"]]))

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


def find_matches(m, b, nret, index_only=False, debug=False):
	# m is 2-d array of binary values, first dimension is value index, second is binary number
	# b is binary number to match
	# nret is number of top matches to return
	# returns sorted tuple (i, c) where i is index and c is number of non-matching bits (0 is perfect match)
	# if index_only is True, only return the indices, not the c
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
	top_matches = matches[0:nret]
	if index_only:
		top_matches = [x[0] for x in top_matches]
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
	idx = rng.choice(word_length, size=num_to_flip)
	iar[idx] *= -1

def int2iar(n, width):
	# convert n (integer) to int array with integer bits 0, 1 mapped to -1, +1 in the array
	# width is the word_length (number of bits) in the integer
	# returns numpy int8 array with: where n bit is zero, -1, where n bit is one, +1
	bool_vals = list(xmpz(n).iter_bits(stop=width))
	s = np.where(bool_vals, 1, -1)
	s = np.flipud(s)   # flip for most significant bit is first
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
	def __init__(self, address_length=128, word_length=128, num_rows=512, nact=5, noise_percent=0, debug=False):
		# nact - number of active addresses used (top matches) for reading or writing
		self.address_length = address_length
		self.word_length = word_length
		self.num_rows = num_rows
		self.nact = nact
		self.noise_percent = noise_percent
		self.data_array = np.zeros((num_rows, word_length), dtype=np.int16)
		self.addresses = initialize_binary_matrix(num_rows, word_length)
		self.debug = debug
		self.fmt = "0%sb" % word_length
		self.hits = np.zeros((num_rows,), dtype=np.int16)

	def store(self, address, data):
		# store binary word data at top nact addresses matching address
		top_matches = find_matches(self.addresses, address, self.nact, index_only = True)
		d = int2iar(data, self.word_length)
		for i in top_matches:
			self.data_array[i] += d
			self.hits[i] += 1
		if self.debug:
			print("store\n addr=%s\n data=%s" % (format(address, self.fmt), format(data, self.fmt)))

	def add_noise(self):
		# add noise to counters to test noise resiliency
		if self.noise_percent == 0:
			return
		for i in range(self.num_rows):
			add_noise(self.data_array[i], self.noise_percent)

	def show_hits(self):
		# display histogram of overlapping hits
		values, counts = np.unique(self.hits, return_counts=True)
		vc = [(values[i], counts[i]) for i in range(len(values))]
		vc.sort(key = lambda y: (y[0], y[1]))
		print("hits - counts:")
		pp.pprint(vc)

	def read(self, address):
		top_matches = find_matches(self.addresses, address, self.nact, index_only = True)
		i = top_matches[0]
		isum = np.int32(self.data_array[i].copy())  # np.int32 is to convert to int32 to have range for sum
		for i in top_matches[1:]:
			isum += self.data_array[i]
		isum2 = iar2int(isum)
		if self.debug:
			print("read\n addr=%s\n top_matches=%s\n data=%s\nisum=%s," % (format(address, self.fmt), top_matches,
				format(isum2, self.fmt), isum))
		return isum2

	def test():
		# test sdm
		word_length = 30
		num_rows = 512
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

	# def clear(self):
	# 	# set data_array contents to zero
	# 	self.data_array.fill(0)

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

	def __init__(self, fsa, word_length, debug, pvals=None):
		if word_length == 0:
			# don't process if wordlength is zero
			return
		self.fsa = fsa
		self.word_length = word_length
		self.debug = debug
		self.pvals = pvals
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
		hdiffs = []
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
				im_matches = find_matches(self.fsa.states_im, found_v, nret, debug=self.debug)
				if self.debug:
					print("find_matches returned: %s" % im_matches)
				found_i = im_matches[0][0]
				hdiff_dif = im_matches[1][1] - im_matches[0][1]
				if found_i != next_state_num:
					# since this is in error, hdiff_dif needs to be calculated based on hdif to next_state_v
					hdiff_dif = im_matches[0][1] - gmpy2.popcount(found_v ^ next_state_v)
					num_errors += 1
					if self.pvals["verbosity"] > 1:
						print("error, expected state=s%s, found_state=s%s, found_hdif=%s, hdif_dif=%s, im_matches=%s" % (
							next_state_num, 
							found_i, gmpy2.popcount(self.fsa.states_im[found_i] ^ found_v), hdiff_dif, im_matches))
				hdiffs.append(hdiff_dif)
		mean = statistics.mean(hdiffs)
		stdev = statistics.stdev(hdiffs)
		probability_of_error = scipy.stats.norm(mean, stdev).cdf(0.0)
		probability_correct = 1.0 - probability_of_error
		actual_fraction_error = num_errors / item_count
		actual_fraction_correct = 1.0 - actual_fraction_error
		print("num_errors=%s/%s, hdiff avg=%0.1f, std=%0.1f, probability of error=%.2e" % (num_errors, item_count,
			mean, stdev, scipy.stats.norm(mean, stdev).cdf(0.0)))
		# ? expected_mean, expected_stdev = get_expected_mean_and_stdev(item_count, )
		# print("Expected: error=%.2e, correct=%.2e; actual: error=%.2e, correct=%.2e" % (
		# 	probability_of_error, probability_correct, actual_fraction_error,
		# 	actual_fraction_correct))
		print("error expected=%s actual=%s;  correct expected=%s actual=%s" % (
			probability_of_error, actual_fraction_error, probability_correct,
			actual_fraction_correct))
		total_storage_required = self.fsa.bytes_required + self.bytes_required
		print("Storage required: %s, %s, total: %.3e" % (self.fsa.get_storage_requirements_for_im(),
			self.get_storage_requirements(), total_storage_required))

		# save info for retrieval if generating table
		self.sinfo = {"item_count": item_count, "num_errors": num_errors,
			"actual_fraction_error": actual_fraction_error,
			"actual_fraction_correct": actual_fraction_correct,
			"probability_of_error": probability_of_error,
			"probability_correct": probability_correct,
			"total_storage_required": total_storage_required}

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
			noise_percent=self.pvals["noise_percent"], debug=self.debug)
		self.bytes_required = self.word_length * self.pvals["num_rows"]

	def save_transition(self, state_v, action_v, next_state_v):
		self.sdm.store(state_v ^ action_v, next_state_v)

	def finalize_store(self):
		if self.pvals["noise_percent"] > 0:
			self.sdm.add_noise()

	def recall_transition(self, state_v, action_v):
		# recall next_state_v from state_v and action_v
		return self.sdm.read(state_v ^ action_v)

	def get_storage_requirements(self):
		return ("sdm counter: %s bytes" % (self.word_length * self.pvals["num_rows"]))


class FSA_combo_store(FSA_store):
	# store FSA using SDM (sparse distributed memory) and binding

	def initialize(self):
		self.sdm = Sdm(address_length=self.word_length, word_length=self.word_length,
			num_rows=self.pvals["num_rows"], nact=self.pvals["activation_count"],
			noise_percent=self.pvals["noise_percent"], debug=self.debug)
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

class Table_Generator():
	# generate table of outputs

	def __init__(self, pvals, fsa):
		self.pvals = pvals
		self.fsa = fsa
		self.num_items = fsa.num_states + fsa.num_actions
		self.storage_min = 100000  # min amount of storage
		self.storage_max = 1000000 # max amount of storage
		self.storage_step = 100000 # step size
		assert self.pvals["num_states"] == 100
		assert self.pvals["num_actions"] == 10
		assert self.pvals["num_choices"] == 10
		assert self.pvals["sdm_word_length"] == 512
		self.generate_table()

	def get_file_name(self):
		# return name of file that does not yet exist
		file_name = "sdata"
		count = 0
		full_file_name = "%s.txt" % file_name
		while os.path.isfile(full_file_name):
			count += 1
			full_file_name = "%s_%s.txt" % (file_name, count)
		return full_file_name

	def format_info(self, rid, storage, mtype, mlen, sinfo):
		# create row in output table from sinfo
		# self.sinfo = {"item_count": item_count, "num_errors": num_errors,
		# 	"actual_fraction_error": actual_fraction_error,
		# 	"actual_fraction_correct": actual_fraction_correct,
		# 	"probability_of_error": probability_of_error,
		# 	"probability_correct": probability_correct.pvals,
		# 	"total_storage_required": total_storage_required}
		assert sinfo["item_count"] == 1000
		row = "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (rid, storage, mtype, mlen, sinfo["num_errors"],
			sinfo["actual_fraction_error"], sinfo["probability_of_error"], sinfo["total_storage_required"])
		return row

	def generate_table(self):
		file_name = self.get_file_name()
		fp = open(file_name,'w')
		fp.write("rid\tstorage\tmtype\tmem_len\terror_count\tfraction_error\tprobability_of_error\ttotal_storage_required\n")
		print("storage\tbind_len\tsdm_len\tparameters")
		sid = 0
		for storage in range(self.storage_min, self.storage_max+1, self.storage_step):
			sid += 1
			tid = 0
			for trial in range(self.pvals["num_trials"]):
				tid += 1
				bind_l = self.bind_len(storage)
				sdm_num_rows = self.sdm_len(storage)
				sdm_activation_count = round(sdm_num_rows / 100)
				parameters = "-b %s -m %s -a %s" % (bind_l, sdm_num_rows, sdm_activation_count)
				self.pvals["num_rows"] = sdm_num_rows
				self.pvals["activation_count"] = sdm_activation_count
				print("%s\t%s\t%s\t%s" % (storage, bind_l, sdm_num_rows, parameters))
				fbs = FSA_bind_store(self.fsa, bind_l, self.pvals["debug"], self.pvals)
				rid = "%s.%s" % (sid, tid)
				fp.write(self.format_info(rid, storage, "bind", bind_l, fbs.sinfo))
				fss = FSA_sdm_store(self.fsa, self.pvals["sdm_word_length"], self.pvals["debug"], self.pvals)
				fp.write(self.format_info(rid, storage, "sdm", sdm_num_rows, fss.sinfo))
		fp.close()




	def bind_len(self, storage):
		# given storage in bytes, compute vector length to use that storage
		length = round((storage * 8) / (self.num_items + 1))
		return length

	def sdm_len(self, storage):
		# given storage in bytes, compute # rows in SDM memory to use that storage
		sdm_word_length = self.pvals["sdm_word_length"] # num bites in sdm address and memory
		length = round((storage - (self.num_items * sdm_word_length / 8)) / sdm_word_length)
		return length


def main():
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
	if env.pvals["generate_table"] == 1:
		Table_Generator(env.pvals, fsa)
		return
	FSA_bind_store(fsa, env.pvals["bind_word_length"], env.pvals["debug"], env.pvals)
	if env.pvals["sdm_method"] in (0, 2):
		FSA_sdm_store(fsa, env.pvals["sdm_word_length"], env.pvals["debug"], env.pvals)
	if env.pvals["sdm_method"] in (1, 2):
		FSA_combo_store(fsa, env.pvals["sdm_word_length"], env.pvals["debug"], env.pvals)

main()
# Bundle.test()
# Sdm.test()


