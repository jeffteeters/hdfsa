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
	 	{ "name":"num_trials", "kw":{"help":"Number of trials to run", "type":int},
	 	  "flag":"t", "required_init":"i", "default":3 }, 
		{ "name":"sdm_word_length", "kw":{"help":"Word length for SDM memory", "type":int},
	 	  "flag":"w", "required_init":"i", "default":512 },
	 	{ "name":"bind_word_length", "kw":{"help":"Word length for binding memory", "type":int},
	 	  "flag":"b", "required_init":"i", "default":512 },
	 	{ "name":"num_rows", "kw":{"help":"Number rows in memory","type":int},
	 	  "flag":"m", "required_init":"i", "default":2048 },
		{ "name":"activation_count", "kw":{"help":"Number memory rows to activate for each address","type":int},
		  "flag":"a", "required_init":"m", "default":20},
		{ "name":"debug", "kw":{"help":"Debug mode","type":int, "choices":[0, 1]},
		  "flag":"d", "required_init":"", "default":0},
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
		# self.fsa = FSA(self.pvals["num_states"], self.pvals["num_actions"], self.pvals["num_choices"])
		# self.fsa_bind_store = FSA_Bind_store(self.fsa, self.pvals["bind_word_length"], self.pvals["debug"])
		# self.fsa_sdm_store = FSA_SDM_store(self.fsa, self.pvals["sdm_word_length"], self.pvals["debug"])
		# self.sdm = SDM(self)

	def display_settings(self):
		print("Current settings:")
		for p in self.parms:
			print(" %s: %s" % (p["name"], self.pvals[p["name"]]))

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


class Sdm:
	# implements a sparse distributed memory
	def __init__(self, address_length=128, word_length=128, num_rows=512, nact=5, debug=False):
		# nact - number of active addresses used (top matches) for reading or writing
		self.address_length = address_length
		self.word_length = word_length
		self.num_rows = num_rows
		self.nact = nact
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

	def __init__(self, word_length, debug=False):
		self.word_length = word_length
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
			next_states = ', '.join(["a%s -> s%s" % (ns[i][0], ns[i][1]) for i in range(len(ns))])
			print("%s: %s" % (state_name, next_states))

	def initialize_item_memory(self, word_length):
		self.states_im = initialize_binary_matrix(self.num_states, word_length, self.debug)
		self.actions_im = initialize_binary_matrix(self.num_actions, word_length, self.debug)


class FSA_store:
	# abstract class for storing FSA.  Must subclass to implement different storage methods

	def __init__(self, fsa, word_length, debug, pvals=None):
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
					print("error, expected state=s%s, found_state=s%s, found_hdif=%s, hdif_dif=%s" % (next_state_num, 
						found_i, gmpy2.popcount(self.fsa.states_im[found_i] ^ found_v), hdiff_dif))
					num_errors += 1
				hdiffs.append(hdiff_dif)
		print("num_errors=%s/%s, hdiff avg=%0.4f, std=%0.4f" % (num_errors, item_count,
			statistics.mean(hdiffs), statistics.stdev(hdiffs)))

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


class FSA_bind_store(FSA_store):
	# store FSA using binding into a single vector

	def initialize(self):
		self.bundle = Bundle(self.word_length)

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

class FSA_sdm_store(FSA_store):
	# store FSA using SDM (sparse distributed memory)

	def initialize(self):
		self.sdm = Sdm(address_length=self.word_length, word_length=self.word_length,
			num_rows=self.pvals["num_rows"], nact=self.pvals["activation_count"], debug=self.debug)

	def save_transition(self, state_v, action_v, next_state_v):
		self.sdm.store(state_v ^ action_v, next_state_v)

	def recall_transition(self, state_v, action_v):
		# recall next_state_v from state_v and action_v
		return self.sdm.read(state_v ^ action_v)


def main():
	env = Env()
	fsa = FSA(env.pvals["num_states"], env.pvals["num_actions"], env.pvals["num_choices"])
	fsa.display()
	FSA_bind_store(fsa, env.pvals["bind_word_length"], env.pvals["debug"])
	FSA_sdm_store(fsa, env.pvals["sdm_word_length"], env.pvals["debug"], env.pvals)

main()
# Bundle.test()
# Sdm.test()


