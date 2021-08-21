# script to store finite-state automaton using high-dimensional vectors with:
# 1) bundling and 2) sparse distributed memory

import numpy as np
import argparse
import sys
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
		{ "name":"word_length", "kw":{"help":"Word length for address and memory", "type":int},
	 	  "flag":"n", "required_init":"i", "default":512 },
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
		self.fsa = FSA(self.pvals["num_states"], self.pvals["num_actions"], self.pvals["num_choices"],
			self.pvals["word_length"])
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

def make_summand(n, width):
	# n is an integer to add to counter, binary bits
	# width is the word_length (number of bits) in the integer
	# returns numpy int8 array with: where n bit is zero, -1, where n bit is one, +1
	bool_vals = list(xmpz(n).iter_bits())
	# make sure length matches width
	while len(bool_vals) < width:
		bool_vals.append(False)
	# make numpy array with all -1
	# s = np.full(width, -1, dtype=np.int8)
	s = np.where(bool_vals, 1, -1)
	return s


class Sdm:
	# implements a sparse distributed memory
	def __init__(self, address_length=128, word_length=128, num_rows=512, nact=5, debug=False):
		# nact - number of active addresses used (top matches) for reading or writing
		self.address_length = address_length
		self.word_length = word_length
		self.num_rows = num_rows
		self.nact = nact
		self.data_array = np.zeros((num_rows, word_length), dtype=np.int8)
		self.addresses = initialize_binary_matrix(num_rows, word_length)
		self.debug = debug
		self.hits = np.zeros((num_rows,), dtype=np.int32)

	def store(self, address, data):
		# store binary word data at top nact addresses matching address
		top_matches = find_matches(self.addresses, address, self.nact, index_only = True)
		d = data.copy()
		d[d==0] = -1  # replace zeros in data with -1
		for i in top_matches:
			self.data_array[i] += d
			self.hits[i] += 1
		if self.debug:
			print("store\n addr=%s\n data=%s" % (bina2str(address), bina2str(data)))

	def show_hits(self):
		# display histogram of overlapping hits
		values, counts = np.unique(self.hits, return_counts=True)
		vc = [(values[i], counts[i]) for i in range(len(values))]
		vc.sort(key = lambda y: (y[0], y[1]))
		print("hits - counts:")
		pp.pprint(vc)

	def read(self, address, match_bits=None):
		top_matches = find_matches(self.addresses, address, self.nact, index_only = True, match_bits=match_bits)
		i = top_matches[0]
		sum = np.int32(self.data_array[i].copy())  # np.int32 is to convert to int32 to have range for sum
		for i in top_matches[1:]:
			sum += self.data_array[i]
		sum[sum<1] = 0   # replace values less than 1 with zero
		sum[sum>0] = 1   # replace values greater than 0 with 1
		if self.debug:
			print("read\n addr=%s\n top_matches=%s\n data=%s" % (bina2str(address), top_matches, bina2str(sum)))
		return sum

	def clear(self):
		# set data_array contents to zero
		self.data_array.fill(0)


class FSA:
	# finite-state automaton

	def __init__(self, num_states, num_actions, num_choices, word_length, debug=False):
		# num_choices is number of actions per state
		self.num_states = num_states
		self.states_im = initialize_binary_matrix(num_states, word_length, debug=False)
		self.num_actions = num_actions
		self.actions_im = initialize_binary_matrix(num_actions, word_length, debug=False)
		self.word_length = word_length
		self.num_choices = num_choices
		fsa = []
		for i in range(num_states):
			fsa.append( [ (randrange(num_actions), randrange(num_states)) for j in range(num_choices)] )
		self.fsa = fsa
		self.save_using_bundle()


	def display(self):
		# display the fsa
		for state_num in range(self.num_states):
			state_name = "s%s" % state_num
			ns = self.fsa[state_num]
			next_states = ', '.join(["a%s -> s%s" % (ns[i][0], ns[i][1]) for i in range(len(ns))])
			print("%s: %s" % (state_name, next_states))

	def save_using_bundle(self):
		# save the fsa using bundling in a single vector
		# do this by adding state_v XOR action_v XOR Right_shift(next_state_v)
		# 
		bundle = np.zeros((self.word_length, ), dtype=np.int16)
		for state_num in range(self.num_states):
			state_v = self.states_im[state_num]
			action_next_state_list = self.fsa[state_num]
			for action_nexts in action_next_state_list:
				action_num, next_state_num = action_nexts
				action_v = self.actions_im[action_num]
				next_state_v = self.states_im[next_state_num]
				add_v = state_v ^ action_v ^ rotate_right(next_state_v, self.word_length)
				d = make_summand(add_v, self.word_length)
				bundle += d
		# convert from counter to binary, then to 
		bundle = np.where(bundle>0, 1, 0)
		bpack = np.packbits(bundle, axis=-1)
		intval = int.from_bytes(bpack.tobytes(), byteorder)
		self.bundle = intval


	def recall_using_bundle(self):
		# recall each action from the fsa bundle vector
		# do this by finding best match for left_rotate(state_v XOR action_v)
		#
		num_errors = 0
		nret = 3
		sum_hdiff = 0.0
		item_count = 0
		for state_num in range(self.num_states):
			state_v = self.states_im[state_num]
			action_next_state_list = self.fsa[state_num]
			for action_nexts in action_next_state_list:
				item_count += 1
				action_num, next_state_num = action_nexts
				action_v = self.actions_im[action_num]
				# next_state_v = self.states_im[next_state_num]
				found_v = rotate_left(state_v ^ action_v ^ self.bundle, self.word_length)
				im_matches = find_matches(self.actions_im, found_v, nret)
				found_i = im_matches[0][0]
				if found_i != next_state_num:
					num_errors += 1
				sum_hdiff += abs(im_matches[0][1] - im_matches[1][1])
		print("num_errors=%s, ave_hdiff=%0.4f" % (num_errors, sum_hdiff / item_count) )

def main():
	env = Env()
	env.fsa.display()
	env.fsa.recall_using_bundle()

main()

