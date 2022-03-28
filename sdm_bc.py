# script to generate plot of SDM recall performance vs bit counter widths
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from labellines import labelLine, labelLines
import sys
import random
import statistics
from scipy.stats import norm
import math

def pop_random_element(L):
	# pop random element from a list
	# from https://stackoverflow.com/questions/10048069/what-is-the-most-pythonic-way-to-pop-a-random-element-from-a-list
	i = random.randrange(len(L)) # get random index
	L[i], L[-1] = L[-1], L[i]    # swap with the last element
	x = L.pop()                  # pop last element O(1)
	return x

def finite_state_automaton(num_actions, num_states, num_choices):
	# return array of transitions [(action, next_state), (action, next_state), ...] for each state 
	assert num_choices <= num_actions
	fsa = []
	for i in range(num_states):
		possible_actions = list(range(num_actions))
		possible_next_states = list(range(num_states))
		nas = []
		for j in range(num_choices):
			nas.append( ( pop_random_element(possible_actions), pop_random_element(possible_next_states) ) )
		fsa.append(nas)
	return fsa

class Sparse_distributed_memory:
	# implements a sparse distributed memory

	def __init__(self, word_length, nrows, nact, bc):
		# nact - number of active addresses used (top matches) for reading or writing
		# bc - number of bits to use in each counter after finalizing counter matrix
		self.word_length = word_length
		self.nrows = nrows
		self.nact = nact
		self.bc = bc
		self.data_array = np.zeros((nrows, word_length), dtype=np.int16)
		self.addresses = np.random.randint(0,high=2,size=(nrows, word_length), dtype=np.int8)
		self.select_cache = {}

	def select(self, address):
		# return index of rows with hard locations (addresses) most closly matching address
		hl_match = np.count_nonzero( address!=self.addresses, axis=1)
		row_ids = np.argpartition(hl_match, self.nact)[0:self.nact]
		# following code to make sure same rows are selected for the same address (safety check)
		addr_str = ''.join(map('{:b}'.format, address))
		if addr_str in self.select_cache:
			assert np.array_equal(row_ids, self.select_cache[addr_str])
		else:
			self.select_cache[addr_str] = row_ids
		return row_ids

	def store(self, address, data):
		# store binary word data at top nact addresses matching address
		row_ids = self.select(address)
		dpn = data * 2 - 1   # convert to +/- 1
		for i in row_ids:
			self.data_array[i] += dpn

	def truncate_counters(self):
		# truncate counters to number of bits in bc.  if bc < 5, zero is included in the range only if
		# bc has a 0.5 fractional part.  if bc >= 5, always include zero in the range.
		include_zero = self.bc >= 0 or int(self.bc * 10) % 10 == 5
		magnitute = 2**int(bc) / 2
		self.data_array[self.data_array > self.counter_magnitude] = self.counter_magnitude
		self.data_array[self.data_array < -self.counter_magnitude] = -self.counter_magnitude
		if not include_zero:
			# replace zero counter values with random +1 or -1
			# random_plus_or_minus_one = np.random.randint(0,high=2,size=(self.num_rows, self.word_length), dtype=np.int8) * 2 -1
			# there might be a fast way to do this in numpy but I couldn't find a way
			for i in range(self.num_rows):
				for j in range(self.word_length):
					if self.data_array[i,j] == 0:
						self.data_array[i,j] = random.choice((-1, +1))

	def recall(self, address):
		row_ids = self.select(address)
		isum = np.int32(self.data_array[row_ids[0]].copy())  # np.int32 is to convert to int32 to have range for sum
		for i in row_ids[1:]:
			isum += self.data_array[i]
		recalled_data = np.int8(isum > 0)
		return recalled_data

def sdm_empirical_response(bc, word_length, actions, states, choices, nrows, nact, ntrials=12000):
	# find empirical response of sdm
	trial_count = 0
	fail_count = 0
	while trial_count < ntrials:
		fsa = finite_state_automaton(actions, states, choices)
		im_actions = np.random.randint(0,high=2,size=(actions, word_length), dtype=np.int8)
		im_states = np.random.randint(0,high=2,size=(states, word_length), dtype=np.int8)
		sdm = Sparse_distributed_memory(word_length, nrows, nact, bc)
		# store transitions
		for state in range(states):
			for transition in fsa[state]:
				action, next_state = transition
				address = np.logical_xor(im_states[state], im_actions[action])
				data = np.logical_xor(address, np.roll(im_states[next_state], 1))
				sdm.store(address, data)
		# recall transitions
		match_hammings = []
		distractor_hammings = []
		for state in range(states):
			for transition in fsa[state]:
				action, next_state = transition
				address = np.logical_xor(im_states[state], im_actions[action])
				recalled_data = sdm.recall(address)
				recalled_next_state_vector = np.roll(np.logical_xor(address, recalled_data) , -1)
				hamming_distances = np.count_nonzero( recalled_next_state_vector!=im_states, axis=1)
				match_hammings.append(hamming_distances[next_state])
				found_next_state = np.argmin(hamming_distances)
				if found_next_state != next_state:
					fail_count += 1
					distractor_hammings.append(hamming_distances[found_next_state])
				else:
					hamming_distances[next_state] = word_length
					closest_distractor = np.argmin(hamming_distances)
					distractor_hammings.append(hamming_distances[closest_distractor])
				trial_count += 1
	error_rate = fail_count / trial_count
	mm = statistics.mean(match_hammings)  # match mean
	mv = statistics.variance(match_hammings)
	dm = statistics.mean(distractor_hammings)  # distractor mean
	dv = statistics.variance(distractor_hammings)
	cm = dm - mm  # combined mean
	cs = math.sqrt(mv + dv)  # combined standard deviation
	predicted_error_rate = norm.cdf(0, loc=cm, scale=cs)
	info = {"error_rate": error_rate, "predicted_error_rate":predicted_error_rate}
	return info


def fraction_rows_activated(m, k):
	# compute optimal fraction rows to activate in sdm
	# m is number rows, k is number items stored in sdm
	return 1.0 / ((2*m*k)**(1/3))

def sdm_response_info(size, bc, word_length=512, actions=10, states=100, choices=10, fimp=1.0):
	# compute sdm recall error for random finite state automata
	# size - number of bytes total storage allocated to sdm and item memory
	# bc - number of bits in each counter after counter finalized.  Has 0.5 added to include zero
	# fimp - fraction of item memory and hard location addresses present (1.0 all present, 0- all generated dynamically)
	# size - word_length - width (in bits) of address and counter matrix and item memory
	# actions - number of actions in finite state automaton
	# states - number of states in finite state automation
	# choices - number of choices per state
	item_memory_size = ((actions + states) * word_length / 8) * fimp  # size in bytes of item memory
	bcu = int(bc + .6)  # round up
	size_one_row = (word_length / 8) * fimp + (word_length * bcu/8)  # size one address and one counter row
	nrows = int((size - item_memory_size) / size_one_row)
	number_items_stored = actions * states
	nact = round(fraction_rows_activated(nrows, number_items_stored)*nrows)
	if nact == 0:
		nact = 1
	sdm_response_info = sdm_empirical_response(bc, word_length, actions, states, choices, nrows, nact)
	info={"err":sdm_response_info["error_rate"], "predicted_error":sdm_response_info["predicted_error_rate"],
		"nrows":nrows, "nact":nact}
	return info

def plot_info(sizes, bc_vals, resp_info):
	plots_info = [
		{"subplot": 221, "key":"err","title":"SDM error with different counter bits", "ylabel":"Recall error"},
		{"subplot": 222, "key":"predicted_error","title":"SDM predicted error with different counter bits",
			"ylabel":"Recall error", "scale":"log"},
		{"subplot": 223, "key":"nrows","title":"Number rows in SDM vs. size and counter bits","ylabel":"Number rows"},
		{"subplot": 224, "key":"nact","title":"SDM activation count vs counter bits and size","ylabel":"Activation Count"},
		 ]
	for pi in plots_info:
		plt.subplot(pi["subplot"])
		log_scale = "scale" in pi and pi["scale"] == "log"
		yvals = [resp_info[bc_vals[0]][i][pi["key"]] for i in range(len(sizes))]
		plt.errorbar(sizes, yvals, yerr=None, label="%s bit" % bc_vals[0]) # fmt="-k"
		for i in range(1, len(bc_vals)):
			yvals = [resp_info[bc_vals[i]][j][pi["key"]] for j in range(len(sizes))]
			plt.errorbar(sizes, yvals, yerr=None, label='%s bit' % bc_vals[i], linewidth=1,)# fmt="-k",) # linestyle='dashed',
		labelLines(plt.gca().get_lines(), zorder=2.5)
		if log_scale:
			plt.yscale('log')
			log_msg = " (log scale)"
			# plt.xticks([2, 3, 4, 5, 10, 20, 40, 100, 200])
			# ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
		else:
			log_msg = ""
			# plt.xticks([2, 25, 50, 75, 100, 125, 150, 175, 200])
		xaxis_labels = ["%s" % int(size/1000) for size in sizes]
		plt.xticks(sizes,xaxis_labels)		
		# xaxis_labels = ["100k", "200k", "300k", "400k", "500k", "600k", "700k", "800k", "900k", "10^6" ]
		# plt.xticks(xvals[mtype],xaxis_labels)
		plt.title(pi["title"]+log_msg)
		plt.xlabel("Size (kB)")
		plt.ylabel(pi["ylabel"])
		plt.grid()
	plt.show()
	return


def main(start_size=10000, step_size=2000, stop_size=30001, bc_vals=[1,1.5, 2, 2.5,3.5,4.5,5.5,8]):
	bc_vals = [1,1.5, 2, 2.5, 3, 3.5 , 8]
	resp_info = {}
	sizes = range(start_size, stop_size, step_size)
	for bc in bc_vals:
		resp_info[bc] = [sdm_response_info(size, bc) for size in sizes]
	# make plot
	plot_info(sizes, bc_vals, resp_info)


if __name__ == "__main__":
	main()