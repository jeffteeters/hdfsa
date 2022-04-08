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
import pprint
pp = pprint.PrettyPrinter(indent=4)

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

class Memory:
	# superclass for SDM and bundle

	def nothing():
		return 1



class Sparse_distributed_memory(Memory):
	# implements a sparse distributed memory

	def __init__(self, word_length, nrows, nact, bc):
		# nact - number of active addresses used (top matches) for reading or writing
		# bc - number of bits to use in each counter after finalizing counter matrix
		self.word_length = word_length
		self.nrows = nrows
		self.nact = nact
		self.bc = bc

	def initialize(self):
		self.data_array = np.zeros((self.nrows, self.word_length), dtype=np.int16)
		self.addresses = np.random.randint(0,high=2,size=(self.nrows, self.word_length), dtype=np.int8)
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
		include_zero = self.bc >= 5 or int(self.bc * 10) % 10 == 5
		magnitude = 2**int(self.bc) / 2
		self.data_array[self.data_array > magnitude] = magnitude
		self.data_array[self.data_array < -magnitude] = -magnitude
		if not include_zero:
			# replace zero counter values with random +1 or -1
			# random_plus_or_minus_one = np.random.randint(0,high=2,size=(self.num_rows, self.word_length), dtype=np.int8) * 2 -1
			# there might be a fast way to do this in numpy but I couldn't find a way
			for i in range(self.nrows):
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


class Bundle_memory(Memory):
		# implements a bundle memory
	def __init__(self, word_length):
		self.word_length = word_length

	def initialize(self):
		self.data_array = np.zeros(self.word_length, dtype=np.int16)
		self.num_items_stored = 0

	def store(self, data):
		# store data
		dpn = data * 2 - 1   # convert to +/- 1
		self.data_array += dpn
		self.num_items_stored += 1

	def truncate_counters(self):
		# Threshold counters. First force the number of items stored to be odd
		if self.num_items_stored % 2 != 1:
			# add random binary value so an odd number of items are stored
			random_plus_or_minus_one = np.random.randint(0,high=2,size=self.word_length, dtype=np.int8) * 2 -1
			self.data_array += random_plus_or_minus_one
		self.data_array = (self.data_array > 0).astype(np.int8)

	def recall(self, address):
		recalled_data = np.logical_xor(address, self.data_array).astype(np.int8)
		return recalled_data

def empirical_response(mem, actions, states, choices, size, ntrials=3000):
	# find empirical response of sdm or bundle (in mem object)
	# size is number of bytes allocated to memory, used only for including in plot titles
	using_sdm = isinstance(mem, Sparse_distributed_memory)
	trial_count = 0
	fail_count = 0
	while trial_count < ntrials:
		fsa = finite_state_automaton(actions, states, choices)
		im_actions = np.random.randint(0,high=2,size=(actions, mem.word_length), dtype=np.int8)
		im_states = np.random.randint(0,high=2,size=(states, mem.word_length), dtype=np.int8)
		mem.initialize()
		# mem = Sparse_distributed_memory(word_length, nrows, nact, bc) if using_sdm else Bundle_memory(word_length)
		# store transitions
		for state in range(states):
			for transition in fsa[state]:
				action, next_state = transition
				address = np.logical_xor(im_states[state], im_actions[action])
				data = np.logical_xor(address, np.roll(im_states[next_state], 1))
				if using_sdm:
					mem.store(address, data)
				else:
					mem.store(data)
		# recall transitions
		mem.truncate_counters()
		hamming_margins = []  # difference between match and distractor hamming distances
		# match_hammings = []
		# distractor_hammings = []
		for state in range(states):
			for transition in fsa[state]:
				action, next_state = transition
				address = np.logical_xor(im_states[state], im_actions[action])
				recalled_data = mem.recall(address)
				if using_sdm:
					recalled_next_state_vector = np.roll(np.logical_xor(address, recalled_data) , -1)
				else:
					recalled_next_state_vector = np.roll(recalled_data, -1)  # xor with address done in bundle recall
				hamming_distances = np.count_nonzero( recalled_next_state_vector!=im_states, axis=1)
				match_hamming = hamming_distances[next_state]
				found_next_state = np.argmin(hamming_distances)
				if found_next_state != next_state:
					fail_count += 1
					distractor_hamming = hamming_distances[found_next_state]
				else:
					hamming_distances[next_state] = mem.word_length
					closest_distractor = np.argmin(hamming_distances)
					distractor_hamming = hamming_distances[closest_distractor]
				hamming_margins.append(distractor_hamming - match_hamming)
				trial_count += 1
	error_rate = fail_count / trial_count
	# mm = statistics.mean(match_hammings)  # match mean
	# mv = statistics.variance(match_hammings)
	# dm = statistics.mean(distractor_hammings)  # distractor mean
	# dv = statistics.variance(distractor_hammings)
	# cm = dm - mm  # combined mean
	# cs = math.sqrt(mv + dv)  # combined standard deviation
	# predicted_error_rate = norm.cdf(0, loc=cm, scale=cs)
	margin_mean = statistics.mean(hamming_margins)
	margin_var = statistics.variance(hamming_margins)
	predicted_error_rate = norm.cdf(0, loc=margin_mean, scale=math.sqrt(margin_var))
	info = {"error_rate": error_rate, "predicted_error_rate":predicted_error_rate,}
		# "mm":mm, "mv":mv, "dm":dm, "dv":dv, "cm":cm, "cs":cs}
	# plot_hist(match_hammings, distractor_hammings, "bc=%s, nact=%s, size=%s" % (bc, nact, size))
	return info


def plot_hist(match_hammings, distractor_hammings, title):
	# plot histograms of match mean and distractor hammings
	bin_start = min(min(match_hammings), min(distractor_hammings))
	bin_end = max(max(match_hammings), max(distractor_hammings))
	bins = np.arange(bin_start, bin_end+1, 1)
	plt.hist(match_hammings, bins, alpha=0.5, label='match_hammings')
	plt.hist(distractor_hammings, bins, alpha=0.5, label='distractor_hammings')
	plt.legend(loc='upper right')
	plt.title(title)
	plt.show()


def fraction_rows_activated(m, k):
	# compute optimal fraction rows to activate in sdm
	# m is number rows, k is number items stored in sdm
	return 1.0 / ((2*m*k)**(1/3))

def sdm_response_info(size, bc=8, nact=None, word_length=512, actions=10, states=100, choices=10, fimp=1.0,
		bc_for_rows=None, empirical=True):
	# compute sdm recall error for random finite state automata
	# size - number of bytes total storage allocated to sdm and item memory
	# bc - number of bits in each counter after counter finalized.  Has 0.5 added to include zero if less than 5
	# nact - sdm activaction count.  If none, calculate using optimal formula
	# fimp - fraction of item memory and hard location addresses present (1.0 all present, 0- all generated dynamically)
	# size - word_length - width (in bits) of address and counter matrix and item memory
	# actions - number of actions in finite state automaton
	# states - number of states in finite state automation
	# choices - number of choices per state
	# bc_for_rows - bc size used to calculate number of rows.  Used to compare effects of bc with same number rows
	# empirical - True if should get empirical response, False otherwise
	item_memory_size = ((actions + states) * word_length / 8) * fimp  # size in bytes of item memory
	if bc_for_rows is None:
		bc_for_rows = bc
	bcu = int(bc_for_rows + .6)  # round up
	size_one_row = (word_length / 8) * fimp + (word_length * bcu/8)  # size one address and one counter row
	nrows = int((size - item_memory_size) / size_one_row)
	number_items_stored = actions * states
	if nact is None:
		nact = round(fraction_rows_activated(nrows, number_items_stored)*nrows)
		if nact < 1:
			nact = 1
	mem = Sparse_distributed_memory(word_length, nrows, nact, bc)
	ri = empirical_response(mem, actions, states, choices, size=size) if empirical else {"error_rate": None}
	byte_operations_required_for_recall = (word_length / 8) * states + (word_length / 8) * nrows + nact * (word_length / 8)
	parallel_operations_required_for_recall = (word_length / 8) + (word_length / 8) + nact * (word_length / 8)
	fraction_memory_used_for_data = (word_length*nrows*bcu) / (size * 8) 
	info={"err":ri["error_rate"], "predicted_error":ri["predicted_error_rate"],
		"nrows":nrows, "nact":nact,
		"word_length":word_length,
		"mem_eff":fraction_memory_used_for_data,
		"recall_ops": byte_operations_required_for_recall,
		"recall_pops": parallel_operations_required_for_recall,
		# "mm":ri["mm"], "ms":math.sqrt(ri["mv"]), "dm":ri["dm"], "ds":math.sqrt(ri["dv"]),
		# "cm":ri["cm"], "cs":ri["cs"]
		}
	return info

def bundle_response_info(size, actions=10, states=100, choices=10, fimp=1.0, empirical=True):
	# compute bundle recall error for random finite state automata
	# size - number of bytes total storage allocated to bundle and item memory
	# fimp - fraction of item memory and hard location addresses present (1.0 all present, 0- all generated dynamically)
	# actions - number of actions in finite state automaton
	# states - number of states in finite state automation
	# choices - number of choices per state
	# empirical - True if should get empirical response, False otherwise
	item_memory_length = actions + states
	word_length = int((size * 8) / (1 + item_memory_length * fimp))
	mem = Bundle_memory(word_length)
	ri = empirical_response(mem, actions, states, choices, size=size) if empirical else {"error_rate": None}
	fraction_memory_used_for_data = word_length / (size * 8)
	byte_operations_required_for_recall = (word_length / 8) * states
	parallel_operations_required_for_recall = word_length / 8
	info={"err":ri["error_rate"], "predicted_error":ri["predicted_error_rate"],
		"mem_eff":fraction_memory_used_for_data,
		"bundle_length":word_length,
		"recall_ops": byte_operations_required_for_recall,
		"recall_pops": parallel_operations_required_for_recall,
		# "mm":ri["mm"], "ms":math.sqrt(ri["mv"]), "dm":ri["dm"], "ds":math.sqrt(ri["dv"]),
		# "cm":ri["cm"], "cs":ri["cs"]
		}
	return info

def plot_info(sizes, bc_vals, resp_info, line_label):
	plots_info = [
		{"subplot": 221, "key":"err","title":"SDM error with different counter bits", "ylabel":"Recall error"},
		{"subplot": 222, "key":"err","title":"SDM error with different counter bits", "ylabel":"Recall error", "scale":"log"},
		# {"subplot": 222, "key":"predicted_error","title":"SDM predicted error with different counter bits",
		#	"ylabel":"Recall error", "scale":"log"},
		{"subplot": 223, "key":"nrows","title":"Number rows in SDM vs. size and counter bits","ylabel":"Number rows"},
		{"subplot": 224, "key":"nact","title":"SDM activation count vs counter bits and size","ylabel":"Activation Count"},
		# {"subplot": 221, "key":"mm","title":"Match mean","ylabel":"Hamming distance"},
		# {"subplot": 222, "key":"ms","title":"Match std","ylabel":"Hamming distance"},
		# {"subplot": 223, "key":"dm","title":"Distractor mean","ylabel":"Hamming distance"},
		# {"subplot": 224, "key":"ds","title":"Distractor std","ylabel":"Hamming distance"},
		 ]
	for pi in plots_info:
		plt.subplot(pi["subplot"])
		log_scale = "scale" in pi and pi["scale"] == "log"
		yvals = [resp_info[bc_vals[0]][i][pi["key"]] for i in range(len(sizes))]
		plt.errorbar(sizes, yvals, yerr=None, label="%s %s" % (bc_vals[0], line_label)) # fmt="-k"
		for i in range(1, len(bc_vals)):
			yvals = [resp_info[bc_vals[i]][j][pi["key"]] for j in range(len(sizes))]
			plt.errorbar(sizes, yvals, yerr=None, label="%s %s" % (bc_vals[i], line_label), linewidth=1,)# fmt="-k",) # linestyle='dashed',
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
		if pi["subplot"] == 224:
			plt.show()
	return


def vary_sdm_bc(start_size=10000, step_size=2000, stop_size=30001, bc_vals=[1,1.5, 2, 2.5,3.5,4.5,5.5,8]):
	bc_vals = [1,1.5, 2, 2.5, 3, 3.5 , 8]
	start_size=20000; step_size=1000; stop_size=33001
	resp_info = {}
	sizes = range(start_size, stop_size, step_size)
	bc_for_rows=None
	nact=None
	for bc in bc_vals:
		resp_info[bc] = [sdm_response_info(size, bc, nact=nact, bc_for_rows=bc_for_rows) for size in sizes]
	# make plot
	plot_info(sizes, bc_vals, resp_info, line_label="bit")

def vary_nact(start_size=10000, step_size=2000, stop_size=30001, nact_vals=[1,2, 3, 4, 5]):
	resp_info = {}
	start_size=20000; step_size=1000; stop_size=33001
	sizes = range(start_size, stop_size, step_size)
	bc = 8  # used fixed bc
	bc_for_rows=1
	for nact in nact_vals:
		resp_info[nact] = [sdm_response_info(size, bc, nact=nact, bc_for_rows=bc_for_rows) for size in sizes]
	# make plot
	plot_info(sizes, nact_vals, resp_info, line_label="act")

def vary_fimp(start_size=10000, step_size=2000, stop_size=30001, nact_vals=[1,3]):
	resp_info = {}
	start_size=18000; step_size=500; stop_size=24001
	sizes = range(start_size, stop_size, step_size)
	bc = 1  # used fixed bc
	for nact in nact_vals:
		resp_info[nact] = [sdm_response_info(size, bc, nact=nact) for size in sizes]
	# make plot
	plot_info(sizes, nact_vals, resp_info, line_label="act")

def folds2fimp(folds):
	# convert number of folds to fraction of item memory present
	return 1.0 / folds



def sdm_vs_bundle():
	# start_size=18000; step_size=500; stop_size=24001
	# start_size=5000; step_size=1000; stop_size=14001
	start_size=100000; step_size=100000; stop_size=1000001
	sizes = range(start_size, stop_size, step_size)
	# bc = 5.5  # used fixed bc
	bc = 8
	nact = None
	fimp=1
	sdm_ri = [sdm_response_info(size, bc, nact=nact, fimp=fimp) for size in sizes]  # ri - response info
	bundle_ri = [bundle_response_info(size, fimp=fimp) for size in sizes]
	# bundle_ri = [{} for size in sizes]
	plots_info = [
		{"subplot": 221, "key":"err","title":"SDM vs bundle error with fimp=%s" % fimp, "ylabel":"Recall error"},
		{"subplot": None, "key":"predicted_error","title":"SDM vs bundle error with fimp=%s" % fimp, "ylabel":"Recall error",
			"label":"predicted "},
		{"subplot": 222, "key":"err","title":"SDM vs bundle error with fimp=%s (log scale)" % fimp,
			"ylabel":"Recall error", "scale":"log"},
		{"subplot": None, "key":"predicted_error","title":"SDM vs bundle error with fimp=%s" % fimp, "ylabel":"Recall error",
			"label":"predicted ", "scale":"log", "legend_location":"lower left"},
		# {"subplot": 222, "key":"mem_eff","title":"SDM vs bundle mem_eff with fimp=%s" % fimp, "ylabel":"Fraction mem used"},
		{"subplot": 223, "key":"nrows","title":"SDM num rows with fimp=%s" % fimp, "ylabel":"Number rows"},
		{"subplot": 224, "key":"bundle_length","title":"bundle_length with fimp=%s" % fimp, "ylabel":"Bundle length"},
		# {"subplot": 224, "key":"nact","title":"sdm activation with fimp=%s" % fimp, "ylabel":"activaction count"},
		]
	for pidx in range(len(plots_info)):
		pi = plots_info[pidx]
		new_plot = pi["subplot"] is not None
		finishing_plot = pidx == len(plots_info)-1 or plots_info[pidx+1]["subplot"] is not None
		if new_plot:
			plt.subplot(pi["subplot"])
		label = pi["label"] if "label" in pi else ""
		log_scale = "scale" in pi and pi["scale"] == "log"
		if pi["key"] in sdm_ri[0]:
			yvals = [sdm_ri[i][pi["key"]] for i in range(len(sizes))]
			plt.errorbar(sizes, yvals, yerr=None, label="%ssdm" % label)
		if pi["key"] in bundle_ri[0]:
			yvals = [bundle_ri[i][pi["key"]] for i in range(len(sizes))]
			plt.errorbar(sizes, yvals, yerr=None, label="%sbundle" % label)
		if finishing_plot:
			xaxis_labels = ["%s" % int(size/1000) for size in sizes]
			plt.xticks(sizes,xaxis_labels)
			if log_scale:
				plt.yscale('log')
			plt.title(pi["title"])
			plt.xlabel("Size (kB)")
			plt.ylabel(pi["ylabel"])
			legend_location = pi["legend_location"] if "legend_location" in pi else "upper right"
			plt.legend(loc=legend_location)
			plt.grid()
	plt.show()


def widths_vs_folds():
	# display plot of SDM length (number of rows) and bundle word length for different number of folds and size
	# memory
	start_size=100000; step_size=100000; stop_size=1000001
	sizes = range(start_size, stop_size, step_size)
	folds = [1, 2, 4, 8, 16, 32, 64, 128, "inf"]
	fimps = [ 1 / f for f in folds[0:-1]] + [0.0]
	sdm_ri = []
	bun_ri = []
	for size in sizes:
		bc = 8
		sdm_ri.append( [sdm_response_info(size, bc, fimp=fimp, empirical=False) for fimp in fimps] ) # ri - response info
		bun_ri.append( [bundle_response_info(size, fimp=fimp, empirical=False) for fimp in fimps] )
	# make plots
	plots_info = [
		{"subplot": 221, "key":"nrows","title":"SDM num rows vs num folds", "ylabel":"Number rows"},
		{"subplot": 222, "key":"bundle_length","title":"bundle_length vs num folds", "ylabel":"Bundle length"},
		{"subplot": 223, "key":"recall_ops","title":"Recall byte operations vs num folds", "ylabel":"Byte operations",
			"scale": "log"},
		{"subplot": 224, "key":"recall_pops","title":"Parallel recall byte operations vs num folds", "ylabel":"Parallel operations",
			"scale": "log"},
		]
	for pi in plots_info:
		plt.subplot(pi["subplot"])
		log_scale = "scale" in pi and pi["scale"] == "log"
		need_mem_label = pi["key"] in sdm_ri[0][0] and pi["key"] in bun_ri[0][0]
		if pi["key"] in sdm_ri[0][0]:
			yvals = [sdm_ri[i][0][pi["key"]] for i in range(len(sizes))]
			mem_label = "sdm " if need_mem_label else ""
			plt.errorbar(sizes, yvals, yerr=None, label="%s%s" % (mem_label, folds[0])) # fmt="-k"
			for j in range(1, len(folds)):
				yvals = [sdm_ri[i][j][pi["key"]] for i in range(len(sizes))]
				plt.errorbar(sizes, yvals, yerr=None, label="%s%s" % (mem_label, folds[j]), linewidth=1,)# fmt="-k",) # linestyle='dashed',
			labelLines(plt.gca().get_lines(), zorder=2.5)
		if pi["key"] in bun_ri[0][0]:
			mem_label = "bun " if need_mem_label else ""
			yvals = [bun_ri[i][0][pi["key"]] for i in range(len(sizes))]
			plt.errorbar(sizes, yvals, yerr=None, label="%s%s" % (mem_label, folds[0])) # fmt="-k"
			for j in range(1, len(folds)):
				yvals = [bun_ri[i][j][pi["key"]] for i in range(len(sizes))]
				plt.errorbar(sizes, yvals, yerr=None, label="%s%s" % (mem_label, folds[j]), linewidth=1,)# fmt="-k",) # linestyle='dashed',
			labelLines(plt.gca().get_lines(), zorder=2.5)
		xaxis_labels = ["%s" % int(size/1000) for size in sizes]
		plt.xticks(sizes,xaxis_labels)
		if log_scale:
			plt.yscale('log')
		plt.title(pi["title"])
		plt.xlabel("Size (kB)")
		plt.ylabel(pi["ylabel"])
		# plt.legend(loc='upper left')
		plt.grid()
	plt.show()

def widths_vs_folds_single_size(size=100000, empirical=True):
	# display plot of SDM length (number of rows) and bundle word length for different number of folds at a fixed size
	# memory
	# empirical == True to include empirical recall error
	folds = [1, 2, 4, 8, 16, 32, 64, 128, "inf"]
	fimps = [ 1 / f for f in folds[0:-1]] + [0.0]
	# folds = [1, 2, 4, 8, 16, 32, 64] # 8, 16, 32, 64]
	# fimps = [ 1 / f for f in folds]
	bc = 1
	sdm_ri = [sdm_response_info(size, bc, fimp=fimp, empirical=empirical) for fimp in fimps]  # ri - response info
	bun_ri = [bundle_response_info(size, fimp=fimp, empirical=empirical) for fimp in fimps]
	# compute ratio, add to sdm_ri
	for i in range(len(fimps)):
		sdm_ri[i]["bun/sdm_pops"] = bun_ri[i]["recall_pops"] / sdm_ri[i]["recall_pops"] 
	# print("sdm_ri=")
	# pp.pprint(sdm_ri)
	# print("bun_ri=")
	# pp.pprint(bun_ri)
	# make plots
	plots_info = [
		{"subplot": 221, "key":"nrows","title":"SDM num rows vs num folds for size=%s" % size, "ylabel":"Number rows"},
		{"subplot": 222, "key":"bundle_length","title":"bundle_length vs num folds for size=%s" % size, "ylabel":"Bundle length"},
		{"subplot": 223, "key":"recall_ops","title":"Recall byte operations vs num folds", "ylabel":"Byte operations",
			"scale": "log"},
		{"subplot": 224, "key":"recall_pops","title":"Parallel recall byte operations vs num folds", "ylabel":"Parallel operations",
			"scale": "log"},
		{"subplot": 121, "key":"bun/sdm_pops", "title":"Ratio of bundle/sdm recall byte operations vs num folds", "ylabel":"Parallel operations",},
		{"subplot": 122, "key":"bun/sdm_pops", "title":"Ratio of bundle/sdm recall byte operations vs num folds (log scale)", "ylabel":"Parallel operations",
			"scale": "log"},
		{"subplot": 221, "key":"err","title":"SDM vs bundle error vs num folds for size=%s" % size, "ylabel":"Recall error"}, # "scale":"log"},
		{"subplot": 222, "key":"err","title":"SDM vs bundle error vs num folds for size=%s (log scale)" % size, "ylabel":"Recall error", "scale":"log"},
		{"subplot": 223, "key":"predicted_error","title":"SDM vs bundle predicted error vs num folds for size=%s" % size, "ylabel":"Recall error"}, # "scale":"log"},
		{"subplot": 224, "key":"predicted_error","title":"SDM vs bundle predicted error vs num folds for size=%s (log scale)" % size, "ylabel":"Recall error", "scale":"log"},

		]
	for pi in plots_info:
		plt.subplot(pi["subplot"])
		log_scale = "scale" in pi and pi["scale"] == "log"
		need_mem_label = pi["key"] in sdm_ri[0] and pi["key"] in bun_ri[0]
		xpos = list(range(len(folds)))
		if pi["key"] in sdm_ri[0]:
			yvals = [sdm_ri[i][pi["key"]] for i in range(len(folds))]
			mem_label = "sdm " if need_mem_label else ""
			plt.errorbar(xpos, yvals, yerr=None, label="%s%s" % (mem_label, pi["key"])) # fmt="-k"
			# yvals = [sdm_ri[i][pi["key"]] for i in range(len(folds))]
			# plt.errorbar(sizes, yvals, yerr=None, label="%s%s" % (mem_label, folds[j]), linewidth=1,)# fmt="-k",) # linestyle='dashed',
			# labelLines(plt.gca().get_lines(), zorder=2.5)
		if pi["key"] in bun_ri[0]:
			mem_label = "bun " if need_mem_label else ""
			yvals = [bun_ri[i][pi["key"]] for i in range(len(folds))]
			plt.errorbar(xpos, yvals, yerr=None, label="%s%s" % (mem_label, pi["key"])) 
		xaxis_labels = ["%s" % f for f in folds]
		plt.xticks(xpos,xaxis_labels)
		if log_scale:
			plt.yscale('log')
		plt.title(pi["title"])
		plt.xlabel("Folds")
		plt.ylabel(pi["ylabel"])
		plt.legend(loc='upper left')
		plt.grid()
		if pi["subplot"] in (224, 122):
			plt.show()
	# plt.show()


if __name__ == "__main__":
	# vary_sdm_bc()
	# vary_nact()
	sdm_vs_bundle()
	# widths_vs_folds()
	# widths_vs_folds_single_size()