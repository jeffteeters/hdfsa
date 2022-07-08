import numpy as np
import matplotlib.pyplot as plt
import math
from numba import jit
from numba import int32, float32, uint32, boolean, float64    # import the types
# from numba.experimental import jitclass
import sys

spec = [
    ('nrows', int32),
    ('ncols', int32),
    ('nact', int32),
    ('actions', int32),
    ('states', int32),
    ('choices', int32),
    ('epochs', int32),
    ('count_multiple_matches_as_error', boolean),
    ('roll_address', boolean),
    ('debug', boolean),
    ('hl_selection_method_hamming', boolean),
    ('save_error_rate_vs_hamming_distance', boolean),
    ('truncate_counters', boolean),
    ('include_zero', boolean),
    ('magnitude', int32),
    ('match_hamming_counts', uint32[:]),
    ('distractor_hamming_counts', uint32[:]),
    ('ehdist', float64[:]),
    ('mean_error', float64),
    ('std_error', float64),
]

# @jitclass(spec)
class Fast_sdm_empirical():

	# calculate SDM empirical error using numpy as much as possible

	# @jit
	def __init__(self, nrows, ncols, nact, actions=10, states=100, choices=10, epochs=10,
			count_multiple_matches_as_error=True, roll_address=True, debug=False,
			hl_selection_method="hamming", bits_per_counter=8, threshold_sum=True,
			save_error_rate_vs_hamming_distance=False):
		# nrows is number of rows (hard locations) in the SDM
		# ncols is the number of columns
		# nact is activaction count
		# actions, states, choices specify finite state automata stored in sdm.  Used to determine number transitions
		# stored (k) and size of item memory (d) used to compute probability of error in recall with d-1 distractors
		# epochs - number of times to store and recall FSA from sdm.  Each epoch stores and recalls all transitions.
		# count_multiple_matches_as_error = True to count multiple distractor hammings == match as error
		# save_error_rate_vs_hamming_distance true to save error_count_vs_hamming, used to figure out what
		# happens to error rate when distractor(s) having same hamming distance as match not always counted as error
		# hl_selection_method - "hamming" to use hamming distance to address to select hard locations, other options:
		# "random" - randomly pick hard locations for each address, "mod" - evenly distribute hard locations
		# (using "mod" function)
		# bits_per_counter - number of bits counter truncated to before reading. 1 to binarize.  Has 0.5 added to
		# include zero. e.g. 1.5 means -1, 0, +1;  If greater than 4, zero is always included
		# threshold_sum set True to threshold sum to binary number before comparing to item memory (hamming distance)
		#  this is the normal SDM.  threshold_sum False means match to item memory done via dot product.
		# print("starting Fast_sdm_empirical, nrows=%s, ncols=%s, nact=%s, actions=%s, states=%s, choices=%s" % (
		# 	nrows, ncols, nact, actions, states, choices))
		self.nrows = nrows
		self.ncols = ncols
		self.nact = nact
		self.actions = actions
		self.states = states
		self.choices = choices
		self.epochs = epochs
		assert actions >= choices, "Number actions must be >= number of choices"
		self.count_multiple_matches_as_error = count_multiple_matches_as_error
		self.roll_address = roll_address
		self.debug = debug
		assert hl_selection_method in ("random", "hamming", "mod")
		self.hl_selection_method_hamming = hl_selection_method == "hamming"
		self.hl_selection_method_mod = hl_selection_method == "mod"
		self.save_error_rate_vs_hamming_distance = save_error_rate_vs_hamming_distance
		assert bits_per_counter > 0
		self.truncate_counters = bits_per_counter < 8
		if self.truncate_counters:
			# truncate counters to number of bits in bpc.  if bpc < 5, zero is included in the range only if
			# bpc has a 0.5 fractional part.  if bpc >= 5, always include zero in the range.
			self.include_zero = bits_per_counter >= 5 or int(bits_per_counter * 10) % 10 == 5
			self.magnitude = 2**int(bits_per_counter) / 2
		self.threshold_sum = threshold_sum
		# self.empiricalError()

	# def empiricalError(self):
		# compute empirical error by storing then recalling finite state automata from SDM
		fail_counts = np.zeros(self.epochs, dtype=np.uint16)
		if threshold_sum:
			distance_counts_len = self.ncols+1
			distance_counts_offset = 0
		else:
			# save counts of dot product match and distractor
			distance_counts_len = 4*(self.ncols * self.nact)  # should be enough range
			distance_counts_offset = int(distance_counts_len / 2)
		self.match_hamming_counts = np.zeros(distance_counts_len, dtype=np.uint32)
		self.distractor_hamming_counts = np.zeros(distance_counts_len, dtype=np.uint32)
		rng = np.random.default_rng()
		num_transitions = self.states * self.choices
		# match and distractor distances used for finding mean and variance of distances to match and distractor
		match_distances = np.empty(self.epochs * num_transitions, dtype=np.int16)
		distractor_distances = np.empty(self.epochs * num_transitions, dtype=np.int16)
		counter_sums = np.empty(self.epochs * num_transitions, dtype=np.int16)
		flag_value = 32000
		counter_sums[-1] = flag_value # to make sure fill all
		match_distances[-1] = flag_value 
		counter_counts_len = int(20 * ((num_transitions * self.nact) / self.nrows))
		counter_counts_offset = int(counter_counts_len / 2)
		counter_counts = np.zeros(counter_counts_len, dtype=np.uint32)  # for distribution of counter sums
		overlap_counts = np.zeros(self.nrows, dtype=np.uint32)
		trial_count = 0
		fail_count = 0
		for epoch_id in range(self.epochs):
			if self.hl_selection_method_hamming:
				# create addresses used for selecting hard locations
				im_address = rng.integers(0, high=2, size=(self.nrows, self.ncols), dtype=np.int8)
				# im_address = np.random.randint(0,high=2,size=(self.nrows, self.ncols)).astype(np.int8)
			im_action = rng.integers(0, high=2, size=(self.actions, self.ncols), dtype=np.int8)
			# im_action = np.random.randint(0,high=2,size=(self.actions, self.ncols)).astype(np.int8)
			im_state = rng.integers(0, high=2, size=(self.states, self.ncols), dtype=np.int8)
			# im_state = np.random.randint(0, high=2, size=(self.states, self.ncols)).astype(np.int8)
			transition_action = np.empty((self.states, self.choices), dtype=np.uint16)
			transition_next_state = np.empty((self.states, self.choices), dtype=np.uint16)
			for i in range(self.states):
				transition_action[i,:] = rng.choice(self.actions, size=self.choices, replace=False)
				# transition_action[i,:] = np.random.choice(self.actions, size=self.choices, replace=False)
				transition_next_state[i,:] = rng.choice(self.states, size=self.choices, replace=False)
				# transition_next_state[i,:] = np.random.choice(self.states, size=self.choices, replace=False)
			transition_state = np.repeat(np.arange(self.states), self.choices)
			transition_action = transition_action.flatten()
			transition_next_state = transition_next_state.flatten()
			address = np.logical_xor(im_state[transition_state], im_action[transition_action])
			assert transition_state.size == num_transitions
			assert transition_action.size == num_transitions
			assert transition_next_state.size == num_transitions
			transition_hard_locations = np.empty((num_transitions, self.nact), dtype=np.uint16)
			if self.hl_selection_method_hamming:
				# assert False, "hamming method not implemented"
				if self.nrows == 1:
					# special case of only one row (same as bundle)
					transition_hard_locations[:] = 0
				else:
					for i in range(num_transitions):
						hl_match = np.count_nonzero(address[i]!=im_address, axis=1)
						transition_hard_locations[i,:] = np.argpartition(hl_match, self.nact)[0:self.nact]
			elif self.hl_selection_method_mod:
				for i in range(num_transitions):
					# evenly distribute rows
					transition_hard_locations[i,:] = np.array([(i + j) % self.nrows for j in range(self.nact)])
			else:
				# use randon locations
				for i in range(num_transitions):
					transition_hard_locations[i,:] = rng.choice(self.nrows, size=self.nact, replace=False)
					# transition_hard_locations[i,:] = np.random.choice(self.nrows, size=self.nact, replace=False)
			contents = np.zeros((self.nrows, self.ncols), dtype=np.int16)
			# save FSA into SDM contents matrix
			if self.roll_address:
				# assert False, "roll_address not supported"
				# roll each row in address by state number.  This to prevent the mysterious interference
				# roll done using method in: https://stackoverflow.com/questions/20360675/roll-rows-of-a-matrix-independently
				rows, column_indices = np.ogrid[:address.shape[0], :address.shape[1]]
				column_indices = column_indices - transition_state[:, np.newaxis]
				address = address[rows, column_indices]
			data = np.logical_xor(address, np.roll(im_state[transition_next_state], 1, axis=1))*2-1
			# import pdb; pdb.set_trace()
			for i in range(num_transitions):
				contents[transition_hard_locations[i,:]] += data[i]
				overlap_counts[transition_hard_locations[i,:]] += 1
			if self.truncate_counters:
				contents[contents > self.magnitude] = self.magnitude
				contents[contents < -self.magnitude] = -self.magnitude
				if not self.include_zero:
					# replace zero counter values with random +1 or -1
					random_plus_or_minus_one = rng.integers(0, high=2, size=(self.nrows, self.ncols), dtype=np.int8)*2-1
					# random_plus_or_minus_one = np.random.randint(0, high=2, size=(self.nrows, self.ncols), dtype=np.int8)*2-1
					mask = contents == 0
					contents[mask] = random_plus_or_minus_one[mask]
			# recall data from contents matrix
			# recalled_data = np.empty((num_transitions, self.ncols), dtype=np.int8)
			if not threshold_sum:
				# not thresholding sum.  Use +1/-1 product and dot product with item memory to calculate distance
				address = address*2-1    # convert address to +1/-1
				im_state = im_state*2-1  # convert im_state to +1/-1
			for i in range(num_transitions):
				recalled_vector = contents[transition_hard_locations[i,:]].sum(axis=0)
				data_1_index = np.where(data[i] == 1)[0][0]
				one_counter_sum = recalled_vector[data_1_index]  # pick a random sum for storing, when target is 1
				# if trial_count < 10:
				# 	print("one_counter_sum=%s" % one_counter_sum)
				counter_sums[trial_count] = one_counter_sum
				counter_counts[one_counter_sum+counter_counts_offset] += 1  # save counts of counters
				if threshold_sum:
					# convert to binary
					# perhaps should deterministically add random +/- 1 before thresholding recalled vector
					recalled_vector = recalled_vector > 0
					recalled_data = np.roll(np.logical_xor(address[i], recalled_vector), -1)
					hamming_distances = np.count_nonzero(im_state[:,] != recalled_data, axis=1)
				else:
					# don't convert sum to binary.  Use dot product to find best match to item memory
					recalled_data = np.roll(address[i] * recalled_vector, -1)
					hamming_distances = np.sum(im_state[:,] * recalled_data, axis=1) # actually dot product distance
				# if trial_count < 10:
				# 	print("hamming_distances=%s" % hamming_distances[0:20])  # shows often either even or odd if ncols even
				self.match_hamming_counts[hamming_distances[transition_next_state[i]]+distance_counts_offset] += 1
				self.distractor_hamming_counts[hamming_distances[(transition_next_state[i]+1) % self.states]+distance_counts_offset] += 1 # a random distractor
				match_distances[trial_count] = hamming_distances[transition_next_state[i]]
				distractor_distances[trial_count] = hamming_distances[(transition_next_state[i]+1) % self.states]
				two_smallest = np.argpartition(hamming_distances, 2)[0:2]
				if hamming_distances[two_smallest[0]] < hamming_distances[two_smallest[1]]:
					if transition_next_state[i] != two_smallest[0]:
						fail_counts[epoch_id] += 1
						fail_count += 1
				elif hamming_distances[two_smallest[1]] < hamming_distances[two_smallest[0]]:
					if transition_next_state[i] != two_smallest[1]:
						fail_counts[epoch_id] += 1
						fail_count += 1
				elif self.count_multiple_matches_as_error or transition_next_state[i] != two_smallest[0]:
					fail_counts[epoch_id] += 1
					fail_count += 1
				trial_count += 1
		# print("fail counts are %s" % fail_counts)
		assert np.sum(fail_counts) == fail_count
		assert trial_count == (num_transitions * self.epochs)
		perr = fail_count / trial_count
		self.ehdist = self.match_hamming_counts / (num_transitions * self.epochs)
		normalized_fail_counts = fail_counts / num_transitions
		self.mean_error = np.mean(normalized_fail_counts)
		self.std_error = np.std(normalized_fail_counts)
		self.match_hamming_distribution = self.match_hamming_counts / trial_count
		self.distractor_hamming_distribution = self.distractor_hamming_counts / trial_count
		assert math.isclose(sum(self.match_hamming_distribution), 1.0), "match_hamming_distribution sum not 1: %s" % (
			sum(self.match_hamming_distribution))
		assert match_distances[-1] != flag_value
		self.match_distance_mean = np.mean(match_distances)
		self.match_distance_std = np.std(match_distances)
		self.distractor_distance_mean = np.mean(distractor_distances)
		self.distractor_distance_std = np.std(distractor_distances)
		self.counter_sum_mean = np.mean(counter_sums)
		self.counter_sum_std = np.std(counter_sums)
		self.counter_counts = counter_counts / np.sum(counter_counts)
		self.counter_counts_offset = counter_counts_offset
		self.overlap_counts = np.around(overlap_counts / epochs).astype(np.uint16)
		overlap_dist = np.bincount(self.overlap_counts, minlength=100)
		self.overlap_dist = overlap_dist / np.sum(overlap_dist)
		assert math.isclose(self.mean_error, perr)
		# print("fast_sdm_empirical epochs=%s, perr=%s, std_error=%s" % (self.epochs, perr, self.std_error))


def main():
	ncols = 512; actions=10; choices=10; states=100
	# ncols = 31; actions=3; choices=3; states=10
	# nrows=239; nact=3  # should give error rate of 10e-6
	# nrows=86; nact=2  # should be error rate of 10e-2
	# nrows=125; nact=2 # should give error rate 10e-3
	nrows = 75; nact=1  # for dot-product match, should give 10e-3
	nrows = 51; nact=1  # for dot-product match, should give 10e-2
	nrows = 39; nact=1  # for dot-product match, should give 10e-1
	nrows = 93; nact=1  # for dot-product match, should give 10e-4
	nrows = 126; nact=2; threshold_sum = True  # for hamming match, should give 10e-3
	# nrows=31; nact=2
	# nrows = 51; nact=1  # should give error 10e-1
	# nrows = 1; ncols=20; nact=1; actions=2; states=3; choices=2
	# fse = Fast_sdm_empirical(nrows, ncols, nact)
	# nrows = 24; ncols=20; nact=2; actions=2; states=7; choices=2
	# threshold_sum = False  # True for normal sdm, False for dot product matching
	epochs=100
	fse = Fast_sdm_empirical(nrows, ncols, nact, actions=actions, states=states, choices=choices,
		threshold_sum=threshold_sum, epochs=epochs)
	print("With nrows=%s, ncols=%s, nact=%s, threshold_sum=%s epochs=%s, mean_error=%s, std_error=%s" % (nrows, ncols,
		nact, threshold_sum, epochs, fse.mean_error, fse.std_error))
	print("match_distance mean=%s, std=%s; distractor_distance mean=%s, std=%s" % (fse.match_distance_mean,
		fse.match_distance_std, fse.distractor_distance_mean, fse.distractor_distance_std))
	print("counter_sum_mean=%s, counter_sum_std=%s" % (fse.counter_sum_mean, fse.counter_sum_std))

	# plot match and distractor hamming distributions
	plt.plot(fse.match_hamming_distribution, label="match")
	plt.plot(fse.distractor_hamming_distribution, label="distractor")
	plt.xlabel("Hamming distance")
	plt.title("match vs distractor hamming distance")
	plt.ylabel("relative frequency")
	plt.legend(loc='upper right')
	plt.grid()
	plt.show()

	# plot counter sums distribution
	xvals = np.arange(fse.counter_counts.size) - fse.counter_counts_offset
	plt.plot(xvals, fse.counter_counts)
	plt.xlabel("counter sum")
	plt.title("distribution of counter sums")
	plt.ylabel("relative frequency")
	# plt.legend(loc='upper right')
	plt.grid()
	plt.show()

if __name__ == "__main__":
	main()
