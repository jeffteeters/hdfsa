import numpy as np
import matplotlib.pyplot as plt
import math
from numba import jit
from numba import int32, float32, uint32, boolean, float64    # import the types
# from numba.experimental import jitclass
import sys

# spec = [
#     ('nrows', int32),
#     ('ncols', int32),
#     ('nact', int32),
#     ('actions', int32),
#     ('states', int32),
#     ('choices', int32),
#     ('epochs', int32),
#     ('count_multiple_matches_as_error', boolean),
#     ('roll_address', boolean),
#     ('debug', boolean),
#     ('hl_selection_method_hamming', boolean),
#     ('save_error_rate_vs_hamming_distance', boolean),
#     ('truncate_counters', boolean),
#     ('include_zero', boolean),
#     ('magnitude', int32),
#     ('match_hamming_counts', uint32[:]),
#     ('distractor_hamming_counts', uint32[:]),
#     ('ehdist', float64[:]),
#     ('mean_error', float64),
#     ('std_error', float64),
# ]

# @jitclass(spec)
class Fast_bundle_empirical():

	# calculate bundle empirical error using numpy as much as possible

	# @jit
	def __init__(self, ncols, actions=10, states=100, choices=10, epochs=10,
			count_multiple_matches_as_error=True, roll_address=True, debug=False,
			binarize_counters=True):
		# ncols is number of components in bundle (superposition vector)
		# actions, states, choices specify finite state automata stored in sdm.  Used to determine number transitions
		# stored (k) and size of item memory (d) used to compute probability of error in recall with d-1 distractors
		# epochs - number of times to store and recall FSA from sdm.  Each epoch stores and recalls all transitions.
		# count_multiple_matches_as_error = True to count multiple distractor hammings == match as error
		# binarize_counters - True if counters should be binarized.  If false, keep full counters and use
		# component multiplication +1 / -1 for binding and dot product for matching to item memory
		self.ncols = ncols
		self.actions = actions
		self.states = states
		self.choices = choices
		self.epochs = epochs
		assert actions >= choices, "Number actions must be >= number of choices"
		self.count_multiple_matches_as_error = count_multiple_matches_as_error
		self.roll_address = roll_address
		self.debug = debug
		self.binarize_counters = binarize_counters

	# def empiricalError(self):
		# compute empirical error by storing then recalling finite state automata from bundle
		fail_counts = np.zeros(self.epochs, dtype=np.uint16)
		rng = np.random.default_rng()
		num_transitions = self.states * self.choices
		trial_count = 0
		fail_count = 0
		for epoch_id in range(self.epochs):
			# import pdb; pdb.set_trace()
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
			# address = im_state[transition_state] * im_action[transition_action]
			assert transition_state.size == num_transitions
			assert transition_action.size == num_transitions
			assert transition_next_state.size == num_transitions
			# contents = np.zeros((self.nrows, self.ncols), dtype=np.int16)
			# save FSA into bundle contents matrix
			if self.roll_address:
				# assert False, "roll_address not supported"
				# roll each row in address by state number.  This to prevent the mysterious interference
				# roll done using method in: https://stackoverflow.com/questions/20360675/roll-rows-of-a-matrix-independently
				rows, column_indices = np.ogrid[:address.shape[0], :address.shape[1]]
				column_indices = column_indices - transition_state[:, np.newaxis]
				address = address[rows, column_indices]
			data = np.logical_xor(address, np.roll(im_state[transition_next_state], 1, axis=1))*2-1
			contents = np.sum(data, axis=0, dtype=np.int16)
			if self.binarize_counters:
				# binerize counters then use xor and hamming distance
				if num_transitions % 2 == 0:
					# add random vector
					random_plus_or_minus_one = rng.integers(0, high=2, size=self.ncols, dtype=np.int8)*2-1
					contents += random_plus_or_minus_one
				contents[contents > 0] = 1
				contents[contents < 0] = 0
			else:
				# don't binerize counters.  Use +1/-1 product and dot product with item memory to calculate distance
				address = address*2-1    # convert address to +1/-1
				im_state = im_state*2-1  # convert im_state to +1/-1
			for i in range(num_transitions):
				if self.binarize_counters:
					recalled_data = np.roll(np.logical_xor(address[i], contents), -1)
					distances = np.count_nonzero(im_state[:,] != recalled_data, axis=1)  # hamming distance
					# self.match_hamming_counts[hamming_distances[transition_next_state[i]]] += 1
					# self.distractor_hamming_counts[hamming_distances[(transition_next_state[i]+1) % self.states]] += 1 # a random distractor
				else:
					# compute distance via dot product.  Will be more negative for closer match due to difference btwn xor and *
					recalled_data = np.roll(address[i] * contents, -1) # would need to multiply by -1 for product has same result as xor
					distances = np.sum(im_state[:,] * recalled_data, axis=1) # dot product distance.
				two_smallest = np.argpartition(distances, 2)[0:2]
				# two_largest = np.argpartition(dot_products , -2)[-2:]
				if distances[two_smallest[0]] < distances[two_smallest[1]]:
					if transition_next_state[i] != two_smallest[0]:
						fail_counts[epoch_id] += 1
						fail_count += 1
				elif distances[two_smallest[1]] < distances[two_smallest[0]]:
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
		# self.ehdist = self.match_hamming_counts / (num_transitions * self.epochs)
		normalized_fail_counts = fail_counts / num_transitions
		self.mean_error = np.mean(normalized_fail_counts)
		self.std_error = np.std(normalized_fail_counts)
		# self.match_hamming_distribution = self.match_hamming_counts / trial_count
		# self.distractor_hamming_distribution = self.distractor_hamming_counts / trial_count
		# self.overlap_counts = np.around(overlap_counts / epochs).astype(np.uint16)
		# overlap_dist = np.bincount(self.overlap_counts, minlength=100)
		# self.overlap_dist = overlap_dist / np.sum(overlap_dist)
		# assert math.isclose(sum(self.match_hamming_distribution), 1.0), "match_hamming_distribution sum not 1: %s" % (
		# 	sum(self.match_hamming_distribution))
		assert math.isclose(self.mean_error, perr)
		# print("fast_sdm_empirical epochs=%s, perr=%s, std_error=%s" % (self.epochs, perr, self.std_error))


	# output for bundle sizes:
	# 	Bundle sizes, k=1000, d=100:
	# 1 - 24002
	# 2 - 40503
	# 3 - 55649
	# 4 - 70239
	# 5 - 84572
	# 6 - 98790
	# 7 - 112965
	# 8 - 127134
	# 9 - 141311

def main():
	# from 8-bit bundle, perr=3, width = 62919		[3, 62919, 300],
	# 	[2, 54000, 200],
	actions=10; choices=10; states=100
	# actions=3; choices=1; states=3; ncols=16	# for testing
	# ncols = 62919; # should give error of 10e-3
	# ncols = 55649;   # should give error of 10e-3; 3 - 55649 for binarized bundle
	ncols = 40503  # should be error 10e-2 for binarized bundle
	binarize_counters=False; epochs=10; roll_address=True
	fbe = Fast_bundle_empirical(ncols, actions=actions, states=states, choices=choices, epochs=epochs,
			count_multiple_matches_as_error=True, roll_address=roll_address, debug=False,
			binarize_counters=binarize_counters)
	print("actions=%s, states=%s, choices=%s, binarize_counters=%s,"
		" epochs=%s, roll_address=%s, ncols=%s, mean_error=%s, std_error=%s" % (actions, states, choices,
		binarize_counters, epochs, roll_address, ncols, fbe.mean_error, fbe.std_error))

if __name__ == "__main__":
	main()
