import numpy as np
import matplotlib.pyplot as plt
import math
# from numba import jit
# from numba import int32, float32, uint32, boolean, float64    # import the types
# from numba.experimental import jitclass
import sys
import time


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
			binarize_counters=True, bipolar_vectors=False, use_numpy_dot=False):
		# ncols is number of components in bundle (superposition vector)
		# actions, states, choices specify finite state automata stored in sdm.  Used to determine number transitions
		# stored (k) and size of item memory (d) used to compute probability of error in recall with d-1 distractors
		# epochs - number of times to store and recall FSA from sdm.  Each epoch stores and recalls all transitions.
		# count_multiple_matches_as_error = True to count multiple distractor hammings == match as error
		# binarize_counters - True if counters should be binarized.  If false, keep full counters and use
		# component multiplication +1 / -1 for binding and dot product for matching to item memory
		# bipolar_vectors - True if should use bipolar vectors (+1 / -1) for all operations. (Option included for
		# testing if is faster to do binding via multiplication; used for finding more accurate recall times).
		# use_numpy_dot - True if should use numpy dot product for non-binarized counters if bipolar_vectors is False
		# Should be the same result but faster.  Seems better to use bipolar_vectors=True
		# Turned out that bipolar_vectors didn't seem to be faster if binary_counters is True, but is faster
		# if binary_counters is false
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
		self.bipolar_vectors = bipolar_vectors

	# def empiricalError(self):
		# compute empirical error by storing then recalling finite state automata from bundle
		fail_counts = np.zeros(self.epochs, dtype=np.uint16)
		recall_times = np.empty(self.epochs, dtype=int)
		rng = np.random.default_rng()
		num_transitions = self.states * self.choices
		trial_count = 0
		fail_count = 0
		self.match_counts = {}  # key is distance, value is count; used in build_eedb to form pmf_stats
		self.distract_counts = {}  # key is distance, value is count; used in build_eedb to form pmf_stats
		for epoch_id in range(self.epochs):
			# import pdb; pdb.set_trace()
			im_action = rng.integers(0, high=2, size=(self.actions, self.ncols), dtype=np.int8)
			# im_action = np.random.randint(0,high=2,size=(self.actions, self.ncols)).astype(np.int8)
			im_state = rng.integers(0, high=2, size=(self.states, self.ncols), dtype=np.int8)
			# im_state = np.random.randint(0, high=2, size=(self.states, self.ncols)).astype(np.int8)
			if bipolar_vectors:
				im_action = im_action*2-1  # convert to bipolar
				im_state = im_state*2-1
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
			# START TIMEING FOR RECALL - part 1 bind action to state
			start_time = time.perf_counter_ns()
			if not bipolar_vectors:
				address = np.logical_xor(im_state[transition_state], im_action[transition_action])
			else:
				address = im_state[transition_state] * im_action[transition_action]
			# recall_times[epoch_id] = time.perf_counter_ns() - start_time
			recall_time1_bind = time.perf_counter_ns() - start_time
			# END TIMEING FOR RECALL - part 1 bind action to state
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
			if not bipolar_vectors:
				data = np.logical_xor(address, np.roll(im_state[transition_next_state], 1, axis=1))*2-1
			else:
				data = address * np.roll(im_state[transition_next_state], 1, axis=1)
			# use int32 for contents sum because dot product with unbinarized counters might not fit in 16 bit integer
			# just to be safe, use it in all cases; although this might slow down processing
			contents = np.sum(data, axis=0, dtype=np.int32)
			if self.binarize_counters:
				# binerize counters then use xor and hamming distance
				if num_transitions % 2 == 0:
					# add random vector
					random_plus_or_minus_one = rng.integers(0, high=2, size=self.ncols, dtype=np.int8)*2-1
					contents += random_plus_or_minus_one
				contents[contents > 0] = 1
				if not bipolar_vectors:
					contents[contents < 0] = 0
				else:
					contents[contents <= 0] = -1
			else:
				# don't binerize counters.  Use +1/-1 product and dot product with item memory to calculate distance
				# this conversion from binary to +1/-1 causes dot product for match to be negative when doing recall
				# which matches hamming distance being smaller than distractor hamming distance.
				if not bipolar_vectors:
					# only convert if not already using bipolar vectors
					address = address*2-1    # convert address to +1/-1
					im_state = im_state*2-1  # convert im_state to +1/-1
				else:
					address *= -1  # invert address so when do dot product match, smallest value is match like hamming
			recall_time2_distance_sum = 0
			for i in range(num_transitions):
				# START TIMEING FOR RECALL - part 2: find distances
				start_time = time.perf_counter_ns()
				if self.binarize_counters:
					if not bipolar_vectors:
						recalled_data = np.roll(np.logical_xor(address[i], contents), -1)
						distances = np.count_nonzero(im_state[:,] != recalled_data, axis=1)  # hamming distance
					else:
						recalled_data = np.roll(address[i] * contents, -1)
						distances = np.dot(im_state[:,], recalled_data) * (-1) # dot product distance.
					# self.match_hamming_counts[hamming_distances[transition_next_state[i]]] += 1
					# self.distractor_hamming_counts[hamming_distances[(transition_next_state[i]+1) % self.states]] += 1 # a random distractor
				else:
					if not bipolar_vectors:
					# if False:   # force to use dot product, see if this fixes timing problem
						# original code
						# compute distance via dot product.  Will be more negative for closer match due to difference btwn xor and *
						recalled_data = np.roll(address[i] * contents, -1) # would need to multiply by -1 for product has same result as xor
						if not use_numpy_dot:
							distances = np.sum(im_state[:,] * recalled_data, axis=1) # dot product distance.
						else:
							distances = np.dot(im_state[:,], recalled_data) # * (-1)  Should be no need to
					else:
						# import pdb; pdb.set_trace()
						# astype(np.int32) needed because np.dot may have overflow
						recalled_data = np.roll(address[i] * contents, -1)
						distances = np.dot(im_state[:,], recalled_data) # * (-1) # dot product distance.
				two_smallest = np.argpartition(distances, 2)[0:2]
				# END TIMING FOR RECALL
				recall_time2_distance = time.perf_counter_ns() - start_time
				recall_time2_distance_sum += recall_time2_distance
				add_count(self.match_counts, distances[transition_next_state[i]])  # add to count for match distance
				# two_largest = np.argpartition(dot_products , -2)[-2:]
				if distances[two_smallest[0]] < distances[two_smallest[1]]:
					if transition_next_state[i] != two_smallest[0]:
						fail_counts[epoch_id] += 1
						fail_count += 1
						closest_distractor = distances[two_smallest[0]]
					else:
						closest_distractor = distances[two_smallest[1]]
				elif distances[two_smallest[1]] < distances[two_smallest[0]]:
					if transition_next_state[i] != two_smallest[1]:
						fail_counts[epoch_id] += 1
						fail_count += 1
						closest_distractor = distances[two_smallest[1]]
					else:
						closest_distractor = distances[two_smallest[0]]
				elif self.count_multiple_matches_as_error or transition_next_state[i] != two_smallest[0]:
					fail_counts[epoch_id] += 1
					fail_count += 1
					closest_distractor = distances[two_smallest[0]]
				add_count(self.distract_counts, closest_distractor)
				trial_count += 1
			recall_time_total = recall_time1_bind + recall_time2_distance_sum
			recall_times[epoch_id] = recall_time_total
			if False and epoch_id <= 10:
				print("bind_time=%.3e (%.4f%%), distance_time=%.3e (%.4f%%)" % (
					recall_time1_bind, recall_time1_bind*100/recall_time_total,
					recall_time2_distance_sum, recall_time2_distance_sum/recall_time_total))
		# print("fail counts are %s" % fail_counts)
		assert np.sum(fail_counts) == fail_count
		assert trial_count == (num_transitions * self.epochs)
		self.num_transitions = num_transitions  # used in build_eedb
		perr = fail_count / trial_count
		# self.ehdist = self.match_hamming_counts / (num_transitions * self.epochs)
		self.fail_counts = fail_counts  # used by build_eedb
		normalized_fail_counts = fail_counts / num_transitions
		self.mean_error = np.mean(normalized_fail_counts)
		self.std_error = np.std(normalized_fail_counts)
		self.recall_time_mean = np.mean(recall_times)
		self.recall_time_std = np.std(recall_times)
		self.recall_time_min = recall_times.min()  # same minimum recall time
		# self.match_hamming_distribution = self.match_hamming_counts / trial_count
		# self.distractor_hamming_distribution = self.distractor_hamming_counts / trial_count
		# self.overlap_counts = np.around(overlap_counts / epochs).astype(np.uint16)
		# overlap_dist = np.bincount(self.overlap_counts, minlength=100)
		# self.overlap_dist = overlap_dist / np.sum(overlap_dist)
		# assert math.isclose(sum(self.match_hamming_distribution), 1.0), "match_hamming_distribution sum not 1: %s" % (
		# 	sum(self.match_hamming_distribution))
		assert math.isclose(self.mean_error, perr)
		# print("fast_sdm_empirical epochs=%s, perr=%s, std_error=%s" % (self.epochs, perr, self.std_error))

def add_count(counts, distance):
	# counts is a dictionary mapping a distance to a count
	# used for match_counts and distract_counts
	if distance in counts:
		counts[distance] += 1
	else:
		counts[distance] = 1


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

# timing results:
#
# binarize_counters=True, bipolar_vectors=False epochs=4, roll_address=False, mean_error=0.00849, std_error=0.00206,
# mean_time=5.320e+09, min_time=5.294e+09  FOR BINARIZED_COUNTERS, BIPOLAR_VECTORS=FALSE BEST
#
# binarize_counters=True, bipolar_vectors=True epochs=4, roll_address=False, mean_error=0.0085, std_error=0.002291,
# mean_time=9.260e+09, min_time=9.206e+09
#
# binarize_counters=False, bipolar_vectors=False epochs=4, roll_address=False, mean_error=0.0, std_error=0.0,
# mean_time=1.959e+10, min_time=1.826e+10
#
# binarize_counters=False, bipolar_vectors=True epochs=4, roll_address=False, mean_error=0.00075, std_error=0.000,
# mean_time=8.569e+09, min_time=8.481e+09  FOR NON-BINARIZE_COUNTERS, BIPOLAR_VECTORS=TRUE BEST
#

def main():
	# from 8-bit bundle, perr=3, width = 62919		[3, 62919, 300],
	# 	[2, 54000, 200],
	actions=10; choices=10; states=100
	# actions=3; choices=1; states=3; ncols=16	# for testing
	# ncols = 62919; # should give error of 10e-3
	# ncols = 55649;   # should give error of 10e-3; 3 - 55649 for binarized bundle
	ncols = 40503  # should be error 10e-2 for binarized bundle
	binarize_counters=False; epochs=4; roll_address=False
	bipolar_vectors=True
	# binarize_counters=False; epochs=10; roll_address=True
	fbe = Fast_bundle_empirical(ncols, actions=actions, states=states, choices=choices, epochs=epochs,
			count_multiple_matches_as_error=True, roll_address=roll_address, debug=False,
			binarize_counters=binarize_counters, bipolar_vectors=bipolar_vectors)
	print("ncols=%s, actions=%s, states=%s, choices=%s, binarize_counters=%s, bipolar_vectors=%s"
		" epochs=%s, roll_address=%s, mean_error=%s, std_error=%s,"
		" mean_time=%.3e, min_time=%.3e" % (ncols, actions, states, choices,
		binarize_counters, bipolar_vectors, epochs, roll_address, fbe.mean_error, fbe.std_error,
		fbe.recall_time_mean, fbe.recall_time_min))

if __name__ == "__main__":
	main()
