import numpy as np
import matplotlib.pyplot as plt


class Fast_sdm_empirical():

	# calculate SDM empirical error using numpy as much as possible

	def __init__(self, nrows, ncols, nact, actions=10, states=100, choices=10, epochs=10, count_multiple_matches_as_error=True, debug=False,
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
		self.nrows = nrows
		self.ncols = ncols
		self.nact = nact
		self.actions = actions
		self.states = states
		self.choices = choices
		self.epochs = epochs
		assert actions >= choices, "Number actions must be >= number of choices"
		self.count_multiple_matches_as_error = count_multiple_matches_as_error
		self.debug = debug
		self.save_error_rate_vs_hamming_distance = save_error_rate_vs_hamming_distance
		self.empiricalError()


	def empiricalError(self):
		# compute empirical error by storing then recalling finite state automata from SDM
		fail_counts = np.zeros(self.epochs, dtype=np.uint16)
		rng = np.random.default_rng()
		num_transitions = self.states * self.choices
		for epoch_id in range(self.epochs):
			im_action = rng.integers(0, high=2, size=(self.actions, self.ncols), dtype=np.int8)
			im_state = rng.integers(0, high=2, size=(self.states, self.ncols), dtype=np.int8)
			transition_action = np.empty((self.states, self.choices), dtype=np.uint16)
			transition_next_state = np.empty((self.states, self.choices), dtype=np.uint16)
			for i in range(self.states):
				transition_action[i,:] = rng.choice(self.actions, size=self.choices, replace=False)
				transition_next_state[i,:] = rng.choice(self.states, size=self.choices, replace=False)
			transition_state = np.repeat(np.arange(self.states), self.choices)
			transition_action = transition_action.flatten()
			transition_next_state = transition_next_state.flatten()
			assert transition_state.size == num_transitions
			assert transition_action.size == num_transitions
			assert transition_next_state.size == num_transitions
			transition_hard_locations = np.empty((num_transitions, self.nact), dtype=np.uint16)
			for i in range(num_transitions):
				transition_hard_locations[i,:] = rng.choice(self.nrows, size=self.nact, replace=False)
			contents = np.zeros((self.nrows, self.ncols), dtype=np.int16)
			# save FSA into SDM contents matrix
			address = np.logical_xor(im_state[transition_state], im_action[transition_action])
			data = np.logical_xor(address, np.roll(im_state[transition_next_state], 1, axis=1))*2-1
			for i in range(num_transitions):
				contents[transition_hard_locations[i,:]] += data[i]
			# recall data from contents matrix
			# recalled_data = np.empty((num_transitions, self.ncols), dtype=np.int8)
			for i in range(num_transitions):
				recalled_vector = (contents[transition_hard_locations[i,:]].sum(axis=0) > 0)
				recalled_data = np.roll(np.logical_xor(address[i], recalled_vector), -1)
				hamming_distances = np.count_nonzero(im_state[:,] != recalled_data, axis=1)
				two_smallest = np.argpartition(hamming_distances, 2)[0:2]
				# import pdb; pdb.set_trace()
				if ((transition_next_state[i] != two_smallest[0] and transition_next_state[i] != two_smallest[1]) or
					(self.count_multiple_matches_as_error
						and hamming_distances[two_smallest[0]] == hamming_distances[two_smallest[1]])):
					fail_counts[epoch_id] += 1
		print("fail counts are %s" % fail_counts)
		normalized_fail_counts = fail_counts / num_transitions
		self.mean_error = np.mean(normalized_fail_counts)
		self.std_error = np.std(normalized_fail_counts)


def main():
	ncols = 512
	# nrows=239; nact=3  # should give error rate of 10e-6
	nrows=86; nact=2  # should be error rate of 10e-2
	# nrows = 51; nact=1  # should give error 10e-1
	fse = Fast_sdm_empirical(nrows, ncols, nact)
	# nrows = 24; ncols=20; nact=2; actions=2; states=7; choices=2
	# fse = Fast_sdm_empirical(nrows, ncols, nact, actions=actions, states=states, choices=choices)
	print("With nrows=%s, ncols=%s, nact=%s, mean_error=%s, std_error=%s" % (nrows, ncols, nact, fse.mean_error,
		fse.std_error))

if __name__ == "__main__":
	main()
