import numpy as np
import matplotlib.pyplot as plt
import math
from statistics import NormalDist
import matplotlib.pyplot as plt


class Find_interference():

	# Find which combinations of terms are reducing accuracy
	# in recall from SDM empirical error using numpy as much as possible

	def __init__(self, nrows, ncols, nact, actions=10, states=100, choices=10, epochs=1000, count_multiple_matches_as_error=True, debug=False,
			save_error_rate_vs_hamming_distance=False, number_terms_to_test=None, num_combinations_to_display=10):
		# nrows is number of rows (hard locations) in the SDM
		# ncols is the number of columns
		# nact is activaction count
		# actions, states, choices specify finite state automata stored in sdm.  Used to determine number transitions
		# stored (k) and size of item memory (d) used to compute probability of error in recall with d-1 distractors
		# epochs - number of times to store and recall FSA from sdm.  Each epoch stores and recalls all transitions.
		# count_multiple_matches_as_error = True to count multiple distractor hammings == match as error
		# save_error_rate_vs_hamming_distance true to save error_count_vs_hamming, used to figure out what
		# happens to error rate when distractor(s) having same hamming distance as match not always counted as error
		# number_terms_to_test - None, test all terms (states * actions), otherwise number of terms to save and recall
		# max_combinations_to_display - combinations to show that have highest difference

		# print("starting Fast_sdm_empirical, nrows=%s, ncols=%s, nact=%s, actions=%s, states=%s, choices=%s" % (
		#	nrows, ncols, nact, actions, states, choices))
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
		self.number_terms_to_test = number_terms_to_test
		self.num_combinations_to_display = num_combinations_to_display
		self.num_transitions = self.states * self.choices
		assert number_terms_to_test is None or number_terms_to_test <= self.num_transitions
		self.number_transitions_to_store = self.num_transitions if number_terms_to_test is None else number_terms_to_test
		self.ri_roll = self.empiricalError(roll_address=True)  # ri - result info
		self.ri_no_roll = self.empiricalError(roll_address=False)
		self.overlaps = self.compute_overlaps()
		self.show_largest_changes()


	def empiricalError(self, roll_address=False):
		# compute empirical error by storing then recalling finite state automata from SDM
		# if roll_address is True, shift address vectors before using address to store and recall

		# following stores count of errors for each stored pattern.  Keys are pattern, like:
		# "s0,a2,p3;s1,a2,p0; ..." (sorted by state e.g s0, s1, s2, ...)
		# values are array of match hamming distances for each recalled item in order of transitions
		pattern_hammings = {}
		fail_counts = np.zeros(self.epochs, dtype=np.uint16)
		self.match_hamming_counts = np.zeros(self.ncols+1, dtype=np.uint32)
		self.distractor_hamming_counts = np.zeros(self.ncols+1, dtype=np.uint32)
		num_transitions = self.num_transitions
		rng = np.random.default_rng()
		trial_count = 0
		fail_count = 0
		# largest_match_hamming = 0
		# combinations_found = 0
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
			for i in range(self.num_transitions):
				transition_hard_locations[i,:] = rng.choice(self.nrows, size=self.nact, replace=False)
			contents = np.zeros((self.nrows, self.ncols), dtype=np.int16)
			# save FSA into SDM contents matrix
			address = np.logical_xor(im_state[transition_state], im_action[transition_action])
			if roll_address:
				# roll each row in address by state number.  This to prevent the mysterious interference
				# roll done using method in: https://stackoverflow.com/questions/20360675/roll-rows-of-a-matrix-independently
				rows, column_indices = np.ogrid[:address.shape[0], :address.shape[1]]
				column_indices = column_indices - transition_state[:, np.newaxis]
				address = address[rows, column_indices]
			data = np.logical_xor(address, np.roll(im_state[transition_next_state], 1, axis=1))*2-1
			transitions_to_save = np.arange(num_transitions) if self.number_terms_to_test is None else rng.choice(num_transitions,
				size=self.number_transitions_to_store, replace=False)
			labels = np.empty(transitions_to_save.size, dtype='<U14')
			for i in range(transitions_to_save.size):
				tri = transitions_to_save[i]  # transition index
				contents[transition_hard_locations[tri,:]] += data[tri]
				labels[i] = "s%s,a%s,p%s" % (transition_state[tri], transition_action[tri], transition_next_state[tri])
			lidx = np.argsort(labels)  # index of sorted order of labels
			sorted_label = ";".join([labels[lidx[i]] for i in range(len(labels))])

			# recall data from contents matrix
			found_transition_hammings = np.empty(transitions_to_save.size, dtype=np.uint16)
			for i in range(transitions_to_save.size):
				tri = transitions_to_save[i]  # transition index
				recalled_vector = (contents[transition_hard_locations[tri,:]].sum(axis=0) > 0)
				recalled_data = np.roll(np.logical_xor(address[tri], recalled_vector), -1)
				hamming_distances = np.count_nonzero(im_state[:,] != recalled_data, axis=1)
				match_hamming_distance = hamming_distances[transition_next_state[tri]]
				found_transition_hammings[lidx[i]] = match_hamming_distance
				# if match_hamming_distance >= largest_match_hamming:
				# 	if epoch_id > self.epochs / 2:
				# 		combinations_found += 1
				# 		if combinations_found < self.max_combinations_to_display:
				# 			print("found match hamming (%s) larger than previous max (%s):" % (match_hamming_distance,
				# 				largest_match_hamming))
				# 			for j in range(len(transitions_to_save)):
				# 				ti = transitions_to_save[j]
				# 				flag = "*" if i == ti else ""
				# 				print("%s%s s%s a%s p%s" % (j, flag, transition_state[ti], transition_action[ti],
				# 					transition_next_state[ti]))
				# 	else:
				# 		largest_match_hamming = match_hamming_distance
				self.match_hamming_counts[match_hamming_distance] += 1
				self.distractor_hamming_counts[hamming_distances[(transition_next_state[i]+1) % self.states]] += 1 # a random distractor

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
			if sorted_label not in pattern_hammings:
				pattern_hammings[sorted_label] = [found_transition_hammings,]
			else:
				pattern_hammings[sorted_label].append(found_transition_hammings)

		# print("fail counts are %s" % fail_counts)
		assert np.sum(fail_counts) == fail_count
		# assert trial_count == (num_transitions * self.epochs)
		perr = fail_count / trial_count
		hdist = self.match_hamming_counts / (num_transitions * self.epochs)
		normalized_fail_counts = fail_counts / num_transitions
		mean_error = np.mean(normalized_fail_counts)
		std_error = np.std(normalized_fail_counts)
		pattern_stats = self.compute_pattern_stats(pattern_hammings)
		info = {"perr": perr, "hdist":hdist, "mean_error": mean_error, "std_error": std_error,
			"pattern_hammings": pattern_hammings, "pattern_stats": pattern_stats}
		return info
		# assert math.isclose(self.mean_error, perr)
		# print("fast_sdm_empirical passed assertions, perr=%s" % perr)
		# print("%s combinations found with hamming distance greater than %s" %(combinations_found, largest_match_hamming))

	def compute_pattern_stats(self, pattern_hammings):
		# computes mean and standard deviation for each component and for all components in pattern hammings
		pattern_stats = {}
		# import pdb; pdb.set_trace()
		for key, ham in pattern_hammings.items():
			hamar = np.empty((len(ham), ham[0].size), dtype=np.uint16)
			# copy values from list of arrays to a single 2d array
			for i in range(len(ham)):
				hamar[i,:] = ham[i]
			mean_p = np.mean(hamar, axis=0)
			std_p = np.std(hamar, axis=0)
			if np.count_nonzero(std_p) < self.number_terms_to_test:
				print("found zero std_p: %s" % std_p)
				import pdb; pdb.set_trace()
			mean_all = np.mean(hamar)
			std_all = np.std(hamar)
			pattern_stats[key] = {"mean_p":mean_p, "std_p":std_p, "mean_all":mean_all, "std_all":std_all}
		return pattern_stats

	def compute_overlaps(self):
		# compute overlaps between pattern hammings
		an = self.ri_no_roll  # address information no roll
		ar = self.ri_roll  # address with roll information
		overlaps = {}
		ov_all = np.empty(len(an["pattern_hammings"].keys()))
		ov_par = np.empty((len(an["pattern_hammings"].keys()), self.number_transitions_to_store))
		ov_key = sorted(an["pattern_hammings"].keys())  # sorted keys
		ia = 0
		for key in ov_key:
			overlap_all = 1.0 - NormalDist(mu=an["pattern_stats"][key]["mean_all"],
				sigma=an["pattern_stats"][key]["std_all"]).overlap(
				NormalDist(mu=ar["pattern_stats"][key]["mean_all"], sigma=ar["pattern_stats"][key]["std_all"]))
			ov_all[ia] = overlap_all
			overlap_parts = np.empty(self.number_transitions_to_store)
			for i in range(self.number_transitions_to_store):
				overlap_parts[i] = 1.0 - NormalDist(mu=an["pattern_stats"][key]["mean_p"][i],
					sigma=an["pattern_stats"][key]["std_p"][i]).overlap(
					NormalDist(mu=ar["pattern_stats"][key]["mean_p"][i], sigma=ar["pattern_stats"][key]["std_p"][i]))
			ov_par[ia,:] = overlap_parts
			ov_key.append(key)
			ia += 1
			overlaps[key] = {"all": overlap_all, "parts": overlap_parts}
		self.overlaps = overlaps
		self.ov_all = ov_all
		self.ov_par = ov_par
		self.ov_key = ov_key
		fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
		n_bins = 100
		# We can set the number of bins with the *bins* keyword argument.
		axs[0].hist(ov_all, bins=n_bins)
		axs[1].hist(ov_par.flatten(), bins=n_bins)
		plt.show()

	def show_largest_changes(self):
		print("error without roll = %s, error with roll =%s" %(self.ri_no_roll["perr"], self.ri_roll["perr"]))
		print("Number patterns without roll=%s, with roll=%s" % (len(self.ri_no_roll["pattern_hammings"]),
			len(self.ri_roll["pattern_hammings"])))
		# find largest values
		nc = self.num_combinations_to_display
		# from: https://stackoverflow.com/questions/10337533/a-fast-way-to-find-the-largest-n-elements-in-an-numpy-array
		ov_parf = ov_par.flatten()
		tmp_idx = np.argpartition(-ov_parf, nc)[0:nc]
		for idx in tmp_idx:
			ov_amt = ov_parf[idx]
			key_i = int(idx / self.ov_par.shape[0])
			trm_i = idx % self.ov_par.shape[0]  # index of term of highest hamming distance
			terms = self.ov_key[key_i].split(";")
			terms[trm_i] = "<%s>" % terms[trm_i]
			print("%s - %s" % (ov_amt, ";".join(terms)))

		# import pdb; pdb.set_trace()


def main():
	# ncols = 512
	# nrows=239; nact=3  # should give error rate of 10e-6
	# nrows=86; nact=2  # should be error rate of 10e-2
	# nrows = 51; nact=1  # should give error 10e-1
	# nrows = 1; ncols=20; nact=1; actions=2; states=3; choices=2
	# fse = Fast_sdm_empirical(nrows, ncols, nact)
	# nrows = 24; ncols=20; nact=2; actions=2; states=7; choices=2
	nrows=1; ncols=128; nact=1; k=9; d=3; actions=3; states=3; choices=3; epochs=1000000; number_terms_to_test=5
	fse = Find_interference(nrows, ncols, nact, actions=actions, states=states, choices=choices, epochs=epochs,
		number_terms_to_test=number_terms_to_test)
	# print("With nrows=%s, ncols=%s, nact=%s, mean_error=%s, std_error=%s" % (nrows, ncols, nact, fse.mean_error,
	# 	fse.std_error))

if __name__ == "__main__":
	main()

""" Output
found match hamming (71) larger than previous max (71):
0 s0 a0 p1
1* s1 a0 p1
2 s0 a1 p0
3 s2 a2 p1
4 s1 a1 p0

Bundle is:
   [ s0 a0 p1 + s1 a0 p1 + s0 a1 p0 + s2 a2 p1 + s1 a1 p0 ]

Multiply by (s1 a0) to find next state, should be p1

(s1 a0) [ s0 a0 p1 + s1 a0 p1 + s0 a1 p0 + s2 a2 p1 + s1 a1 p0 ]

=  (s1 a0) (s0 a0 p1) + (s1 a0) (s1 a0 p1) + (s1 a0) (s0 a1 p0) + (s1 a0) (s2 a2 p1) + (s1 a0) (s1 a1 p0) ]

=  (s0 s1 p1) +                (p1) +         (s0 s1 a0 a1 p0)      + (s1 s2 a0 a2 p1) + (a0 a1 p0)

Has term: (a0 a1 p0) which is contained in term: (s0 s1 a0 a1 p0)


found match hamming (71) larger than previous max (71):
0 s2 a0 p0
1 s0 a2 p1
2 s1 a0 p1
3* s2 a2 p1
4 s1 a2 p0



Bundle is:
[ (s2 a0 p0) + (s0 a2 p1) + (s1 a0 p1) + (s2 a2 p1) + (s1 a2 p0)]

Multiply by (s2 a2) to obtain next state, should be (p1)

(s2 a2)  [ (s2 a0 p0) + (s0 a2 p1) + (s1 a0 p1) + (s2 a2 p1) + (s1 a2 p0)]

=        [ (s2 a2) (s2 a0 p0) +  (s2 a2) (s0 a2 p1) + (s2 a2) (s1 a0 p1) + (s2 a2) (s2 a2 p1) + (s2 a2) (s1 a2 p0) ]

=        [ (a0 a2 p0)         +   (s0 s2 p1)        +  (s1 s2 a0 a2 p1)  + ( p1 )  + (s1 s2 p0)]

Nothing I could see

Another one:

0 s1 a2 p0
1 s1 a1 p2
2 s1 a0 p1
3 s2 a2 p0
4* s2 a0 p1

(s2 a0) *
	s1 a2 p0 = s1 s2 a0 a2 p0
	s1 a1 p2 = s1 s2 a0 a1 p2
	s1 a0 p1 = s1 s2 p1
	s2 a2 p0 = a0 a2 p0
	* s2 a0 p1 = p1

	Contains one part inside


"""
