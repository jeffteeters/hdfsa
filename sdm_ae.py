# Class to find sdm error analytically

import math
from scipy.stats import hypergeom
from scipy.stats import binom
# from scipy.special import binom as binom_coef
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import copy

import pprint
pp = pprint.PrettyPrinter(indent=4)


class Cop:
	# chunk overlap pattern

	def __init__(self, nrows, nact, k, threshold=10000, show_pruning=False, show_items=True):
		# nrows - number of rows in sdm
		# nact - activaction count
		# k - number of items to store in sdm
		# threshold - maximum ratio of largest to smallest probability.  Drop patterns that have smaller probability
		#  This done to limit number of patterns to only those that are contributing the most to the result
		self.nrows = nrows
		self.nact = nact
		self.k = k
		self.threshold = threshold
		self.show_pruning = show_pruning
		self.show_items = show_items
		self.ov1_pmf = self.compute_one_item_overlap_pmf()
		# print("self.ov1_pmf=%s" % self.ov1_pmf)
		self.key_increments = self.compute_key_increments()
		self.chunk_probabilities = self.set_initial_chunk_probabilities()
		# print("before add_overlap, self.chunk_probabilities=%s" % self.chunk_probabilities)
		for i in range(2, k):
			self.add_overlap(i)
			end = "\n" if i % 10 == 0 else ""
			print("%s-%s "%(i,len(self.chunk_probabilities)), end=end)
			# print("after add_overlap %s self.chunk_probabilities=\n%s" % (i, self.chunk_probabilities))
		self.add_error_rate()
		self.display_result()


class Sdm_error_analytical:

	# Compute error of recall vector from SDM and matching to item memory

	def __init__(self, nrows, nact, k, threshold=10000, show_pruning=False, show_items=True):
		# nrows - number of rows in sdm
		# nact - activaction count
		# k - number of items to store in sdm
		# threshold - maximum ratio of largest to smallest probability.  Drop patterns that have smaller probability
		#  This done to limit number of patterns to only those that are contributing the most to the result
		self.nrows = nrows
		self.nact = nact
		self.k = k
		self.threshold = threshold
		self.show_pruning = show_pruning
		self.show_items = show_items
		self.ov1_pmf = self.compute_one_item_overlap_pmf()
		# print("self.ov1_pmf=%s" % self.ov1_pmf)
		self.key_increments = self.compute_key_increments()
		self.cop_key = self.key_increments.copy()
		self.cop_prb = self.ov1_pmf.copy()
		# print("before add_overlap, self.chunk_probabilities=%s" % self.chunk_probabilities)
		for i in range(2, k):
			self.add_overlap(i)
			end = "\n" if i % 10 == 0 else ""
			print("%s-%s "%(i,len(self.chunk_probabilities)), end=end)
			# print("after add_overlap %s self.chunk_probabilities=\n%s" % (i, self.chunk_probabilities))
		self.add_error_rate()
		self.display_result()

	def compute_one_item_overlap_pmf(self):
		# based on: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html#scipy.stats.hypergeom
		[M, n, N] = [self.nrows, self.nact, self.nact]
		rv = hypergeom(M, n, N)
		x = np.arange(0, n+1)
		pmf = rv.pmf(x)
		return pmf

	def compute_key_increments(self):
		# keys in the self.chunk_probabilities dictionary are integers with each 7 bits (0 to 127) representing the
		# overlap count for 0 through nact overlaps.  Least significant digits are used for count of 0 overlaps.
		# In other words, keys look like: 444333222111000, where 000 is the count of 0 overlaps, and 111 is the
		# count of 1 overlaps.  The key increment array has the numbers to add to the previous key to convert it
		# to the new key.
		key_increments=np.arange(self.nact+1, dtype=np.uint)*128
		key_increments[0] = 1   # set first key increment so it's not zero.  Others are all 1 (multiple of 128)
		# ki = 1
		# key_increments = []
		# for i in range(self.nact+1):
		# 	key_increments.append(ki)
		# 	ki = ki * 128
		# # print("key_increments=%s" % key_increments)
		return key_increments

	def add_overlap(self, iteration):
		# givin current cop_key, cop_prob, update it by multiplying each probability by every probability in
		# self.ov1_pmf and then combine terms.  Remove any terms with probability less than threshold
		cop_key = np.add.outer(self.key_increments, cop_key).flatten()
		cop_prb = np.outer(self.ov1_pmf, cop_prb).flatten()
		




		chunk_probabilities = self.chunk_probabilities
		items_before = len(chunk_probabilities)
		max_probability = 0.
		for prev_key in list(chunk_probabilities):
			prev_prob = chunk_probabilities[prev_key]
			for i in range(self.nact + 1):
				new_prob = prev_prob * self.ov1_pmf[i]
				if new_prob > max_probability:
					max_probability = new_prob
				new_key = prev_key + self.key_increments[i]
				if new_key in chunk_probabilities:
					chunk_probabilities[new_key] += new_prob
				else:
					chunk_probabilities[new_key] = new_prob
				# print("i=%i, prev=(%s, %s) new=(%s,%s), cp[nk]=%s" % (i, prev_key, prev_prob, new_key, new_prob,
				# 	chunk_probabilities[new_key]))
			del chunk_probabilities[prev_key]
		# now check for terms to remove
		min_allowd_probability = max_probability / self.threshold
		prune_count = 0
		prune_prob = 0.0
		total_prob = 0
		items_before_prune = len(chunk_probabilities)
		pruned_items = {}
		for key in list(chunk_probabilities):
			prob = chunk_probabilities[key]
			total_prob += prob
			if prob < min_allowd_probability:
				pruned_items[key] = prob
				del chunk_probabilities[key]
				prune_count += 1
				prune_prob += prob
		items_after_prune = len(chunk_probabilities)
		if self.show_pruning:
			print("cop %s overlap, items_before=%s, items_before_prune=%s, items_after_prune=%s, prune_count=%s,"
				" prune_prob=%s, total_prob=%s" % (iteration, items_before, items_before_prune, items_after_prune,
					prune_count, prune_prob, total_prob))
			print("pruned_items are: %s" % pruned_items)



class Cop:
	# "chunk overlay probability" - keeps track of multiple overlaps (chunks) onto target rows from the same item.
	# This is needed because multiple overlaps from the same item change the probability of error when computing
	# the sum.  For example, a chunk of size 2 would contribute -2 or +2 to the sum.  But two independent items
	# would contribute either (-2, 0, +2).  Another example, chunk of size 3 contributes -3 or +3.  But three
	# independent items could contribute: -3, -1, 1, or 3.

	def __init__(self, nrows, nact, k, threshold=10000, show_pruning=False, show_items=True):
		# nrows - number of rows in sdm
		# nact - activaction count
		# k - number of items to store in sdm
		# threshold - maximum ratio of largest to smallest probability.  Drop patterns that have smaller probability
		#  This done to limit number of patterns to only those that are contributing the most to the result
		self.nrows = nrows
		self.nact = nact
		self.k = k
		self.threshold = threshold
		self.show_pruning = show_pruning
		self.show_items = show_items
		self.ov1_pmf = self.compute_one_item_overlap_pmf()
		# print("self.ov1_pmf=%s" % self.ov1_pmf)
		self.key_increments = self.compute_key_increments()
		self.chunk_probabilities = self.set_initial_chunk_probabilities()
		# print("before add_overlap, self.chunk_probabilities=%s" % self.chunk_probabilities)
		for i in range(2, k):
			self.add_overlap(i)
			end = "\n" if i % 10 == 0 else ""
			print("%s-%s "%(i,len(self.chunk_probabilities)), end=end)
			# print("after add_overlap %s self.chunk_probabilities=\n%s" % (i, self.chunk_probabilities))
		self.add_error_rate()
		self.display_result()





	def add_overlap(self, iteration):
		# givin current chunk_probabilities, update it by multiplying each probability by every probability in
		# self.ov1_pmf and then combine terms.  Remove any terms with probability less than threshold.
		chunk_probabilities = self.chunk_probabilities
		items_before = len(chunk_probabilities)
		max_probability = 0.
		for prev_key in list(chunk_probabilities):
			prev_prob = chunk_probabilities[prev_key]
			for i in range(self.nact + 1):
				new_prob = prev_prob * self.ov1_pmf[i]
				if new_prob > max_probability:
					max_probability = new_prob
				new_key = prev_key + self.key_increments[i]
				if new_key in chunk_probabilities:
					chunk_probabilities[new_key] += new_prob
				else:
					chunk_probabilities[new_key] = new_prob
				# print("i=%i, prev=(%s, %s) new=(%s,%s), cp[nk]=%s" % (i, prev_key, prev_prob, new_key, new_prob,
				# 	chunk_probabilities[new_key]))
			del chunk_probabilities[prev_key]
		# now check for terms to remove
		min_allowd_probability = max_probability / self.threshold
		prune_count = 0
		prune_prob = 0.0
		total_prob = 0
		items_before_prune = len(chunk_probabilities)
		pruned_items = {}
		for key in list(chunk_probabilities):
			prob = chunk_probabilities[key]
			total_prob += prob
			if prob < min_allowd_probability:
				pruned_items[key] = prob
				del chunk_probabilities[key]
				prune_count += 1
				prune_prob += prob
		items_after_prune = len(chunk_probabilities)
		if self.show_pruning:
			print("cop %s overlap, items_before=%s, items_before_prune=%s, items_after_prune=%s, prune_count=%s,"
				" prune_prob=%s, total_prob=%s" % (iteration, items_before, items_before_prune, items_after_prune,
					prune_count, prune_prob, total_prob))
			print("pruned_items are: %s" % pruned_items)

	def add_error_rate(self):
		# add error rate to chunk_probabilities
		chunk_probabilities = self.chunk_probabilities
		for key in list(chunk_probabilities):
			prob = chunk_probabilities[key]
			err = self.cop_err(self.nact, key)
			chunk_probabilities[key] = (prob, err)

	def display_result(self):
		num_items = len(self.chunk_probabilities)
		total_prob = 0.0
		for key, info in self.chunk_probabilities.items():
			prob, err = info
			total_prob += prob
		print("After %s items stored with nact=%s, threshold=%s, size cop = %s, total_probability = %s" % (self.k,
			self.nact, self.threshold, num_items, total_prob))
		if self.show_items:
			max_num_to_show = 50
			print("items are:")
			for key, info in sorted(self.chunk_probabilities.items())[0:max_num_to_show]:
				print("%s -> %s" % (key, info))
			# self.test_cop_err()

	def cop_err(self, nact, key):
		# determine probability of error given chunk overlap pattern specified in key
		cfreq=[]
		# print("entered cop_err, key=%s" % key)
		dkey = int(key / 1000)  # strip off no overlap count
		if dkey == 0:
			return 0.0  # no overlap, so error rate is zero
		while dkey > 0:
			cfreq.append(dkey % 1000)
			dkey = int(dkey / 1000)
		assert len(cfreq) <= nact
		cw = []  # chunk weights, will store all possible weights for each chunk size, weights are multiples of nact
		cp = []  # chunk probabilities, stores probability of each weight, from binomonial distribution
		for i in range(len(cfreq)):
			weight = i+1
			nchunks = cfreq[i]   # number of chunks of size i+1
			x = np.arange(nchunks+1)
			chunk_weights = x * weight - (nchunks - x) * weight  # positive weights - negative weights
			chunk_probabilities = binom.pmf(x, nchunks, 0.5)  # binomonial coefficients, give number of possible ways to select x items
			cw.append(chunk_weights)
			cp.append(chunk_probabilities)
		chunk_weights = cw[0]
		chunk_probabilities = cp[0]
		for i in range(1, len(cfreq)):
			chunk_weights = np.add.outer(cw[i],chunk_weights).flatten()
			chunk_probabilities = np.outer(cp[i],chunk_probabilities).flatten()
		# if target positive, to cause error, want sum of overlaps plus target <= 0
		perror_sum = np.dot(chunk_weights + nact <= 0, chunk_probabilities)
		# if target negative, to cause error, want sum of overlaps plus target > 0
		nerror_sum = np.dot(chunk_weights - nact > 0, chunk_probabilities)
		prob_sum = np.sum(chunk_probabilities)
		error_rate = (perror_sum + nerror_sum) / (2* prob_sum)  # divide by 2 because positive and negative errors summed
		return error_rate

	def cop_err_empirical(self, nact, key, trials=100000):
		# perform multiple trials to calculate empirical error, used to compare with perdicted error
		cfreq=[]
		dkey = int(key / 1000)  # strip off no overlap count
		if dkey == 0:
			return 0.0  # no overlap, so error rate is zero
		while dkey > 0:
			cfreq.append(dkey % 1000)
			dkey = int(dkey / 1000)
		assert len(cfreq) <= nact
		weights = np.arange(1,len(cfreq)+1)  # will be like: 1,2,3,4,5
		sw = np.repeat(weights, cfreq) # like: [1,1,1,1,1,2,2,3,3,3,4,4,5] if cfreq=[5,2,3,2,1]
		mul = np.random.choice([-1, 1], size=(trials, len(sw)))
		sums = np.matmul(mul, sw)
		pfail = np.count_nonzero(sums + nact <= 0)
		nfail = np.count_nonzero(sums - nact > 0)
		error_count = pfail + nfail
		trial_count = 2 * trials
		error_rate = error_count / trial_count
		return error_rate


	def test_cop_err(self, ntrials=10):
		max_overlaps = np.array([20, 4, 6, 5, 3], dtype=np.int8)
		error_rates = np.zeros(len(max_overlaps), dtype=float)
		for nact in range(1, len(max_overlaps) + 1):
			trial_count = 0
			error_count = 0
			while trial_count < ntrials:
				cfreq = np.random.randint(max_overlaps[0:nact]+1, high=None)
				if(sum(cfreq)==0):
					continue
				key = 0
				for i in range(nact):
					key = key * 1000 + cfreq[nact - i - 1]
				key = key * 1000   # shift so no overlaps takes first three digits in key
				predicted_error_rate = self.cop_err(nact, key)
				found_error_rate = self.cop_err_empirical(nact, key)
				if predicted_error_rate == found_error_rate:
					percent_difference = "None"
				else:
					percent_difference = round(abs(predicted_error_rate - found_error_rate)* 100 /
					   ((predicted_error_rate + found_error_rate)/2), 2)
				print("nact=%s, key=%s, error predicted=%s, found=%s, difference=%s%%" % (
					nact, key, predicted_error_rate, found_error_rate, percent_difference))
				trial_count += 1

	def compute_hamming_dist(self, ncols):
		# compute distribution of probability of each hamming distance
		n = ncols
		chunk_probabilities = self.chunk_probabilities
		pov = np.empty(len(chunk_probabilities), dtype=np.float64)  # probability of overlaps
		perr = np.empty(len(chunk_probabilities), dtype=np.float64) # probability of error for each pattern of overlap
		i = 0
		for key, info in chunk_probabilities.items():
			prob, err = info
			pov[i] = prob
			perr[i] = err
			i += 1
		assert len(pov) == len(perr)
		pmf = np.empty(n + 1)  # probability mass function
		for h in range(len(pmf)):  # hamming distance
			phk = binom.pmf(h, n, perr)
			pmf[h] = np.dot(phk, pov)
		print("hdist (pmf) sum is %s (should be close to 1)" % np.sum(pmf))
		# assert math.isclose(np.sum(pmf), 1.0), "hdist sum is not equal to 1, is: %s" % np.sum(pmf)
		self.ncols = ncols
		self.hdist = pmf

	def compute_overall_perr(self, d):
		# compute overall error, by integrating over all hamming distances
		# ncols is the number of columns in each row of the sdm
		# d is number of item in item memory
		n = self.ncols
		hdist = self.hdist
		h = np.arange(len(hdist))
		# self.plot(binom.pmf(h, n, 0.5), "distractor pmf", "hamming distance", "probability")
		# distractor_pmf = binom.pmf(h, n, 0.5)  # this tried to account for matching hammings, but does not work
		# match_hammings_area = (hdist / (hdist + distractor_pmf)) / n
		ph_corr = binom.sf(h, n, 0.5) ** (d-1)
		# ph_corr = (binom.sf(h, n, 0.5) + match_hammings_area) ** (d-1)
		# self.plot(ph_corr, "probability correct", "hamming distance", "fraction correct")
		# self.plot(ph_corr * hdist, "p_corr weighted by hdist", "hamming distance", "weighted p_corr")
		p_corr = np.dot(ph_corr, hdist)
		self.overall_perr = 1 - p_corr
		return self.overall_perr

	def p_error_binom(self, d):
		# compute error by integrating predicted distribution with distractors
		# should do same function as above
		match_hammings = self.hdist
		phds = np.arange(self.ncols+1)  # possible hamming distances
		distractor_hammings = binom.pmf(phds, self.ncols, 0.5)
		# if self.include_empirical:
		# 	self.plot(match_hammings, "match_hammings vs distractor_hammings", "hamming distance",
		# 		"relative frequency", label="match_hammings", data2=distractor_hammings, label2="distractor_hammings")
		num_distractors = d - 1
		dhg = 1.0 # fraction distractor hamming greater than match hamming
		p_corr = 0.0  # probability correct
		for k in phds:
			# compute p_correct if match and distractor hammings are the same
			p_corr_same_hamming = match_hammings[k] *distractor_hammings[k] * (match_hammings[k] / (match_hammings[k] + distractor_hammings[k]) ** num_distractors)
			# p_corr_same_hamming = match_hammings[k] *distractor_hammings[k] * (1-((distractor_hammings[k] / (match_hammings[k] + distractor_hammings[k])) ** num_distractors))
			# compute p_correct if match and distractor hammings are different
			dhg -= distractor_hammings[k]
			p_corr_different_hamming = match_hammings[k] * (dhg ** num_distractors)
			p_corr_combined = p_corr_same_hamming + p_corr_different_hamming
			p_corr += p_corr_combined
			# p_err += match_hammings[k] * (1.0 - dhg ** num_distractors)
		p_err = 1 - p_corr
		self.overall_perr_binom = p_err
		return p_err

	def p_error_binom_orig(self, d):
		# compute error by integrating predicted distribution with distractors
		# should do same function as above
		match_hammings = self.hdist
		phds = np.arange(self.ncols+1)  # possible hamming distances
		distractor_hammings = binom.pmf(phds, self.ncols, 0.5)
		# if self.include_empirical:
		# 	self.plot(match_hammings, "match_hammings vs distractor_hammings", "hamming distance",
		# 		"relative frequency", label="match_hammings", data2=distractor_hammings, label2="distractor_hammings")
		num_distractors = d - 1
		dhg = 1.0 # fraction distractor hamming greater than match hamming
		p_err = 0.0  # probability error
		for k in phds:
			dhg -= distractor_hammings[k]
			p_err += match_hammings[k] * (1.0 - dhg ** num_distractors)
		self.overall_perr_binom = p_err
		return p_err
