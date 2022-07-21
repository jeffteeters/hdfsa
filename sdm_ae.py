# Class to find sdm error analytically

import math
from scipy.stats import hypergeom
from scipy.stats import binom
from scipy.stats import norm
# from scipy.special import binom as binom_coef
# from scipy.stats import norm
import numpy as np
from fractions import Fraction
import matplotlib.pyplot as plt
# import copy

# import pprint
# pp = pprint.PrettyPrinter(indent=4)


class Sdm_error_analytical:

	# Compute error of recall vector from SDM and matching to item memory
	# Uses concept of 'cop', stands for:
	# "chunk overlay probability" - keeps track of multiple overlaps (chunks) onto target rows from the same item.
	# This is needed because multiple overlaps from the same item change the probability of error when computing
	# the sum.  For example, a chunk of size 2 would contribute -2 or +2 to the sum.  But two independent items
	# would contribute either (-2, 0, +2).  Another example, chunk of size 3 contributes -3 or +3.  But three
	# independent items could contribute: -3, -1, 1, or 3.

	def __init__(self, nrows, nact, k, ncols=None, d=None, match_method="both",
		threshold=10000000, show_pruning=False, show_items=False,
		show_progress=False, prune=False, prune_zeros=True, debug=False):
		# nrows - number of rows in sdm
		# nact - activaction count
		# k - number of items to store in sdm
		# ncols - number of columns in each row of sdm
		# d - size of item memory (d-1 are distractors)
		# match_method - "hamming" - match using hamming distance (threshold counters and use hamming match)
		#   "dot" - use dot product of counter sums for matching with item memory (non-thresholded sums)
		#   "both" - compute using both hamming and dot product 
		#    - False if match using hamming distance (threshold counters and use hamming match)
		# threshold - maximum ratio of largest to smallest probability.  Drop patterns that have smaller probability
		#  This done if prune=True, to limit number of patterns to only those that are contributing the most to the result
		# prune - True if should drop patterns that have smaller probability than maximum probability / threshold
		# prune_zeros - True if should drop patterns that have probability == 0.0 (due to underflow)
		self.nrows = nrows
		self.nact = nact
		self.k = k
		self.ncols = ncols
		self.d = d
		assert match_method in ("hamming", "dot", "both")
		self.hamm_match = match_method in ("hamming", "both")
		self.dotp_match = match_method in ("dot", "both")
		self.threshold = threshold
		self.show_pruning = show_pruning
		self.show_items = show_items
		self.prune = prune
		self.prune_zeros = prune_zeros
		self.debug = debug
		self.ov1_pmf = self.compute_one_item_overlap_pmf()
		if self.show_items:
			print("self.ov1_pmf=%s" % self.ov1_pmf)
		self.key_increments = self.compute_key_increments()
		self.cop_key = self.key_increments.copy()
		self.cop_prb = self.ov1_pmf.copy()
		for i in range(2, k):
			self.add_overlap(i)
			if show_progress:
				end = "\n" if i % 10 == 0 else ""
				print("%s-%s "%(i,len(self.cop_key)), end=end)
		if show_progress:
			print("\nNumber keys=%s.  Computing error_rates..." % len(self.cop_key))
		self.cop_err = np.array([self.cop_error_rate(self.nact, key) for key in self.cop_key])
		if self.dotp_match:
			assert self.ncols is not None, "if using dotp_match, must specify ncols"
			assert self.d is not None
			# self.compute_sump_distribution(self.ncols)
			self.compute_dot_product_perror(self.ncols, self.d)
		if ncols is not None:
			self.compute_hamming_dist()
			if d is not None:
				self.compute_overall_perr()
				# perr_fraction is leveling out when error gets small.  When running plot_sdm_error_vs_dims:
				# starting 4 (10e-5 error)
				# warning, in sdmae, perror=9.953150801877975e-06, perr_fraction=2.0241882869370637e-05
				# starting 5 (10e-6 error)
				# warning, in sdmae, perror=9.686530652031067e-07, perr_fraction=1.1265137007996028e-05
				# starting 6
				# warning, in sdmae, perror=1.0220631629920263e-07, perr_fraction=1.0753236151307854e-05
				# self.compute_overall_perr_fraction()
				# if not math.isclose(self.perr, self.perr_fraction):
				# 	print("warning, in sdmae, perror=%s, perr_fraction=%s" % (self.perr, self.perr_fraction))
		if show_progress:
			self.display_result()

	def compute_one_item_overlap_pmf(self):
		max_num_first_terms = 3  # limit to first 3 terms.  Assume terms beyond that (6 or more overlaps) probability too small)
		if self.nrows == 1:
			assert self.nact == 1, "If nrows is 1, nact must be 1"
			return np.array([1.0,], dtype=np.float64)  # special case
		# based on: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html#scipy.stats.hypergeom
		[M, n, N] = [self.nrows, self.nact, self.nact]
		rv = hypergeom(M, n, N)
		x = np.arange(0, min(n+1, max_num_first_terms))
		pmf = rv.pmf(x)
		if self.debug:
			print("One item overlap pmf=%s" % pmf)
		return pmf

	def compute_key_increments(self):
		# keys in the self.chunk_probabilities dictionary are integers with each three digits representing the
		# overlap count for 0 through nact overlaps.  Least significant digits are used for count of 0 overlaps.
		# In other words, keys look like: 444333222111000, where 000 is the count of 0 overlaps, and 111 is the
		# count of 1 overlaps.  The key increment array has the numbers to add to the previous key to convert it
		# to the new key.
		if self.nrows == 1:
			# special case
			return(np.array([1000,], dtype=np.uint))
		# print("computing key increments, nact=%s, len(ov1_pmf)=%s" % (self.nact, len(self.ov1_pmf)))
		key_increments=np.empty(len(self.ov1_pmf), dtype=np.ulonglong)
		ki = 1
		for i in range(len(self.ov1_pmf)):
			# print("i=%s, storing %s" % (i, ki))
			key_increments[i] = ki
			ki = ki * 1000
		# print("key_increments=%s" % key_increments)
		return key_increments

	def add_overlap(self, iteration):
		# givin current cop_key, cop_prob, update it by multiplying each probability by every probability in
		# self.ov1_pmf and then combine terms.  Remove any terms with probability less than threshold
		assert self.cop_key.size == self.cop_prb.size, "starting add overlap, len(cop_key)=%s, len(cop_prb)=%s" % (
			self.cop_key.size, self.cop_prb.size)
		self.cop_key = np.add.outer(self.key_increments, self.cop_key).flatten()
		self.cop_prb = np.outer(self.ov1_pmf, self.cop_prb).flatten()
		assert self.cop_key.size == self.cop_prb.size, "after doing outer, len(cop_key)=%s, len(cop_prb)=%s" % (
			self.cop_key.size, self.cop_prb.size)
		# make array of bin numbers for using bincount to combine probabilities that have the same key
		# from: https://stackoverflow.com/questions/3403973/fast-replacement-of-values-in-a-numpy-array
		# Method method_searchsort(), in post by Jean Lescut
		if self.debug:
			print("before combine: len(cop_key)=%s" % len(self.cop_key))
			print("cop_key=%s" % self.cop_key)
			print("cop_prb=%s" % self.cop_prb)
		keys = np.unique(self.cop_key)
		bins = np.arange(keys.size)
		sort_idx = np.argsort(keys)
		idx = np.searchsorted(keys, self.cop_key, sorter=sort_idx)
		out = bins[sort_idx][idx]
		self.cop_prb = np.bincount(out, weights=self.cop_prb, minlength=len(bins))
		self.cop_key = keys[bins]
		if self.debug:
			print("after combine: len(cop_key)=%s" % len(self.cop_key))
			print("cop_key=%s" % self.cop_key)
			print("cop_prb=%s" % self.cop_prb)
		assert self.cop_key.size == self.cop_prb.size, "after combining like keys, len(cop_key)=%s, len(cop_prb)=%s" % (
			self.cop_key.size, self.cop_prb.size)
		if self.prune_zeros:
			mask = self.cop_prb > 0.0  # keep terms larger than zero
			self.cop_key = self.cop_key[mask]
			self.cop_prb = self.cop_prb[mask]
			if self.show_pruning:
				number_pruned = mask.size - self.cop_key.size
				if number_pruned > 0:
					print("Pruned_zeros: %s before, %s after, %s pruned" % (mask.size, self.cop_key.size, number_pruned))
		# prune terms with probability smaller than threshold
		if not self.prune:
			return
		max_prb = self.cop_prb.max()
		thrsh = max_prb / self.threshold
		mask = self.cop_prb > thrsh  # keep terms larger than threshold
		self.cop_key = self.cop_key[mask]
		self.cop_prb = self.cop_prb[mask]
		if self.show_pruning:
			number_pruned = mask.size - self.cop_key.size
			if number_pruned > 0:
				print("Pruning: %s before, %s after, %s pruned" % (mask.size, self.cop_key.size, number_pruned))
		assert self.cop_key.size == self.cop_prb.size, "after pruning, len(cop_key)=%s, len(cop_prb)=%s" % (
			self.cop_key.size, self.cop_prb.size)

	def display_result(self):
		num_items = len(self.cop_key)
		total_prob = sum(self.cop_prb)
		print("After %s items stored with nact=%s, threshold=%s, size cop = %s, total_probability = %s" % (self.k,
			self.nact, self.threshold, num_items, total_prob))
		if self.show_items:
			max_num_to_show = 10  # actually show twice this many
			print("items are:")
			for i in range(min(max_num_to_show, len(self.cop_key))):
				print("%s -> %s,%s" % (self.cop_key[i], self.cop_prb[i], self.cop_err[i]))
			if len(self.cop_key) > max_num_to_show:
				print("...")
			for i in range(len(self.cop_key) - max_num_to_show, len(self.cop_key)):
				print("%s -> %s,%s" % (self.cop_key[i], self.cop_prb[i], self.cop_err[i]))
			# self.test_cop_err()  # uncomment to test cop_error_rate routine

	def cop_error_rate(self, nact, key):
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
		cop_error_rate = (perror_sum + nerror_sum) / (2* prob_sum)  # divide by 2 because positive and negative errors summed
		return cop_error_rate

	def compute_chunk_weights_and_probabilities(self, key):
		# determine weights and probabilities given chunk overlap pattern specified in key
		nact = self.nact
		dkey = int(key / 1000)  # strip off no overlap count
		if dkey == 0:
			# no overlap, so weight of overlap is zero with probability 1
			chunk_weights = np.array([0],dtype=int)
			chunk_probabilities = np.array([1.0],dtype=np.float64)
			return (chunk_weights, chunk_probabilities)
		cfreq=[]
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
		return (chunk_weights, chunk_probabilities)

	def pv_union(self, p1, v1, p2, v2):
		# combine array pairs (p1, v1) with (p2, v2)
		# p1 and p2 are arrys of probabilities
		# v1 and v2 are arrays of the corresponding values (integers)
		# find values that are common between v1 and v2, and add the probabilities
		assert len(p1) == len(v1)
		assert len(p2) == len(v2)
		assert len(np.unique(v1)) == len(v1), "duplicate values in v1"
		assert len(np.unique(v2)) == len(v2), "duplicate values in v2"
		cv, cv1, cv2 = np.intersect1d(v1, v2, assume_unique=True, return_indices=True)
		new_len = len(v1) + len(v2) - len(cv)
		new_p = np.empty(new_len, dtype=np.float64)
		new_v = np.empty(new_len, dtype=np.int32)
		new_p[0:len(p1)] = p1
		new_v[0:len(v1)] = v1
		new_p[cv1] += p2[cv2]  # add in probabilities for common values
		v2add = np.setdiff1d(np.arange(len(v2)), cv2)  # get indicies of values to add (not common)
		assert len(v2add) + len(v1) == new_len, "lengths are not as expected"
		new_p[len(p1):] = p2[v2add]
		new_v[len(v1):] = v2[v2add]
		return (new_p, new_v)

	def pv_stats(self, p1, v1):
		# return mean and variance of discrete distribution with probabilities given in p1 and values given in p2
		# from equations under "Discrete random variable" at: https://en.wikipedia.org/wiki/Variance
		assert math.isclose(p1.sum(), 1.0)
		assert len(p1) == len(v1)
		mean = np.dot(p1, v1)
		variance = np.dot(p1, (v1-mean)**2)
		return (mean, variance)

	def dot_product_stats(self, mean1, var1, ncols):
		# compute new mean and variance (var) found by central limit theorm, summing values ncols times
		# this done to form mean and variance for dot product with vectors length ncols
		mean_sum = mean1 * ncols
		var_sum = var1 * ncols
		return (mean_sum, var_sum)

	def compute_dot_product_perror(self, ncols, d):
		# compute overall perror assuming counter sums are not thresholded and match to item
		# memory made using dot product.
		# ncols - width of vector, included so can call with different widths
		# d- number of items in item memory, so there are d-1 distractors 
		assert self.cop_key.size == self.cop_prb.size
		perr = 0.0
		for i in range(self.cop_key.size):
			cw, cp = self.compute_chunk_weights_and_probabilities(self.cop_key[i])
			assert math.isclose(cp.sum(), 1.0), "cp sum is not one (is %s) for i=%s, key=%s" % (
				cp.sum(), i, self.cop_key[i])
			# compute sum then product with target bit (+1 or -1) for match
			positive_target_sump = cw + self.nact  # same as (cs + nact) * 1
			negative_target_sump = self.nact - cw  # same as (cw - nact) * -1
			half_cp = cp / 2   # weight probability by half for positive target and half for negative target
			match_p, match_w = self.pv_union(half_cp, positive_target_sump, half_cp, negative_target_sump)
			match_mean1, match_var1 = self.pv_stats(match_p, match_w)
			match_mean_dot, match_var_dot = self.dot_product_stats(match_mean1, match_var1, ncols)
			# compute for distractor
			qtr_cp = cp / 4
			positive_target_sump_no_match = -positive_target_sump
			negative_target_sump_no_match = -negative_target_sump
			no_match_p, no_match_w = self.pv_union(qtr_cp, positive_target_sump_no_match, qtr_cp, negative_target_sump_no_match)
			# add in probability and match for match (via chance, one half)
			distactor_p, distractor_w = self.pv_union(match_p / 2.0, match_w, no_match_p, no_match_w)
			distractor_mean1, distractor_var1 = self.pv_stats(distactor_p, distractor_w)
			distractor_mean_dot, distractor_var_dot = self.dot_product_stats(distractor_mean1, distractor_var1, ncols)
			# print("match mean = %s, var=%s; distractor mean=%s, var=%s" % (match_mean_dot, match_var_dot,
			# 	distractor_mean_dot, distractor_var_dot))
			perr_part = self.compute_dot_product_perror_from_distributions(match_mean_dot, match_var_dot,
				distractor_mean_dot, distractor_var_dot, d)
			perr += perr_part * self.cop_prb[i]
		self.perr_dot = perr


	def compute_dot_product_perror_from_distributions(self, match_mean, match_var,
				distractor_mean, distractor_var, d):
		# d-1 is number of distractors
		sfactor = 6.0  # number of standard deviations from mean
		lin_steps = 10000
		match_std = math.sqrt(match_var)
		distractor_std = math.sqrt(distractor_var)
		if match_std == 0.0:
			match_std = 10.0  # have at least some variance.  Should maybe have another why to fix this
		# assert match_std > 0
		assert distractor_std > 0
		low_limit = min(match_mean - sfactor * match_std, distractor_mean - sfactor * distractor_std)
		high_limit = max(match_mean + sfactor * match_std, distractor_mean + sfactor * distractor_std)
		x = np.linspace(low_limit, high_limit, lin_steps)
		match_pdf = norm.pdf(x, loc=match_mean, scale=match_std)
		distractor_cdfd = norm.cdf(x, loc=distractor_mean, scale=distractor_std)**(d-1)
		p_corr = np.dot(match_pdf, distractor_cdfd)
		perr = 1 - p_corr
		return perr

	def compute_sump_distribution(self, ncols):
		# compute matching dot product distribution (from sums of counters multiplied by target value in item memory)
		# returned array used for computing error rate with dot product
		# ncols - width of vector, included so can call with different widths 
		assert self.cop_key.size == self.cop_prb.size
		# print("k=%s" % self.k)
		# print("cop_key=%s" % self.cop_key)
		# print("cop_prb=%s" % self.cop_prb)
		# print("sum cop_prb=%s" % self.cop_prb.sum())
		max_dotp = self.nact * ncols # assume this is maximum value of dot product (might not be true, try it)
		sump_dist_len = 2 * max_dotp + 1   # zero value is at index max_dotp
		sump_dist = np.zeros(sump_dist_len, dtype=np.float64)
		cop_prob_sum = 0.0;
		term_count = 0
		# import pdb; pdb.set_trace()
		for i in range(self.cop_key.size):
			cw, cp = self.compute_chunk_weights_and_probabilities(self.cop_key[i])
			term_count += len(cp)
			assert math.isclose(cp.sum(), 1.0), "cp sum is not one (is %s) for i=%s, key=%s" % (
				cp.sum(), i, self.cop_key[i])
			# compute sum then product with target bit (+1 or -1)
			positive_target_sump = cw + self.nact  # same as (cs + nact) * 1
			negative_target_sump = self.nact - cw  # same as (cw - nact) * -1
			cop_prob_sum += self.cop_prb[i]
			half_cp = self.cop_prb[i] * cp / 2   # weight probability by half for positive target and half for negative target
			sump_dist[positive_target_sump + max_dotp] += half_cp
			sump_dist[negative_target_sump + max_dotp] += half_cp
		print("cop_prob_sum=%s, term_count=%s" % (cop_prob_sum, term_count))
		assert math.isclose(np.sum(sump_dist), 1.0), "sump_dist is not equal to 1, is: %s" % np.sum(sump_dist)
		# trim sump_dist to only have non-zero values
		sdnz = sump_dist.nonzero()
		first_non_zero = sdnz[0][0]
		last_non_zero = sdnz[0][-1]
		left_side_len = max_dotp - first_non_zero
		right_side_len = last_non_zero - max_dotp
		new_side_len = max(left_side_len, right_side_len)
		sump_dist = sump_dist[max_dotp - new_side_len: max_dotp + new_side_len + 1]
		self.sump_dist = sump_dist
		print("after trimming sump_dist, len before=%s, len after=%s" % (sump_dist_len, sump_dist.size))
		plt.plot(sump_dist, label="match distribution")
		self.compute_overall_perr_dot()
		plt.legend(loc='upper left')
		plt.show()



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
				predicted_error_rate = self.cop_error_rate(nact, key)
				found_error_rate = self.cop_err_empirical(nact, key)
				if predicted_error_rate == found_error_rate:
					percent_difference = "None"
				else:
					percent_difference = round(abs(predicted_error_rate - found_error_rate)* 100 /
					   ((predicted_error_rate + found_error_rate)/2), 2)
				print("nact=%s, key=%s, error predicted=%s, found=%s, difference=%s%%" % (
					nact, key, predicted_error_rate, found_error_rate, percent_difference))
				trial_count += 1


	def compute_hamming_dist(self, ncols=None):
		# compute distribution of probability of each hamming distance
		if ncols is None:
			assert self.ncols is not None, "must specify ncols to compute_hamming_dist"
			ncols = self.ncols
		else:
			self.ncols = ncols  # save passed in ncols
		# print("start compute_hamming_dist for ncols=%s" % ncols)
		hdist = np.empty(ncols + 1)  # probability mass function
		for h in range(len(hdist)):  # hamming distance
			phk = binom.pmf(h, ncols, self.cop_err)
			hdist[h] = np.dot(phk, self.cop_prb)
		# print("hdist (pmf) sum is %s (should be close to 1)" % np.sum(hdist))
		# assert math.isclose(np.sum(pmf), 1.0), "hdist sum is not equal to 1, is: %s" % np.sum(pmf)
		self.hdist = hdist

	def compute_overall_perr(self, d=None):
		# compute overall error rate, by integrating over all hamming distances with distractor distribution
		# ncols is the number of columns in each row of the sdm
		# d is number of item in item memory
		if d is None:
			assert self.d is not None, "Must specify d to compute_overall_perr"
			d = self.d
		else:
			self.d = d  # save passed in d
		n = self.ncols
		hdist = self.hdist
		h = np.arange(len(hdist))
		self.distractor_pmf = binom.pmf(h, n, 0.5)
		# self.plot(binom.pmf(h, n, 0.5), "distractor pmf", "hamming distance", "probability")
		self.ph_corr = binom.sf(h, n, 0.5) ** (d-1)
		# ph_corr = binom.sf(h, n, 0.5) ** (d-1)
		# ph_corr = (binom.sf(h, n, 0.5) + match_hammings_area) ** (d-1)
		# self.plot(ph_corr, "probability correct", "hamming distance", "fraction correct")
		# self.plot(ph_corr * hdist, "p_corr weighted by hdist", "hamming distance", "weighted p_corr")
		hdist = hdist / np.sum(hdist)  # renormalize to increase due to loss of terms
		p_corr = np.dot(self.ph_corr, hdist)
		self.perr = 1 - p_corr
		return self.perr

	def compute_overall_perr_dot(self, d=None):
		# for dot product match to item memory
		# compute overall error rate, by integrating over all hamming distances with distractor distribution
		# ncols is the number of columns in each row of the sdm
		# d is number of item in item memory
		if d is None:
			assert self.d is not None, "Must specify d to compute_overall_perr_dot"
			d = self.d
		else:
			self.d = d  # save passed in d
		n = self.ncols
		hdist = self.sump_dist
		h = np.arange(len(hdist))
		zero_index = (hdist.size - 1) / 2
		dstd = math.sqrt(self.ncols)  # distractor standard deviation 
		distractor_pdf = norm.pdf(h, loc=zero_index, scale=math.sqrt(self.ncols))    #  binom.pmf(h, n, 0.5)
		plt.plot(distractor_pdf, label="distractor pdf")
		ph_corr = norm.sf(h, loc=zero_index, scale=dstd) ** (d-1)  # binom.sf(h, n, 0.5) ** (d-1)
		# ph_corr = binom.sf(h, n, 0.5) ** (d-1)
		# ph_corr = (binom.sf(h, n, 0.5) + match_hammings_area) ** (d-1)
		# self.plot(ph_corr, "probability correct", "hamming distance", "fraction correct")
		# self.plot(ph_corr * hdist, "p_corr weighted by hdist", "hamming distance", "weighted p_corr")
		hdist = hdist / np.sum(hdist)  # renormalize to increase due to loss of terms
		p_corr = np.dot(ph_corr, hdist)
		self.perr_dot = 1 - p_corr
		# print("perr_dot=%s" % self.perr_dot)
		return self.perr_dot

	def compute_overall_perr_fraction(self, d=None):
		# computer overall probability error using python Fraction module to prevent floating point underflow
		if d is None:
			assert self.d is not None, "Must specify d to p_error_Fraction"
			d = self.d
		else:
			self.d = d  # save passed in d
		wl = self.ncols   # word length
		num_distractors = d - 1
		pdist = Fraction(1, 2) # probability of distractor bit matching
		match_hamming = []
		distractor_hamming = []
		for k in range(wl+1):
			match_hamming.append(Fraction(self.hdist[k]))  # convert to fraction
			ncomb = Fraction(math.comb(wl, k))
			# match_hamming.append(ncomb * pflip ** k * (1-pflip) ** (wl-k))
			distractor_hamming.append(ncomb * pdist ** k * (1-pdist) ** (wl-k))
		# print("sum match_hamming=%s, distractor_hamming=%s" % (sum(match_hamming), sum(distractor_hamming)))
		# if self.env.pvals["show_histograms"]:
		# 	fig = Figure(title="match and distractor hamming distributions for %s%% bit flips" % xval,
		# 		xvals=range(wl), grid=False,
		# 		xlabel="Hamming distance", ylabel="Probability", xaxis_labels=None,
		# 		legend_location="upper right",
		# 		yvals=match_hamming, ebar=None, legend="match_hamming", fmt="-g")
		# 	fig.add_line(distractor_hamming, legend="distractor_hamming", fmt="-m")
		# 	self.figures.append(fig)
		dhg = Fraction(1) # fraction distractor hamming greater than match hamming
		pcor = Fraction(0)  # probability correct
		for k in range(wl+1):
			dhg -= distractor_hamming[k]
			pcor += match_hamming[k] * dhg ** num_distractors
		# error = (float((1 - pcor) * 100))  # convert to percent
		error = float(1 - pcor) # * 100))  # convert to percent
		self.perr_fraction = error
		return error
		# return float(pcor)

	def ae(nrows, ncols, nact, k, d):
		# compute analytical error rate
		# Class function to enable computing error rate with one call
		sae = Sdm_error_analytical(nrows, nact, k, ncols, d)
		return sae.perr

def main():
	# nrows = 6; nact = 2; k = 5; d = 27; ncols = 33  # original test case
	# nrows=80; nact=3; k=50; d=27; ncols=33  # gives 0.0178865?
	# ---
	# test with d==2:
	# nrows=65; nact=1; k=1000; d=2; ncols=512  # compare to predicted from simple_predict_sizes.py
	# for k=1000, d=2, sdm size=(65, 512, 1), predicted analytical error=0.001263169
	# matches:
	# python fast_sdm_empirical.py
	# With nrows=65, ncols=512, nact=1, threshold_sum=True, only_one_distractor=True, epochs=100, mean_error=0.00127, std_error=0.00111
	# ---
	# nrows = 80; nact = 6; k = 1000; d = 27; ncols = 51  # near full size
	# nrows = 1; nact = 1; k = 3; d = 3; ncols = 3  # test for understand match hamming
	# test new cop class
	# cop = Cop(nrows, nact, k)
	# return

	# nrows = 2; nact = 2; k = 2; d = 27; ncols = 33 	# test smaller with overlap all the time
	# nrows = 80; nact = 3; k = 300; d = 27; ncols = 51  # near full size
	# nrows=39; nact=1; k=1000; d=100; ncols=512;  # should give 10^-1 error if using dot product match
	nrows=76; nact=1; k=1000; d=100; ncols=512;  # should give 10^-3 error if using dot product match
	sea = Sdm_error_analytical(nrows, nact, k, ncols=ncols, d=d)
	# ae = Sdm_error_analytical.ae(nrows, ncols, nact, k, d)
	print("for k=%s, d=%s, sdm size=(%s, %s, %s), perr=%s, perr_dot=%s" % (k, d, nrows, ncols, nact, sea.perr,
		sea.perr_dot))

	# ae = Sdm_error_analytical(nrows, nact, k)
	# ae.error_rate
	# nrows, nact, k, ncols=None, d=
	# ae.perr()


	# predicted_using_theory_dist = ov.p_error_binom() # ov.compute_overall_perr()
	# predicted_using_empirical_dist = ov.p_error_binom(use_empirical=True)
	# predicted_using_cop = ov.cop.overall_perr
	# predicted_using_cop_binom = ov.cop.overall_perr_binom
	# # overall_perr = Ovc.compute_overall_error(nrows, ncols, nact, k, d)
	# empirical_err = ov.emp_overall_perr
	# print("for k=%s, d=%s, sdm size=(%s, %s, %s), predicted_using_theory_dist=%s, predicted_using_empirical_dist=%s,"
	# 	" predicted_using_cop=%s, predicted_using_cop_binom=%s, empirical_err=%s" % (k, d, nrows, ncols, nact, predicted_using_theory_dist,
	# 		predicted_using_empirical_dist, predicted_using_cop, predicted_using_cop_binom, empirical_err))
	# ncols = 52
	# overall_perr = Ovc.compute_overall_error(nrows, ncols, nact, k)
	# print("for k=%s, sdm size=(%s, %s, %s), overall_perr=%s" % (k, nrows, ncols, nact, overall_perr))


if __name__ == "__main__":
	main()
