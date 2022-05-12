# Class to find sdm error analytically

import math
from scipy.stats import hypergeom
from scipy.stats import binom
# from scipy.special import binom as binom_coef
# from scipy.stats import norm
import numpy as np
# import matplotlib.pyplot as plt
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

	def __init__(self, nrows, nact, k, ncols=None, d=None, threshold=10000, show_pruning=False, show_items=False):
		# nrows - number of rows in sdm
		# nact - activaction count
		# k - number of items to store in sdm
		# ncols - number of columns in each row of sdm
		# d - size of item memory (d-1 are distractors)
		# threshold - maximum ratio of largest to smallest probability.  Drop patterns that have smaller probability
		#  This done to limit number of patterns to only those that are contributing the most to the result
		self.nrows = nrows
		self.nact = nact
		self.k = k
		self.ncols = ncols
		self.d = d
		self.threshold = threshold
		self.show_pruning = show_pruning
		self.show_items = show_items
		self.ov1_pmf = self.compute_one_item_overlap_pmf()
		# print("self.ov1_pmf=%s" % self.ov1_pmf)
		self.key_increments = self.compute_key_increments()
		self.cop_key = self.key_increments.copy()
		self.cop_prb = self.ov1_pmf.copy()
		for i in range(2, k):
			self.add_overlap(i)
			end = "\n" if i % 10 == 0 else ""
			print("%s-%s "%(i,len(self.cop_key)), end=end)
		print("\nNumber keys=%s.  Computing error_rates..." % len(self.cop_key))
		self.cop_err = np.array([self.cop_error_rate(self.nact, key) for key in self.cop_key])
		if ncols is not None:
			self.compute_hamming_dist()
			if d is not None:
				self.compute_overall_perr()
		self.display_result()

	def compute_one_item_overlap_pmf(self):
		# based on: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html#scipy.stats.hypergeom
		[M, n, N] = [self.nrows, self.nact, self.nact]
		rv = hypergeom(M, n, N)
		x = np.arange(0, n+1)
		pmf = rv.pmf(x)
		return pmf

	def compute_key_increments(self):
		# keys in the self.chunk_probabilities dictionary are integers with each three digits representing the
		# overlap count for 0 through nact overlaps.  Least significant digits are used for count of 0 overlaps.
		# In other words, keys look like: 444333222111000, where 000 is the count of 0 overlaps, and 111 is the
		# count of 1 overlaps.  The key increment array has the numbers to add to the previous key to convert it
		# to the new key.
		key_increments=np.empty(self.nact+1, dtype=np.uint)
		ki = 1
		for i in range(self.nact+1):
			key_increments[i] = ki
			ki = ki * 1000
		print("key_increments=%s" % key_increments)
		return key_increments

	def add_overlap(self, iteration):
		# givin current cop_key, cop_prob, update it by multiplying each probability by every probability in
		# self.ov1_pmf and then combine terms.  Remove any terms with probability less than threshold
		self.cop_key = np.add.outer(self.key_increments, self.cop_key).flatten()
		self.cop_prb = np.outer(self.ov1_pmf, self.cop_prb).flatten()
		# make array of bin numbers for using bincount to combine probabilities that have the same key
		# from: https://stackoverflow.com/questions/3403973/fast-replacement-of-values-in-a-numpy-array
		# Method method_searchsort(), in post by Jean Lescut
		# print("before combine: len(cop_key)=%s" % len(self.cop_key))
		# print("cop_key=%s" % self.cop_key)
		# print("cop_prb=%s" % self.cop_prb)
		keys = np.unique(self.cop_key)
		bins = np.arange(keys.size)
		sort_idx = np.argsort(keys)
		idx = np.searchsorted(keys, self.cop_key, sorter=sort_idx)
		out = bins[sort_idx][idx]
		self.cop_prb = np.bincount(out, weights=self.cop_prb, minlength=len(bins))
		self.cop_key = keys[bins]
		# print("after combine: len(cop_key)=%s" % len(self.cop_key))
		# print("cop_key=%s" % self.cop_key)
		# print("cop_prb=%s" % self.cop_prb)
		# import pdb; pdb.set_trace()

	def display_result(self):
		num_items = len(self.cop_key)
		total_prob = sum(self.cop_prb)
		print("After %s items stored with nact=%s, threshold=%s, size cop = %s, total_probability = %s" % (self.k,
			self.nact, self.threshold, num_items, total_prob))
		if self.show_items:
			max_num_to_show = 50
			print("items are:")
			for i in range(min(max_num_to_show, len(self.cop_key))):
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
		print("start compute_hamming_dist for ncols=%s" % ncols)
		hdist = np.empty(ncols + 1)  # probability mass function
		for h in range(len(hdist)):  # hamming distance
			phk = binom.pmf(h, ncols, self.cop_err)
			hdist[h] = np.dot(phk, self.cop_prb)
		print("hdist (pmf) sum is %s (should be close to 1)" % np.sum(hdist))
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
		# self.plot(binom.pmf(h, n, 0.5), "distractor pmf", "hamming distance", "probability")
		# distractor_pmf = binom.pmf(h, n, 0.5)  # this tried to account for matching hammings, but does not work
		# match_hammings_area = (hdist / (hdist + distractor_pmf)) / n
		ph_corr = binom.sf(h, n, 0.5) ** (d-1)
		# ph_corr = (binom.sf(h, n, 0.5) + match_hammings_area) ** (d-1)
		# self.plot(ph_corr, "probability correct", "hamming distance", "fraction correct")
		# self.plot(ph_corr * hdist, "p_corr weighted by hdist", "hamming distance", "weighted p_corr")
		p_corr = np.dot(ph_corr, hdist)
		self.perr = 1 - p_corr
		return self.perr

	def ae(nrows, ncols, nact, k, d):
		# compute analytical error rate
		# Class function to enable computing error rate with one call
		sae = Sdm_error_analytical(nrows, nact, k, ncols, d)
		return sae.perr

def main():
	# nrows = 6; nact = 2; k = 5; d = 27; ncols = 33  # original test case
	nrows=80; nact=3; k=50; d=27; ncols=33  # gives 0.0178865
	# nrows = 80; nact = 6; k = 1000; d = 27; ncols = 51  # near full size
	# nrows = 1; nact = 1; k = 3; d = 3; ncols = 3  # test for understand match hamming
	# test new cop class
	# cop = Cop(nrows, nact, k)
	# return

	# nrows = 2; nact = 2; k = 2; d = 27; ncols = 33 	# test smaller with overlap all the time
	# nrows = 80; nact = 3; k = 300; d = 27; ncols = 51  # near full size
	ae = Sdm_error_analytical.ae(nrows, ncols, nact, k, d)
	print("for k=%s, d=%s, sdm size=(%s, %s, %s), predicted analytical error=%s" % (k, d, nrows, ncols, nact, ae))

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
