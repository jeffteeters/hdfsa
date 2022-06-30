# Class to find sdm error analytically

import math
from scipy.stats import hypergeom
from scipy.stats import binom
# from scipy.special import binom as binom_coef
# from scipy.stats import norm
import numpy as np
from fractions import Fraction
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

	def __init__(self, nrows, nact, k, ncols=None, d=None, threshold=100000, show_pruning=False, show_items=False,
		show_progress=False, prune=True, debug=False):
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
		self.prune = prune
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
