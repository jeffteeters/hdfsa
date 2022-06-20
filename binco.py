# script to calculate delta (normalized hamming distance) when there are
# coordinated overlaps (onto multiple counters) of a binarized sdm.
# this is replacing the calculation of pvote:
#    pvote = np.sum(np.prod(pc_er[mpc], axis=1))
# in binarized_sdm_analytical.py

import itertools
import numpy as np
# import copy
from scipy.special import comb
from scipy.stats import binom

class Binco:

	# calculate delta (normalized hamming distance) when there are
	# coordinated overlaps (onto multiple counters) of a binarized sdm.

	def __init__(self, nrows, nact, k, novl, epochs=10, debug=False):
		# nrows - number of rows in sdm
		# nact - activaction count
		# k - number of items (transitions) to store in sdm
		# novl - int arry, stores number of overlaps (including target) of each selected row
		# epochs - used for testing
		assert nact > 1
		assert novl.size == nact
		self.nrows = nrows
		self.nact = nact
		self.k = k
		self.novl = novl
		self.epochs = epochs
		self.debug = debug
		rng = np.random.default_rng()
		self.co_probs = self.calculate_co_probs()  # calculate probabilities of coordinated overlaps
		# print("co_probs=%s" % self.co_probs)
		son = novl.copy()   # single overlap number.  Will be modified when coordinated overlaps added
		nco_added = 0       # number coordinated overlaps added
		ave_num_overlaps = k * nact / nrows
		max_num_coordinated_overlaps = max(10, 2*round(ave_num_overlaps))
		co_ptr = np.zeros((nact-1, max_num_coordinated_overlaps), dtype = np.uint16)  # has index of values in covals
		covals = np.zeros( (nact-1)* max_num_coordinated_overlaps, dtype =np.int8)  # has values of coordinated overlaps, +-1
		num_covals = 0  # number coordinated overlaps added (length of valid covals)
		num_co_ptr = np.zeros(nact-1, dtype=np.uint16)  # has number of values in each row of co_ptr

	def calculate_co_probs(self):
		co_probs = np.empty(self.nact-1, dtype=np.float64)
		for i in range(self.nact-1):
			ovl = i + 2
			co_probs[i] = self.predict_ratio_binom(ovl)
		return co_probs

	def predict_ratio_binom(self, ovl):
		# ovl is number of coordinated overlaps, 2 <= ovl <= nact
		# find number of coordinated overlaps of this size stored
		# use binonomial
		assert ovl >= 2 and ovl <= self.nact
		nstored = self.k * comb(self.nact, ovl)
		# ave_overlaps_per_row = (self.k * self.nact / self.nrows) - 1
		p_overlap_per_item = self.nact / self.nrows
		ave_overlaps_per_row = self.expected_min_multiple_binonomial(self.k-1, ovl, p = p_overlap_per_item)
		if ave_overlaps_per_row <= 0:
			ave_overlaps_per_row += 1
		npossible = comb(self.nrows, ovl) * ave_overlaps_per_row
		ratio = nstored / npossible
		return ratio

	def expected_min_multiple_binonomial(self, ntrials_in_each_binomial, num_binomials, p = 0.5):
		# return expected minimum of multiple binonomials
		# from: https://www.quora.com/What-is-the-average-lowest-value-from-N-random-values-from-the-binomial-distribution
		n = ntrials_in_each_binomial
		big_n = num_binomials
		x = np.arange(n)
		eMin = np.sum( (1 - binom.cdf(x, n, p)) ** big_n )
		return eMin


def main():
	# nrows = 6; nact = 2; k = 5; d = 27; ncols = 33  # original test case
	nrows=300; nact=7; k=1000;
	novl = np.array([34,13,45,21,23,56,45], dtype=np.uint16)
	bc = Binco(nrows, nact, k, novl)


if __name__ == "__main__":
	main()
