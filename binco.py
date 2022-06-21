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

	def __init__(self, nrows, nact, k, novl, delt_overlap, epochs=10, debug=False):
		# nrows - number of rows in sdm
		# nact - activaction count
		# k - number of items (transitions) to store in sdm
		# novl - int arry, stores number of overlaps (including target) of each selected row
		# delt_overlap - normalized hamming distance for number of overlaps without? counting target, indexed by overlaps
		# epochs - used for testing
		assert nact > 1
		assert novl.size == nact
		assert np.amin(novl) > 0, "must be at least target stored in counter"
		self.nrows = nrows
		self.nact = nact
		self.k = k
		self.novl = novl
		self.epochs = epochs
		self.debug = debug
		rng = np.random.default_rng()
		self.co_probs = self.calculate_co_probs()  # calculate probabilities of coordinated overlaps
		# print("co_probs=%s" % self.co_probs)
		son = novl.copy()-1 # single overlap number.  Will be modified when coordinated overlaps added. -1 for target
		nco_added = 0       # number coordinated overlaps added
		ave_num_overlaps = k * nact / nrows
		max_num_coordinated_overlaps = max(10, 2*round(ave_num_overlaps))
		co_ptr = np.full((nact-1, max_num_coordinated_overlaps), -1, dtype = np.int16)  # has index of values in covals;
			# initialize with -1 so can see more easily when values filled in
		covals = np.zeros( (nact-1)* max_num_coordinated_overlaps, dtype =np.int8)  # has values of coordinated overlaps, +-1
		num_covals = 0  # number coordinated overlaps added (length of valid covals)
		num_co_ptr = np.zeros(nact-1, dtype=np.uint16)  # has number of values in each row of co_ptr
		# potentially create coordinated overlaps of different sizes, starting from largest size down
		for nc in range(nact, 1, -1):  # number in overlap from same item
			seli = list(itertools.combinations(range(nact), nc))  # counters to compare
			for si in seli:
				min_length = np.amin(son[si])
				num_co_to_add = binom.rvs(min_length, co_probs[nc-2])
				for j in num_co_to_add:
					for i in si:
						co_ptr[i, num_co_ptr[i]] = num_covals
						num_co_ptr[i] += 1
					num_covals += 1
					son[si] -= 1   # reduce number of single overlaps by one, since one was used
		# now, calculate overall delta using configuration with added overlaps
		if num_covals == 0:
			# no coordinated values added, return normal voting calculation, same as in binarized_sdm_analytical
			pc_er = np.empty(2*nact, dtype=np.float64)  # for holding probability correct then probability error 
			er = delt_overlap[son]  # probability error
			pc = 1 - er # probability correct
			pc_er[0:nact] = pc  # copy probability correct
			pc_er[nact:] = er   # copy probability error
			pvote = np.sum(np.prod(pc_er[mpc], axis=1))
			delta = 1 - pvote
		else:
			# compute response to every combination of overlap values
			sv = list(itertools.combinations(range(nact), nc))  # counters to compare



				10e6  # larger than number in any counter
				for hli in seli[si]:  # hard location index
					cl = len(son[hli])
					if cl < min_length:
						min_length = cl
						# min_length is the maximum number of overlaps of length nc
						# now find actual number of overlaps
						cs = sdm_rcp[seli[si][0]]
						for hli in seli[si][1:]:  # hard location index
							cs = cs.intersection(sdm_rcp[hli])
						found_common = len(cs)
						assert found_common >= 0
						assert min_length >= 0
						assert min_length >= found_common
						checked_found[nc-2]['checked'] += min_length
						checked_found[nc-2]['found'] += found_common
						# remove found_common from columns so not counted again
						if found_common > 1:
							for hli in seli[si]:  # hard location index
								sdm_rcp[hli].difference_update(cs)
		self.check_found = checked_found




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
