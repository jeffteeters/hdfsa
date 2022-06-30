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

	def __init__(self, nrows, nact, k, novl, delt_overlap=None, epochs=10, debug=False):
		# nrows - number of rows in sdm
		# nact - activaction count
		# k - number of items (transitions) to store in sdm
		# novl - int arry, stores number of overlaps (including target) of each selected row
		# delt_overlap - normalized hamming distance for number of overlaps without? counting target, indexed by overlaps
		# epochs - used for testing
		assert nact > 1, "should only be used when nact > 1"
		assert nact % 2 == 1, "only implemented for nact odd"
		assert novl.size == nact
		assert np.amin(novl) > 0, "must be at least target stored in counter"
		self.nrows = nrows
		self.nact = nact
		self.k = k
		self.novl = novl
		self.epochs = epochs
		self.debug = debug
		self.delt_overlap = delt_overlap if delt_overlap is not None else self.compute_delt_overlap()
		rng = np.random.default_rng()
		self.co_probs = self.calculate_co_probs()  # calculate probabilities of coordinated overlaps
		# print("co_probs=%s" % self.co_probs)
		son = novl.copy()-1 # single overlap number.  Will be modified when coordinated overlaps added. -1 for target
		nco_added = 0       # number coordinated overlaps added
		ave_num_overlaps = k * nact / nrows
		max_num_coordinated_overlaps = max(10, 2*round(ave_num_overlaps))
		co_ptr = np.full((nact, max_num_coordinated_overlaps), -1, dtype = np.int16)  # has index of values in covals;
			# initialize with -1 so can see more easily when values filled in
		covals = np.zeros( nact* max_num_coordinated_overlaps, dtype =np.int8)  # has values of coordinated overlaps, +-1
		num_covals = 0  # number coordinated overlaps added (length of valid covals)
		num_co_ptr = np.zeros(nact, dtype=np.uint16)  # has number of values in each row of co_ptr
		# potentially create coordinated overlaps of different sizes, starting from largest size down
		for nc in range(nact, 1, -1):  # number in overlap from same item
			seli = list(itertools.combinations(range(nact), nc))  # counters to compare
			for m in range(len(seli)):
				# import pdb; pdb.set_trace()
				si = np.array(seli[m], dtype=np.uint16)
				min_length = np.amin(son[si])
				num_co_to_add = binom.rvs(min_length, self.co_probs[nc-2])
				if num_co_to_add > 0:
					print("adding %s when nc=%s, min_length=%s, si=%s, co_probs=%s" % (num_co_to_add,
						nc, min_length, si, self.co_probs[nc-2]))
				for j in range(num_co_to_add):
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
			print("num_covals=%s, num_co_ptr=%s" % (num_covals, num_co_ptr))
			# compute response to every combination of overlap values
			# create mpc and masks for weights for voting
			maski = []  # mask index
			for i in range(int((nact+1)/2),nact+1):  # i - all possible majority counts
				maski += list(itertools.combinations(range(nact), i))
			mpc = np.empty((len(maski), nact), dtype=np.int16)
			for i in range(nact):
				mpc[:,i] = i+nact  # default to point to error term
			for i in range(len(maski)):
				mpc[i,maski[i]] = maski[i]  # set to point to correct term
			delta_sum = 0  # for storing sum of computed delta for each possibility
			possible_patterns = list(itertools.product([-1, 1], repeat=num_covals)) # pattern of values.  Will be like (-1, 1, -1, -1), ...
			assert len(possible_patterns) == 2**num_covals
			er = np.empty(nact, dtype=np.float64)  # for storing error probability for each counter
			pc_er = np.empty(2*nact, dtype=np.float64)  # for holding probability correct then probability error
			for cvpat in possible_patterns: 
				covals[0:num_covals] = cvpat
				# cv_sums = np.zeros(nact, dtype=np.int16)   # for storing sums of coordinated overlap values for each column
				for i in range(nact):
					cv_sum = np.sum(covals[co_ptr[0:num_co_ptr[i]]]) if num_co_ptr[i] > 0 else 0
					nsingo = son[i]   # number single overlaps
					x = np.arange(nsingo+1)
					so_weights = x - (nsingo - x)  # for nsingo==9, gives: array([-9, -7, -5, -3, -1,  1,  3,  5,  7,  9])
					so_probabilities = binom.pmf(x, nsingo, 0.5)  # binomonial coefficients, give number of possible ways to select x items
					# if target positive, to cause error, want sum of overlaps plus target <= 0
					perror_sum = np.dot(so_weights + cv_sum + 1 <= 0, so_probabilities)
					# if target negative, to cause error, want sum of overlaps plus target > 0
					nerror_sum = np.dot(so_weights + cv_sum - 1 > 0, so_probabilities)
					prob_sum = np.sum(so_probabilities)
					vp_error_rate = (perror_sum + nerror_sum) / (2* prob_sum)  # divide by 2 because positive and negative errors summed
					er[i] = vp_error_rate
				# now calculate delta for this pattern by voting from each counter
				pc = 1 - er # probability correct
				pc_er[0:nact] = pc  # copy probability correct
				pc_er[nact:] = er   # copy probability error
				pvote = np.sum(np.prod(pc_er[mpc], axis=1))
				delta = 1 - pvote
				delta_sum += delta
			delta = delta_sum / len(possible_patterns)
		self.delta = delta  # computed normalized hamming distance

	def compute_delt_overlap(self):
		num_possible_overlaps = self.k - 1 # number of possible overlaps onto target item (1 row)
		possible_overlaps = np.arange(num_possible_overlaps+1)
		prob_one_trial_overlap = self.nact/self.nrows
		prob_overlap = binom.pmf(possible_overlaps, num_possible_overlaps, prob_one_trial_overlap )
		oddup = np.floor((possible_overlaps+1)/2)*2  # round odd values up to next even integer, e.g.: [ 0.,  2.,  2.,  4.,  4.,  6.,  6., ...
		delt_overlap = binom.cdf(oddup/2-1, oddup, 0.5) # normalized hamming distance for each overlap
		return delt_overlap

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
	print("delta=%s" % bc.delta)


if __name__ == "__main__":
	main()
