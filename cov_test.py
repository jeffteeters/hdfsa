# script to test overlap weighting fraction

import itertools
import numpy as np
import copy
from scipy.special import comb
from scipy.stats import binom

class Sdm_store_history:

	# Record complete history of writing to sdm for purposes of calculating
	# statistics of overlaps (one item, two item, ...)

	def __init__(self, nrows, nact, k, epochs=10, ncols=None, d=None, debug=False):
		# nrows - number of rows
		# nact - activaction count
		# k - number of items (transitions) to store in sdm
		# ncols - number of columns in each row of sdm
		# d - size of item memory (d-1 are distractors)
		self.nrows = nrows
		self.nact = nact
		self.k = k
		self.ncols = ncols
		self.d = d
		self.epochs = epochs
		rng = np.random.default_rng()
		thl = np.empty((self.k, self.nact), dtype=np.uint16)  # transition hard locations
		selected_rows = np.empty(self.k * self.nact, dtype=np.uint16)
		checked_found = np.zeros(self.nact - 1, dtype=[('checked', np.uint32), ('found', np.uint32)])  #,0 for checked, ,1 for found
		for epoch_id in range(self.epochs):
			for i in range(k):
				thl[i,:] = rng.choice(self.nrows, size=self.nact, replace=False)
			sdm_rows = {}  # contain index of all items stored in each row
			for i in range(nrows):
				sdm_rows[i] = []
			# simulate storing all transitions into sdm
			for i in range(k):
				for j in range(self.nact):
					sdm_rows[thl[i,j]].append(i)
			# convert row list of items to sets
			for i in range(nrows):
				sdm_rows[i] = set(sdm_rows[i])
			# now find number of overlaps of different sizes
			for i in range(k):
				# make copy of selected hard locations for this item, but with index for this item removed
				sdm_rcp = {}
				for j in range(nact):
					sdm_rcp[j] = copy.copy(sdm_rows[thl[i,j]])
					sdm_rcp[j].remove(i)
				# find overlaps of different sizes
				for nc in range(nact, 1, -1):  # number in overlap from same item
					assert nc >= 2
					seli = list(itertools.combinations(range(nact), nc))  # counters to compare
					for si in range(len(seli)):
						min_length = 10e6  # larger than number in any counter
						for hli in seli[si]:  # hard location index
							cl = len(sdm_rcp[hli])
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
		print("nrows=%s, nact=%s, k=%s, epochs=%s" % (self.nrows, self.nact, self.k, self.epochs))
		print("ovl\tchecked\tfound\tratio\tpredicted\t%diff")
		for ovl in range(nact-1):
			checked = checked_found[ovl]['checked']
			found = checked_found[ovl]['found']
			ratio = found/checked
			predicted_ratio = self.predict_ratio_binom(ovl+2)
			den = ratio if ratio > 0 else predicted_ratio
			diff = round((predicted_ratio - ratio)*100 / den, 3)
			print("%s\t%s\t%s\t%s\t%s\t%s" % (ovl+2, checked, found, ratio, predicted_ratio, diff))

	def predict_ratio(self, ovl):
		# ovl is number of coordinated overlaps, 2 <= ovl <= nact
		# find number of coordinated overlaps of this size stored
		assert ovl >= 2 and ovl <= self.nact
		nstored = self.k * comb(self.nact, ovl)
		ave_overlaps_per_row = (self.k * self.nact / self.nrows) - 1
		if ave_overlaps_per_row <= 0:
			ave_overlaps_per_row += 1
		npossible = comb(self.nrows, ovl) * ave_overlaps_per_row
		ratio = nstored / npossible
		return ratio

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
	ssh = Sdm_store_history(nrows, nact, k, epochs=1000)


if __name__ == "__main__":
	main()
