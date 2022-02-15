# script to count overlaps in SDM
import math
from scipy.stats import hypergeom
from scipy.stats import binom
import numpy as np

class Ovc:

	def __init__(self, nrows, nact, k):
		# compute overlaps of target (nact rows) in SDM when k items are stored (k-1 which cause overlaps)
		# nrows is number of rows (hard locations) in the SDM
		# nact is activaction count
		# k is number of items stored in sdm
		self.nrows = nrows
		self.nact = nact
		self.k = k
		self.ov = {1 : self.one_item_overlaps()}  # 1 item overlaps, means 2 items are stored
		for i in range(2, k):
			self.ov[i] = self.n_item_overlaps(i)  # i items overlap, means i+1 items are stored
		self.verify_sums()
		self.perr = self.compute_perr()

	def verify_sums(self):
		# make sure probabilities for each number of items sum to one
		error_count = 0
		for i in range(1, self.k):
			probs = self.ov[i]
			total = probs[0]
			for j in range(1, len(probs)):
				total += probs[j]
			if total != 1:
				print("Error in probabilities for %s items, total = %s" % (i, total))
				error_count += 1
		print("%s errors in sums" % error_count)

	def one_item_overlaps(self):
		# compute probability of: (0, 1, 2, ... nact) overlaps if there is just one additional
		# item stored beyond the first item (which is the target, has nact rows that are targets)
		# nt - number of target rows
		# nd - number of non-target rows (distractors)
		nt = self.nact
		nd = self.nrows - nt
		# based on: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html#scipy.stats.hypergeom
		[M, n, N] = [self.nrows, self.nact, self.nact]
		rv = hypergeom(M, n, N)
		x = np.arange(0, n+1)
		pmf_ov = rv.pmf(x)
		return pmf_ov

	def n_item_overlaps(self, n_items):
		# n_items is number of items beyound the first being stored in the sdm.  In other words, n_items is
		# the number of items which could contribute to overlaps onto target rows.  Maximum number of possible overlaps
		# is n_items * nact.  This would occur if every item overlapped the target rows completely.
		# compute probability of: (0, 1, 2, ... (n_items * nact) overlaps if there are n_items additional
		# item stored beyond the first item (which is the target, has nact rows that are targets).
		nact = self.nact
		max_num_overlaps = nact * n_items
		max_num_previous_overlaps = max_num_overlaps - nact
		pmf = np.empty(max_num_overlaps + 1)  # probability mass function
		for no in range(max_num_overlaps + 1):  # number overlaps
			prob = 0.0
			if no > max_num_previous_overlaps:
				cio_start = no - max_num_previous_overlaps
			else:
				cio_start = 0
			if no < nact:
				cio_stop = no
			else:
				cio_stop = nact
			# print("n_items=%s, no=%s, cio_start=%s, stop=%s, max_num_overlaps=%s, max_num_previous_overlaps=%s" % (n_items,
			# 	no, cio_start, cio_stop, max_num_overlaps, max_num_previous_overlaps))
			for cio in range(cio_start, cio_stop+1):  # contributions from current item to overlaps
				# print("\tAdding ov[%s][%s] * ov[1][%s]" % (n_items-1, no-cio, cio))
				prob += self.ov[n_items - 1][no - cio] * self.ov[1][cio]
			pmf[no] = prob
		return pmf

	def compute_perr(self):
		# compute probability of error given number of overlaps
		nact = self.nact
		n_items = self.k - 1  # number of items beyond first (that is overlapping)
		max_num_overlaps = nact * n_items
		no = np.arange(max_num_overlaps + 1)  # number overlaps
		perr = binom.sf(nact, no, 0.5)
		return perr




def main():
	nrows = 6
	nact = 2
	k = 4
	ov = Ovc(nrows, nact, k)

main()

