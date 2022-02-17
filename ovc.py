# script to count overlaps in SDM
import math
from scipy.stats import hypergeom
from scipy.stats import binom
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

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
		self.show_found_values()
		self.make_hamming_hist()

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
		# for i in range(len(no)):
		# 	ner = (no - nact)
		# 	# if bit stored is positive, then error occurs if counter sum is zero or negative
		# 	pe_plus = 
		mer = (no - nact)/2.0
		self.plot_mer(mer)
		perr = binom.cdf(mer, no, 0.5)
		return perr

	def compute_overall_perr(self, n):
		# n is width (number of columns) in SDM
		# compute overall error, assuming each overlap is a separate binomonal distribution and compute
		# the probability of error for that.  Then combine them all by weighting them according to the
		# probability of the overlaps
		mm = self.perr   # match mean (single bit error rate for each overlap number)
		mv = mm*(1.-mm)/n  # match variance
		dm = 0.5
		dv = dm*(1.-dm)/n  # distractor variance
		cm = dm - mm       # combined mean
		cv = np.sqrt(mv + dv)       # combined standard deviation
		ov_per = norm.cdf(0, cm, cv)  # error in each overlap
		overall_perr = np.dot(ov_per, self.ov[self.k - 1])
		return overall_perr

	def plot_mer(self, mer):
		nact = self.nact
		n_items = self.k - 1  # number of items beyond first (that are overlapping)
		max_num_overlaps = nact * n_items
		no = np.arange(max_num_overlaps + 1)  # number overlaps
		# show frequency of overlaps
		plt.plot(no, mer, "o-")
		plt.xlabel('Number of overlaps')
		plt.ylabel('int((#overlaps - nact)/2)')
		plt.title('Input to binom.cdf function when  k=%s' % self.k)
		plt.grid(True)
		plt.show()



	def show_found_values(self):
		nact = self.nact
		n_items = self.k - 1  # number of items beyond first (that are overlapping)
		max_num_overlaps = nact * n_items
		no = np.arange(max_num_overlaps + 1)  # number overlaps
		# show frequency of overlaps
		plt.plot(no, self.ov[n_items], "o-")
		plt.xlabel('Number of overlaps (x)')
		plt.ylabel('Relative frequency')
		plt.title('Relative probablilty of x overlaps when  k=%s' % self.k)
		plt.grid(True)
		plt.show()
		# show probability of error vs. number of overlaps
		plt.plot(no, self.perr, "o-")
		plt.xlabel('Number of overlaps (x)')
		plt.ylabel('Probability of error')
		plt.title('Probability of error vs. overlaps when  k=%s' % self.k)
		plt.grid(True)
		plt.show()


	def make_hamming_hist(self):
		# compute histogram of hamming distance frequencies and make a histogram
		hx = self.perr * self.ov[self.k-1]
		print("hx - input to histogram is:\n%s" % hx)

		n, bins, patches = plt.hist(hx, 50, density=True, facecolor='g', alpha=0.75)
		plt.xlabel('Normalized hamming distance')
		plt.ylabel('Probability')
		plt.title('Histogram of Hamming distance for k=%s' % self.k)
		# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
		# plt.xlim(40, 160)
		# plt.ylim(0, 0.03)
		plt.grid(True)
		plt.show()


def main():
	nrows = 6
	nact = 2
	k = 5
	ov = Ovc(nrows, nact, k)
	ncols = 52
	overall_perr = ov.compute_overall_perr(ncols)
	print("for k=%s, sdm size=(%s, %s, %s), overall_perr=%s" % (k, nrows, ncols, nact, overall_perr))


main()

