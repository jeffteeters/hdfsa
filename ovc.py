# script to count overlaps in SDM
import math
from scipy.stats import hypergeom
from scipy.stats import binom
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

class Ovc:

	def __init__(self, nrows, ncols, nact, k, d=2):
		# compute overlaps of target (nact rows) in SDM when k items are stored (k-1 which cause overlaps)
		# nrows is number of rows (hard locations) in the SDM
		# ncols is the number of columns
		# nact is activaction count
		# k is number of items stored in sdm
		# d is the number of items in item memory.  Used to compute probability of error in recall
		self.nrows = nrows
		self.ncols = ncols
		self.nact = nact
		self.k = k
		self.d = d
		self.ov = {1 : self.one_item_overlaps()}  # 1 item overlaps, means 2 items are stored
		for i in range(2, k):
			self.ov[i] = self.n_item_overlaps(i)  # i items overlap, means i+1 items are stored
		# self.verify_sums()
		self.perr = self.compute_perr()
		self.hdist = self.compute_hamming_dist()
		self.show_found_values()
		# self.make_hamming_hist()

	def compute_overall_error(nrows, ncols, nact, k, d=2):
		ov = Ovc(nrows, ncols, nact, k, d)
		overall_perr = ov.compute_overall_perr()
		return overall_perr


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
		assert math.isclose(np.sum(pmf), 1.0), "sum of pmf should be 1 is: %s" % np.sum(pmf)
		return pmf

	def compute_perr(self):
		# compute probability of error given number of overlaps
		nact = self.nact
		n_items = self.k - 1  # number of items beyond first (that is overlapping)
		max_num_overlaps = nact * n_items
		no = np.arange(max_num_overlaps + 1)  # number overlaps
		thres = (no - nact)/2
		pe_plus = binom.cdf(thres, no, 0.5)
		self.plot(pe_plus, "pe_plus", "# overlaps", "binom.cdf(thres, no, 0.5)")
		pe_minus = binom.cdf(thres - .25, no, 0.5)
		self.plot(pe_minus, "pe_minus", "# overlaps", "binom.cdf(thres - 1, no, 0.5)")
		perr = (pe_plus + pe_minus) / 2.0
		# for i in range(len(no)):
		#	thres = (no - nact)/2
			# if bit stored is positive, then error occurs if counter sum is zero or negative
			# that is, if count of #1's in overlaps is <= thresh

			# if bit stored is negative, then error occurs if counter sum is > nact
			# that is, if count of #-1's is < thresh.  < thresh means <= thresh -1
		return perr

	def compute_hamming_dist(self):
		# compute distribution of probability of each hamming distance
		n = self.ncols
		pov = self.ov[self.k - 1]  # probability of overlaps
		perr = self.perr    # probability of error for each number of overlap
		assert len(pov) == len(perr)
		pmf = np.empty(n + 1)  # probability mass function
		for h in range(len(pmf)):  # hamming distance
			phk = binom.pmf(h, n, self.perr)
			pmf[h] = np.dot(phk, pov)
		assert math.isclose(np.sum(pmf), 1.0), "hdist sum is not equal to 1, is: %s" % np.sum(pmf)
		return pmf

	def compute_perr_orig(self):
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
		# self.plot_mer(mer)
		perr = binom.cdf(mer, no, 0.5)
		return perr

	def compute_overall_perr(self):
		# compute overall error, by integrating over all hamming distances
		n = self.ncols
		hdist = self.hdist
		h = np.arange(len(hdist))
		self.plot(binom.pmf(h, n, 0.5), "distractor pmf", "hamming distance", "probability")
		ph_corr = binom.sf(h, n, 0.5) ** (self.d-1)
		self.plot(ph_corr, "probability correct", "hamming distance", "fraction correct")
		self.plot(ph_corr * hdist, "p_corr weighted by hdist", "hamming distance", "weighted p_corr")
		p_corr = np.dot(ph_corr, hdist)
		return 1 - p_corr


	def compute_overall_perr_orig(self):
		# compute overall error, assuming each overlap is a separate binomonal distribution and compute
		# the probability of error for that.  Then combine them all by weighting them according to the
		# probability of the overlaps
		n = self.ncols
		mm = self.perr   # match mean (single bit error rate for each overlap number)
		mv = mm*(1.-mm)/n  # match variance
		dm = 0.5
		dv = dm*(1.-dm)/n  # distractor variance
		cm = dm - mm       # combined mean
		cs = np.sqrt(mv + dv)       # combined standard deviation
		ov_per = norm.cdf(0, cm, cs)  # error in each overlap
		# self.plot_ov_per(ov_per)
		self.plot(ov_per, "Error in each overlap computed by cdf of difference between distributions", "# overlaps",
			"fraction error")
		weighted_error = ov_per * self.ov[self.k - 1]
		self.plot(weighted_error, "weighted error", "# overlaps", "fraction error")
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

	def plot(self, data, title, xlabel, ylabel):
		xvals = range(len(data))
		plt.plot(xvals, data, "o-")
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.title(title)
		plt.grid(True)
		plt.show()


	# def plot_ov_per(self, ov_per):
	# 	nact = self.nact
	# 	n_items = self.k - 1  # number of items beyond first (that are overlapping)
	# 	max_num_overlaps = nact * n_items
	# 	no = np.arange(max_num_overlaps + 1)  # number overlaps
	# 	# show frequency of overlaps
	# 	plt.plot(no, ov_per, "o-")
	# 	plt.xlabel('Number of overlaps')
	# 	plt.ylabel('norm.cdf(0, cm, cs)')
	# 	plt.title('Result of norm.cdf(0, cm, cs) function when  k=%s' % self.k)
	# 	plt.grid(True)
	# 	plt.show()

	def show_found_values(self):
		nact = self.nact
		n_items = self.k - 1  # number of items beyond first (that are overlapping)
		max_num_overlaps = nact * n_items
		no = np.arange(max_num_overlaps + 1)  # number overlaps
		# show frequency of overlaps
		plt.plot(no, self.ov[n_items]*(n_items*nact), "o-")
		plt.xlabel('Number of overlaps (x)')
		# plt.ylabel('Relative frequency')
		plt.ylabel('Count of overlaps')
		# plt.title('Relative probablilty of x overlaps when  k=%s' % self.k)
		plt.title('Expected count of overlaps when  k=%s' % self.k)
		plt.grid(True)
		plt.show()
		# show probability of error vs. number of overlaps
		plt.plot(no, self.perr, "o-")
		plt.xlabel('Number of overlaps (x)')
		plt.ylabel('Probability of error')
		plt.title('Probability of error vs. overlaps when  k=%s' % self.k)
		plt.grid(True)
		plt.show()
		# show hamming distribution
		self.plot(self.hdist, "hamming distribution", "hamming distance", "probability")


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
	d = 2 #7
	# ov = Ovc(nrows, ncols, nact, k, d)
	ncols = 33
	# overall_perr = ov.compute_overall_perr(ncols)
	overall_perr = Ovc.compute_overall_error(nrows, ncols, nact, k, d)
	print("for k=%s, d=%s, sdm size=(%s, %s, %s), overall_perr=%s" % (k, d, nrows, ncols, nact, overall_perr))
	# ncols = 52
	# overall_perr = Ovc.compute_overall_error(nrows, ncols, nact, k)
	# print("for k=%s, sdm size=(%s, %s, %s), overall_perr=%s" % (k, nrows, ncols, nact, overall_perr))


if __name__ == "__main__":
	main()
