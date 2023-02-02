# Script to test if distribution of overlaps onto target rows is Poisson

from scipy.stats import binom
from scipy.stats import hypergeom 
import numpy as np
import matplotlib.pyplot as plt


rng = np.random.default_rng()

class Target_overlap():

	def __init__(self, nrows, ncols, thresh=101, k=500):
		# nrows - number rows in sdm contents matrix
		# ncols - number columns in sdm contents matrix
		# thresh - threshold for activating hard location
		# k - number of vectors being stored in sdm.  Should be small because will
		#     check all pairs for overlaps, e.g. 500 generates 124750 pairs
		self.nrows = nrows
		self.ncols = ncols
		self.thresh = thresh
		self.k = k
		self.empirical()
		self.numerical_variance()
		self.compare_empirical_numerical()


	def empirical(self):
		print("starting empirical, thresh=%s, nrows=%s" % (self.thresh, self.nrows))
		# generate sdm addresses
		addresses = rng.integers(0, high=2, size=(self.nrows, self.ncols), dtype=np.int8)
		hard_locations = []
		select_counts = np.empty(self.k, dtype=int)
		for i in range(self.k):
			random_address = rng.integers(0, high=2, size=self.ncols, dtype=np.int8)
			hamming_distances = np.count_nonzero(addresses!=random_address, axis=1)
			selected_locations = np.where(hamming_distances <= self.thresh)[0]
			hard_locations.append(selected_locations)
			select_counts[i] = len(selected_locations)
		# save histogram (pmf) of select counts
		select_counts_hist = np.bincount(select_counts)
		self.empirical_select_counts_pmf = select_counts_hist / select_counts_hist.sum()
		# print("hard_locations=")
		# print(hard_locations)
		# now find number of overlaps between pairs
		overlap_counts = []
		count_ten_sizes = []
		for i in range(self.k-1):
			for j in range(i+1, self.k):
				overlap_count = np.intersect1d(hard_locations[i], hard_locations[j]).size
				overlap_counts.append(overlap_count)
				if overlap_count==10:
					count_ten_sizes.append( (hard_locations[i].size, hard_locations[j].size, ))
		# print("overlap counts=")
		# print(overlap_counts)
		print("len(count_ten_sizes)=%s, fraction=%s" % (len(count_ten_sizes), len(count_ten_sizes)/len(overlap_counts)))
		mean = np.mean(overlap_counts)
		var = np.var(overlap_counts)
		predicted_mean = binom.cdf(self.thresh, self.ncols, 0.5)**2*self.nrows
		# perdict using hypergeometeric distribution
		expected_activaction_count = round(binom.cdf(self.thresh, self.ncols, 0.5) * self.nrows)
		n = N = expected_activaction_count
		M = self.nrows
		hyper_mean, hyper_var = hypergeom.stats(M, n, N)[0:2]
		print("for thresh %s, predicted_mean=%.4f; found mean=%.4f, variance=%.4f, hmean=%.4f, hvar=%.4f, npairs=%s" % (
			self.thresh, predicted_mean, mean, var, hyper_mean, hyper_var, len(overlap_counts)))
		# convert overlap_counts to histogram, then to a probability density function
		overlap_counts_hist = np.bincount(overlap_counts)
		self.empirical_oc_pmf = overlap_counts_hist / overlap_counts_hist.sum()

	def numerical_variance(self):
		# calculate overlap mean and variance numerically
		print("starting numerical, thresh=%s, nrows=%s" % (self.thresh, self.nrows))
		pin = binom.cdf(self.thresh, self.ncols, 0.5)  # probability random hard location is selected (Hamming distance <= thresh)
		x = np.arange(self.nrows + 1)  # all the possible number or rows selected for a given address
		pnin = binom.pmf(x, self.nrows + 1, pin)  # probability each number of rows selected (in circle)
		self.numerical_select_counts = pnin
		max_pnin = pnin.max()    # maximum probability
		cutoff_threshold = 10**6
		itemindex = np.where(pnin > max_pnin/cutoff_threshold)
		from_idx = itemindex[0][0]
		to_idx = itemindex[0][-1]
		width = to_idx - from_idx + 1
		total = pnin[from_idx:to_idx+1].sum()
		print("thresh=%s, from_idx=%s, to_idx=%s, width=%s, total=%s" % (self.thresh, from_idx, to_idx, width, total))
		ovp = np.zeros(to_idx + 1)  # overlap probabilities
		ovn = np.arange(to_idx + 1) # overlap numbers (e.g. 0, 1, 2, ... to_idx)
		for saved_size in range(from_idx, to_idx+1):
			for add_size in range(from_idx, to_idx+1):
				max_overlap = min(saved_size, add_size)
				possible_overlaps = range(max_overlap + 1)
				# use hypergeometric distribution
				M = self.nrows    # total number objects
				n = saved_size  # total number type 1 objects
				N = add_size   # number drawn without replacement
				overlap_probabilities = hypergeom.pmf(possible_overlaps, M, n, N)
				# add these probabilities to the overall sum, weighted by the probabilities
				ovp[0:max_overlap+1] += overlap_probabilities * pnin[saved_size] * pnin[add_size]
		self.numerical_oc_pmf = ovp
		print("Numerical probability of overlap 10 is: %s" % ovp[10])
		# see what it looks like
		mean = np.dot(ovp, ovn)
		ex2 = np.dot(ovp, ovn**2)
		var = ex2 - mean**2
		print("found ovp, total=%s, mean=%s, var=%s" % (ovp.sum(), mean, var))

	def compare_empirical_numerical(self):
		# first plot empirical and numberical select counts
		limit = self.empirical_select_counts_pmf.size
		# print("compare empirical vs numerical, thresh=%s" % self.thresh)
		# print("empirical select_counts (len=%s):\n%s" % (
		# 	limit, self.empirical_select_counts_pmf))
		# print("numerical select_counts (len=%s/%s):\n%s" % (
		# 	limit, self.numerical_select_counts.size, self.numerical_select_counts[0:limit]))
		x = np.arange(limit)
		plt.title("Select counts for thresh=%s, nrows=%s" % (self.thresh, self.nrows))
		plt.plot(x, self.empirical_select_counts_pmf, label = "empirical")
		plt.plot(x, self.numerical_select_counts[0:limit], label = "numerical")
		plt.legend(loc="upper left")
		plt.show()
		# plot comparison of overlap counts
		limit = self.empirical_oc_pmf.size
		x = np.arange(limit)
		plt.title("Overlap counts for thresh=%s, nrows=%s" % (self.thresh, self.nrows))
		plt.plot(x, self.empirical_oc_pmf, label = "empirical")
		plt.plot(x, self.numerical_oc_pmf[0:limit], label = "numerical")
		plt.legend(loc="upper right")
		plt.show()


def pentti_math():
	nv = 10  # number of vectors
	random_overlaps = rng.integers(0, high=20, size=nv, dtype=np.int8)
	random_bit_vals = rng.integers(0, high=2, size=nv, dtype=np.int8)*2-1
	print("overlaps=%s" % random_overlaps)
	print("bit_vals=%s" % random_bit_vals)
	var = np.var(random_overlaps)
	print("var=%0.4f" % var)
	var_weighted = np.var(random_overlaps * random_bit_vals)
	print("var_weighted=%0.4f" % var_weighted)


def main():
	nrows=20000
	ncols=250
	thresh=101
	get_overlap_stats(nrows, ncols, thresh)
	get_overlap_stats(nrows, ncols, thresh=104)
	get_overlap_stats(nrows, ncols, thresh=107)

def main_numerical():
	nrows=200000
	ncols=250
	thresh=101
	# Target_overlap(nrows, ncols, thresh)
	# Target_overlap(nrows, ncols, thresh=104)
	Target_overlap(nrows, ncols, thresh=107)
	# Target_overlap(nrows, ncols, thresh=110)


# main()
main_numerical()

