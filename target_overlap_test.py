# Script to test if distribution of overlaps onto target rows is Poisson

from scipy.stats import binom
from scipy.stats import hypergeom 
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.integrate as integrate


rng = np.random.default_rng()

class Target_overlap():

	def __init__(self, nrows, ncols, thresh=101, k=500, nact=None):
		# nrows - number rows in sdm contents matrix
		# ncols - number columns in sdm contents matrix
		# thresh - threshold for activating hard location
		# k - number of vectors being stored in sdm.  Should be small because will
		#     check all pairs for overlaps, e.g. 500 generates 124750 pairs
		# if nact is not None, use that as the number of selected rows
		self.nrows = nrows
		self.ncols = ncols
		self.thresh = thresh
		self.k = k
		self.nact = nact
		self.empirical()
		# self.numerical_variance()
		# self.compare_empirical_numerical()


	def empirical(self):
		print("starting empirical, thresh=%s, nrows=%s" % (self.thresh, self.nrows))
		# generate sdm addresses
		addresses = rng.integers(0, high=2, size=(self.nrows, self.ncols), dtype=np.int8)
		hard_locations = []
		select_counts = np.empty(self.k, dtype=int)
		for i in range(self.k):
			random_address = rng.integers(0, high=2, size=self.ncols, dtype=np.int8)
			hamming_distances = np.count_nonzero(addresses!=random_address, axis=1)
			if self.nact is not None:
				# pick nact random locations
				# selected_locations = rng.integers(0, high=self.nrows, size=self.nact)
				selected_locations = rng.choice(self.nrows, size=self.nact, replace=False)
			else:
				# use hamming distance to select locations
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
		print("starting numerical, thresh=%s, nrows=%s, nact=%s" % (self.thresh, self.nrows, self.nact))
		if self.nact is not None:
			# calculate overlap pmf based on fixed nact
			# use hypergeometric distribution
			max_overlap = self.nact
			possible_overlaps = range(max_overlap + 1)
			M = self.nrows    # total number objects
			n = self.nact  # total number type 1 objects
			N = self.nact   # number drawn without replacement
			ovp = hypergeom.pmf(possible_overlaps, M, n, N)
			self.numerical_oc_pmf = ovp
			self.numerical_select_counts = np.zeros(self.nact+1)
			self.numerical_select_counts[self.nact] = 1
			return

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

def circle_intersect(thresh=107, d=0):
	# thresh is threshold for activation also radius of circle
	# d is hamming distance between circle centers
	# returns fraction of circle area that overlaps
	r = thresh
	theta = np.arccos(d / (2 * r))
	w = math.sqrt(r**2 - (d/2)**2)
	area = 2*(r**2*theta - (w*d/2))
	return area / (math.pi * r**2)

def gpx(p, x, ncols):
	# function defined in Pentti's book, page 133; g(p, x) = ...
	# p - fraction of locations
	# x - fraction of ditance between circles; also the variable of integration
	# ncols - dnumber of components in vector
	# --
	# convert p (fraction of locations) to rp (radius in bits for that fraction)
	rp = binom.ppf(p, ncols, .5)
	# calculate cp given rp and ncols
	cp = (rp - ncols/2)/math.sqrt(ncols/4)
	# calculate gpx using equation
	result = 1 / (2*math.pi*math.sqrt(x*(1-x)))*math.exp(-0.5*cp**2/(1-x))
	return result

def normalize_circle_intersect(p, d, ncols):
	# function defined in Pentti's book, page 133; i(p, d) = ...
	# function to integrate
	fun = lambda x: gpx(p, x, ncols)
	range_start = d
	range_end = 1
	acc = integrate.quad(fun, range_start, range_end) # integrate over range
	portion = acc[0]
	return portion

def numerical_estimate(nrows, ncols, thresh):
	# A numerical estimate that takes into account the Hamming distance between addresses
	# first calculate probability of different Hamming distances between two vectors
	x = np.arange(ncols + 1)  # all possible hamming distances between two random vectors
	pham = binom.pmf(x, ncols, 0.5)  # probability each hamming distance
	max_pham = pham.max()    # maximum probability
	cutoff_threshold = 10**6
	itemindex = np.where(pham > max_pham/cutoff_threshold)
	from_idx = itemindex[0][0]
	to_idx = itemindex[0][-1]
	width = to_idx - from_idx + 1
	total = pham[from_idx:to_idx+1].sum()
	print("most probable Hamming distances, from_idx=%s, to_idx=%s, width=%s, total=%s" % (
		from_idx, to_idx, width, total))
	prob_both_selected = np.zeros(width)
	# for each possible Hamming distance between two addresses
	for h in range(from_idx, to_idx + 1):
		# for each possible Hamming distance between row in SDM and one of the addresses, which is selected
		# tprob = 0  # total probability
		for da in range(from_idx, thresh + 1):
			if h==5 and da==3 and False:
				import pdb; pdb.set_trace()
			max_hamming_bits_in_non_match_region = min(da, h)
			mc = np.arange(max_hamming_bits_in_non_match_region + 1)  # count of bits different in non-match area
			M = ncols  # total number of animals (bits)
			n = h      # number that are dogs (non-matching region size)
			N = da     # number of animals selecting (Hamming distance between address A and row)
			pnnm = hypergeom.pmf(mc, M, n, N)  # probability number in non-matching region
			# db = mc + h - (da - mc)  # Hamming distances to second address
			db = da + h - 2*mc
			pb_selected = np.dot(pnnm, db <= thresh)  # probability row selected by address B (under threshold)
			# print("h=%s, da=%s, db=%s, pb_selected=%s" % (h, da, db, pb_selected))
			prob_both_selected[h-from_idx] += pham[da] * pb_selected  # make total of weighted probabilities
	# After above, have probability row selected by both addresses for each Hamming distances between addresses
	# next, convert to distribution of overlaps
	# expected number of rows selected by one address
	expected_activaction_count = round(binom.cdf(thresh, ncols, 0.5) * nrows)
	max_num_overlap = round(expected_activaction_count / 2) # should be large enough to be very improbable
	overlaps = np.arange(max_num_overlap + 1)  # all possible overlaps
	prob_overlaps = np.zeros(max_num_overlap + 1)  # will have probability of each overlap
	for h in range(from_idx, to_idx + 1):
		prob_overlaps += pham[h]*binom.pmf(overlaps, nrows, prob_both_selected[h-from_idx])  # add distribution for each hamming distance
	# finally, calculate mean and variance of overlap_distribution
	mean_overlaps = np.dot(prob_overlaps, overlaps)
	var_overlaps = np.dot(prob_overlaps, (overlaps - mean_overlaps)**2)
	print("mean_overlaps=%s, var_overlaps=%s" % (mean_overlaps, var_overlaps))

def main():
	nrows=20000
	ncols=250
	thresh=101
	get_overlap_stats(nrows, ncols, thresh)
	get_overlap_stats(nrows, ncols, thresh=104)
	get_overlap_stats(nrows, ncols, thresh=107)

def main_numerical():
	nrows=200000
	# nrows=6
	ncols=250
	thresh=101
	# Target_overlap(nrows, ncols, thresh)
	# Target_overlap(nrows, ncols, thresh=104)
	# Target_overlap(nrows, ncols, thresh=107, nact=2)
	Target_overlap(nrows, ncols, thresh=107)
	# Target_overlap(nrows, ncols, thresh=110)

def circle_intersect_test():
	thresh = 101
	for d in range(0,121,10):
		print("thresh=%s, d=%s, intersect=%s" % (thresh, d, circle_intersect(thresh, d)))

def numerical_estimate_test():
	nrows=200000
	ncols=250
	thresh=107
	Target_overlap(nrows, ncols, thresh=thresh)
	numerical_estimate(nrows, ncols, thresh)

def numerical_estimate_debug():
	nrows=20
	ncols=10
	thresh=4
	numerical_estimate(nrows, ncols, thresh)

# main()
# main_numerical()
# circle_intersect_test()
numerical_estimate_test()
# numerical_estimate_debug()
# intersect_range_analysis(thresh=107, nrows=20000, ncols=250)

