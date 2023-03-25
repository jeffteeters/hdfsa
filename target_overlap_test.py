# Script to test if distribution of overlaps onto target rows is Poisson

from scipy.stats import binom
from scipy.stats import hypergeom 
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.integrate as integrate


rng = np.random.default_rng()

class Target_overlap():

	def __init__(self, nrows, ncols, thresh, k=500):
		# nrows - number rows in sdm contents matrix
		# ncols - number columns in sdm contents matrix
		# thresh - threshold for activating hard location (Hamming distance)
		# k - number of vectors being stored in sdm for empirical test.
		#     Should be small because will check all pairs for overlaps,
		#     e.g. 500 generates 124750 pairs
		self.nrows = nrows
		self.ncols = ncols
		self.thresh = thresh
		self.k = k
		self.empirical()
		self.numerical()
		self.naive()
		self.compare_empirical_numerical()


	def empirical(self):
		# print("starting empirical, thresh=%s, nrows=%s" % (self.thresh, self.nrows))
		# generate sdm addresses
		addresses = rng.integers(0, high=2, size=(self.nrows, self.ncols), dtype=np.int8)
		hard_locations = []
		select_counts = np.empty(self.k, dtype=int)
		for i in range(self.k):
			random_address = rng.integers(0, high=2, size=self.ncols, dtype=np.int8)
			hamming_distances = np.count_nonzero(addresses!=random_address, axis=1)
			# use hamming distance to select locations
			selected_locations = np.where(hamming_distances <= self.thresh)[0]
			hard_locations.append(selected_locations)
			select_counts[i] = len(selected_locations)
		# save histogram (pmf) of select counts
		select_counts_hist = np.bincount(select_counts)
		self.empirical_select_counts_pmf = select_counts_hist / select_counts_hist.sum()
		# now find number of overlaps between pairs
		overlap_counts = []
		for i in range(self.k-1):
			for j in range(i+1, self.k):
				overlap_count = np.intersect1d(hard_locations[i], hard_locations[j]).size
				overlap_counts.append(overlap_count)
		self.empirical_mean = np.mean(overlap_counts)
		self.empirical_var = np.var(overlap_counts)
		self.predicted_mean = binom.cdf(self.thresh, self.ncols, 0.5)**2*self.nrows
		overlap_counts_hist = np.bincount(overlap_counts)
		self.empirical_oc_pmf = overlap_counts_hist / overlap_counts_hist.sum()

	def numerical(self):
		# A numerical estimate that takes into account the Hamming distance between addresses
		nrows = self.nrows
		ncols = self.ncols
		thresh = self.thresh
		# first calculate probability of different Hamming distances between two vectors
		x = np.arange(ncols + 1)  # all possible hamming distances between two random vectors
		pham = binom.pmf(x, ncols, 0.5)  # probability each hamming distance
		max_pham = pham.max()    # maximum probability
		cutoff_threshold = 10**6
		itemindex = np.where(pham > max_pham/cutoff_threshold)
		# find range of probabilities above the cutoff_threshold
		from_idx = itemindex[0][0]
		to_idx = itemindex[0][-1]
		width = to_idx - from_idx + 1
		total = pham[from_idx:to_idx+1].sum()
		# print("most probable Hamming distances, from_idx=%s, to_idx=%s, width=%s, total=%s" % (
		# 	from_idx, to_idx, width, total))
		# goal is to fill in the following array with the probabilities that both addresses are selected
		prob_both_selected = np.zeros(width)
		# for each possible Hamming distance between two addresses
		for h in range(from_idx, to_idx + 1):
			# for each possible Hamming distance between row in SDM and one of the addresses which is selected
			for da in range(from_idx, thresh + 1):
				max_hamming_bits_in_non_match_region = min(da, h)
				mc = np.arange(max_hamming_bits_in_non_match_region + 1)  # count of bits different in non-match area
				# setup call to hypergeom, based on example at:
				# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html
				# probability of selecting mc dogs if slecting N animals from M animals if n of them are dogs
				M = ncols  # total number of animals (bits)
				n = h      # number that are dogs (non-matching region size)
				N = da     # number of animals selecting (Hamming distance between address A and row)
				pnnm = hypergeom.pmf(mc, M, n, N)  # probability number in non-matching region
				db = da + h - 2*mc  # Hamming distances to second address
				pb_selected = np.dot(pnnm, db <= thresh)  # probability row selected by address B (Hamming <= threshold)
				prob_both_selected[h-from_idx] += pham[da] * pb_selected  # make total of weighted probabilities
		# After above, have probability row selected by both addresses for each Hamming distances between addresses
		# next, convert to distribution of overlaps
		# expected number of rows selected by one address
		expected_activaction_count = round(binom.cdf(thresh, ncols, 0.5) * nrows)
		max_num_overlap = round(expected_activaction_count / 2) # should be large enough so more is very improbable
		overlaps = np.arange(max_num_overlap + 1)  # all possible overlaps
		prob_overlaps = np.zeros(max_num_overlap + 1)  # will have probability of each overlap
		for h in range(from_idx, to_idx + 1):
			prob_overlaps += pham[h]*binom.pmf(overlaps, nrows, prob_both_selected[h-from_idx])  # add distribution for each Hamming distance
		# finally, calculate mean and variance of overlap_distribution
		mean_overlaps = np.dot(prob_overlaps, overlaps)
		self.numerical_mean = mean_overlaps
		self.numerical_var = np.dot(prob_overlaps, (overlaps - mean_overlaps)**2)
		self.numerical_oc_pmf = prob_overlaps

	def compare_empirical_numerical(self):
		print("nrows=%s, ncols=%s, thresh=%s. Empirical / Numerical mean=%0.3f / %0.3f; variance=%0.3f / %0.3f" %
			(self.nrows, self.ncols, self.thresh, self.empirical_mean, self.numerical_mean,
				self.empirical_var, self.numerical_var))
		print("   naive mean=%0.3f, variance=%0.3f" % (self.naive_mean, self.naive_var))
		# plot comparison of overlap count PMF
		limit = self.empirical_oc_pmf.size
		x = np.arange(limit)
		plt.title("Overlap count PMF for threshold=%s, nrows=%s, ncols=%s" % (self.thresh, self.nrows, self.ncols))
		plt.plot(x, self.empirical_oc_pmf, label = "empirical", lw=2.0)
		plt.plot(x, self.numerical_oc_pmf[0:limit], '--', label = "multiple binomial", lw=2.0)
		plt.plot(x, self.naive_oc_pmf[0:limit], 'r--', label = "single binomial", lw=2.0)
		plt.xlabel("Overlap count")
		plt.ylabel("Probability of overlap")
		plt.legend(loc="upper right")
		plt.show()

	def naive(self):
		# try naive approach -- using a single value for the overlap probability to generate the distribution
		prob_row_selected = binom.cdf(self.thresh, self.ncols, 0.5)
		max_num_overlap = round(prob_row_selected * self.nrows / 2)  # should be enough to cover likely possibilities
		prob_both_selected = prob_row_selected**2
		overlaps = np.arange(max_num_overlap + 1)  # all most likely overlaps
		prob_overlaps = binom.pmf(overlaps, self.nrows, prob_both_selected)  # probability each overlap
		mean_overlaps = np.dot(prob_overlaps, overlaps)
		self.naive_mean = mean_overlaps
		self.naive_var = np.dot(prob_overlaps, (overlaps - mean_overlaps)**2)
		self.naive_oc_pmf = prob_overlaps



	# def numerical_variance(self):
	# 	# calculate overlap mean and variance numerically
	# 	print("starting numerical, thresh=%s, nrows=%s, nact=%s" % (self.thresh, self.nrows, self.nact))
	# 	if self.nact is not None:
	# 		# calculate overlap pmf based on fixed nact
	# 		# use hypergeometric distribution
	# 		max_overlap = self.nact
	# 		possible_overlaps = range(max_overlap + 1)
	# 		M = self.nrows    # total number objects
	# 		n = self.nact  # total number type 1 objects
	# 		N = self.nact   # number drawn without replacement
	# 		ovp = hypergeom.pmf(possible_overlaps, M, n, N)
	# 		self.numerical_oc_pmf = ovp
	# 		self.numerical_select_counts = np.zeros(self.nact+1)
	# 		self.numerical_select_counts[self.nact] = 1
	# 		return

	# 	pin = binom.cdf(self.thresh, self.ncols, 0.5)  # probability random hard location is selected (Hamming distance <= thresh)
	# 	x = np.arange(self.nrows + 1)  # all the possible number or rows selected for a given address
	# 	pnin = binom.pmf(x, self.nrows + 1, pin)  # probability each number of rows selected (in circle)
	# 	self.numerical_select_counts = pnin
	# 	max_pnin = pnin.max()    # maximum probability
	# 	cutoff_threshold = 10**6
	# 	itemindex = np.where(pnin > max_pnin/cutoff_threshold)
	# 	from_idx = itemindex[0][0]
	# 	to_idx = itemindex[0][-1]
	# 	width = to_idx - from_idx + 1
	# 	total = pnin[from_idx:to_idx+1].sum()
	# 	print("thresh=%s, from_idx=%s, to_idx=%s, width=%s, total=%s" % (self.thresh, from_idx, to_idx, width, total))
	# 	ovp = np.zeros(to_idx + 1)  # overlap probabilities
	# 	ovn = np.arange(to_idx + 1) # overlap numbers (e.g. 0, 1, 2, ... to_idx)
	# 	for saved_size in range(from_idx, to_idx+1):
	# 		for add_size in range(from_idx, to_idx+1):
	# 			max_overlap = min(saved_size, add_size)
	# 			possible_overlaps = range(max_overlap + 1)
	# 			# use hypergeometric distribution
	# 			M = self.nrows    # total number objects
	# 			n = saved_size  # total number type 1 objects
	# 			N = add_size   # number drawn without replacement
	# 			overlap_probabilities = hypergeom.pmf(possible_overlaps, M, n, N)
	# 			# add these probabilities to the overall sum, weighted by the probabilities
	# 			ovp[0:max_overlap+1] += overlap_probabilities * pnin[saved_size] * pnin[add_size]
	# 	self.numerical_oc_pmf = ovp
	# 	print("Numerical probability of overlap 10 is: %s" % ovp[10])
	# 	# see what it looks like
	# 	mean = np.dot(ovp, ovn)
	# 	ex2 = np.dot(ovp, ovn**2)
	# 	var = ex2 - mean**2
	# 	print("found ovp, total=%s, mean=%s, var=%s" % (ovp.sum(), mean, var))

	# def compare_empirical_numerical(self):
	# 	# first plot empirical and numberical select counts
	# 	limit = self.empirical_select_counts_pmf.size
	# 	# print("compare empirical vs numerical, thresh=%s" % self.thresh)
	# 	# print("empirical select_counts (len=%s):\n%s" % (
	# 	# 	limit, self.empirical_select_counts_pmf))
	# 	# print("numerical select_counts (len=%s/%s):\n%s" % (
	# 	# 	limit, self.numerical_select_counts.size, self.numerical_select_counts[0:limit]))
	# 	x = np.arange(limit)
	# 	plt.title("Select counts for thresh=%s, nrows=%s" % (self.thresh, self.nrows))
	# 	plt.plot(x, self.empirical_select_counts_pmf, label = "empirical")
	# 	plt.plot(x, self.numerical_select_counts[0:limit], label = "numerical")
	# 	plt.legend(loc="upper left")
	# 	plt.show()
	# 	# plot comparison of overlap counts
	# 	limit = self.empirical_oc_pmf.size
	# 	x = np.arange(limit)
	# 	plt.title("Overlap counts for thresh=%s, nrows=%s" % (self.thresh, self.nrows))
	# 	plt.plot(x, self.empirical_oc_pmf, label = "empirical")
	# 	plt.plot(x, self.numerical_oc_pmf[0:limit], label = "numerical")
	# 	plt.legend(loc="upper right")
	# 	plt.show()


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

# def numerical_estimate(nrows, ncols, thresh):
# 	# A numerical estimate that takes into account the Hamming distance between addresses
# 	# first calculate probability of different Hamming distances between two vectors
# 	x = np.arange(ncols + 1)  # all possible hamming distances between two random vectors
# 	pham = binom.pmf(x, ncols, 0.5)  # probability each hamming distance
# 	max_pham = pham.max()    # maximum probability
# 	cutoff_threshold = 10**6
# 	itemindex = np.where(pham > max_pham/cutoff_threshold)
# 	from_idx = itemindex[0][0]
# 	to_idx = itemindex[0][-1]
# 	width = to_idx - from_idx + 1
# 	total = pham[from_idx:to_idx+1].sum()
# 	print("most probable Hamming distances, from_idx=%s, to_idx=%s, width=%s, total=%s" % (
# 		from_idx, to_idx, width, total))
# 	prob_both_selected = np.zeros(width)
# 	# for each possible Hamming distance between two addresses
# 	for h in range(from_idx, to_idx + 1):
# 		# for each possible Hamming distance between row in SDM and one of the addresses, which is selected
# 		# tprob = 0  # total probability
# 		for da in range(from_idx, thresh + 1):
# 			if h==5 and da==3 and False:
# 				import pdb; pdb.set_trace()
# 			max_hamming_bits_in_non_match_region = min(da, h)
# 			mc = np.arange(max_hamming_bits_in_non_match_region + 1)  # count of bits different in non-match area
# 			M = ncols  # total number of animals (bits)
# 			n = h      # number that are dogs (non-matching region size)
# 			N = da     # number of animals selecting (Hamming distance between address A and row)
# 			pnnm = hypergeom.pmf(mc, M, n, N)  # probability number in non-matching region
# 			# db = mc + h - (da - mc)  # Hamming distances to second address
# 			db = da + h - 2*mc
# 			pb_selected = np.dot(pnnm, db <= thresh)  # probability row selected by address B (under threshold)
# 			# print("h=%s, da=%s, db=%s, pb_selected=%s" % (h, da, db, pb_selected))
# 			prob_both_selected[h-from_idx] += pham[da] * pb_selected  # make total of weighted probabilities
# 	# After above, have probability row selected by both addresses for each Hamming distances between addresses
# 	# next, convert to distribution of overlaps
# 	# expected number of rows selected by one address
# 	expected_activaction_count = round(binom.cdf(thresh, ncols, 0.5) * nrows)
# 	max_num_overlap = round(expected_activaction_count / 2) # should be large enough to be very improbable
# 	overlaps = np.arange(max_num_overlap + 1)  # all possible overlaps
# 	prob_overlaps = np.zeros(max_num_overlap + 1)  # will have probability of each overlap
# 	for h in range(from_idx, to_idx + 1):
# 		prob_overlaps += pham[h]*binom.pmf(overlaps, nrows, prob_both_selected[h-from_idx])  # add distribution for each hamming distance
# 	# finally, calculate mean and variance of overlap_distribution
# 	mean_overlaps = np.dot(prob_overlaps, overlaps)
# 	var_overlaps = np.dot(prob_overlaps, (overlaps - mean_overlaps)**2)
# 	print("mean_overlaps=%s, var_overlaps=%s" % (mean_overlaps, var_overlaps))

def multiple_binomial_test():
	# compare single binomal to multi-binomal; plots used in writeup of multibinomal algorithm
	ncols=250
	for nrows in [20000, 200000]:
		for thresh in [101, 104, 107]:
			Target_overlap(nrows, ncols, thresh)

def multiple_binomial_test_vary_ncols():
	# compare single binomal to multi-binomal; plots used in writeup of multibinomal algorithm
	orig_ncols=250
	for ncols in [250, 500, 1000, 2000, 2500]:
	# for nrows in [200000]: # [20000, 200000]:
		nrows = 200000
		# for thresh in [101, 104, 107]:
		for thresh in [107]:
			new_thresh = round(binom.ppf(binom.cdf(thresh, orig_ncols, 0.5), ncols, 0.5))
			# import pdb; pdb.set_trace()
			Target_overlap(nrows, ncols, new_thresh)

def circle_intersect_test():
	thresh = 101
	for d in range(0,121,10):
		print("thresh=%s, d=%s, intersect=%s" % (thresh, d, circle_intersect(thresh, d)))

def pentti_paper_parameter_test():
	# test example numbers used in Pentti's book to see if activation distribution is what is predicted (it is)
	ncols=1000
	nrows=1000000
	thresh=447
	Target_overlap(nrows, ncols, thresh)

# pentti_paper_parameter_test()
multiple_binomial_test_vary_ncols()
# circle_intersect_test()
# numerical_estimate_test()
# numerical_estimate_debug()
# intersect_range_analysis(thresh=107, nrows=20000, ncols=250)

