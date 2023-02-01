# Script to test if distribution of overlaps onto target rows is Poisson

from scipy.stats import binom
from scipy.stats import hypergeom 
import numpy as np

rng = np.random.default_rng()

def get_overlap_stats(nrows, ncols, thresh=101, k=500):
	# nrows - number rows in sdm contents matrix
	# ncols - number columns in sdm contents matrix
	# thresh - threshold for activating hard location
	# k - number of vectors being stored in sdm.  Should be small because will
	#     check all pairs for overlaps, e.g. 500 generates 124750 pairs
	# generate sdm addresses
	addresses = rng.integers(0, high=2, size=(nrows, ncols), dtype=np.int8)
	hard_locations = []
	for i in range(k):
		random_address = rng.integers(0, high=2, size=ncols, dtype=np.int8)
		hamming_distances = np.count_nonzero(addresses!=random_address, axis=1)
		hard_locations.append(np.where(hamming_distances <= thresh))
	# print("hard_locations=")
	# print(hard_locations)
	# now find number of overlaps between pairs
	overlap_counts = []
	for i in range(k-1):
		for j in range(i+1, k):
			overlap_counts.append(np.intersect1d(hard_locations[i], hard_locations[j]).size)
	# print("overlap counts=")
	# print(overlap_counts)
	mean = np.mean(overlap_counts)
	var = np.var(overlap_counts)
	predicted_mean = binom.cdf(thresh, ncols, 0.5)**2*nrows
	# perdict using hypergeometeric distribution
	expected_activaction_count = round(binom.cdf(thresh, ncols, 0.5) * nrows)
	n = N = expected_activaction_count
	M = nrows
	hyper_mean, hyper_var = hypergeom.stats(M, n, N)[0:2]
	print("for thresh %s, predicted_mean=%.4f; found mean=%.4f, variance=%.4f, hmean=%.4f, hvar=%.4f, npairs=%s" % (
		thresh, predicted_mean, mean, var, hyper_mean, hyper_var, len(overlap_counts)))

def numerical_variance(nrows, ncols, thresh=101):
	# calculate overlap mean and variance numerically
	pin = binom.cdf(thresh, ncols, 0.5)  # probability random hard location is selected (Hamming distance <= thresh)
	x = np.arange(nrows + 1)  # all the possible number or rows selected for a given address
	pnin = binom.pmf(x, nrows + 1, pin)  # probability each number of rows selected (in circle)
	max_pnin = pnin.max()    # maximum probability
	cutoff_threshold = 10**6
	itemindex = np.where(pnin > max_pnin/cutoff_threshold)
	from_idx = itemindex[0][0]
	to_idx = itemindex[0][-1]
	width = to_idx - from_idx + 1
	total = pnin[from_idx:to_idx+1].sum()
	print("thresh=%s, from_idx=%s, to_idx=%s, width=%s, total=%s" % (thresh, from_idx, to_idx, width, total))
	ovp = np.zeros(to_idx + 1)  # overlap probabilities
	ovn = np.arange(to_idx + 1) # overlap numbers (e.g. 0, 1, 2, ... to_idx)
	for saved_size in range(from_idx, to_idx+1):
		for add_size in range(from_idx, to_idx+1):
			max_overlap = min(saved_size, add_size)
			possible_overlaps = range(max_overlap + 1)
			# use hypergeometric distribution
			M = nrows    # total number objects
			n = saved_size  # total number type 1 objects
			N = add_size   # number drawn without replacement
			overlap_probabilities = hypergeom.pmf(possible_overlaps, M, n, N)
			# add these probabilities to the overall sum, weighted by the probabilities
			ovp[0:max_overlap+1] += overlap_probabilities * pnin[saved_size] * pnin[add_size]
	# see what it looks like
	mean = np.dot(ovp, ovn)
	ex2 = np.dot(ovp, ovn**2)
	var = ex2 - mean**2
	print("found ovp, total=%s, mean=%s, var=%s" % (ovp.sum(), mean, var))



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
	nrows=20000
	ncols=250
	thresh=101
	numerical_variance(nrows, ncols, thresh)
	numerical_variance(nrows, ncols, thresh=104)
	numerical_variance(nrows, ncols, thresh=107)


main()
# main_numerical()

