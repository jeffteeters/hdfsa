# Script to test if distribution of overlaps onto target rows is Poisson

from scipy.stats import binom
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
	print("for thresh %s, predicted_mean=%.4f; found mean=%.4f, variance=%.4f, npairs=%s" % (
		thresh, predicted_mean, mean, var, len(overlap_counts)))


def main():
	nrows=20000
	ncols=250
	thresh=101
	get_overlap_stats(nrows, ncols, thresh)
	get_overlap_stats(nrows, ncols, thresh=104)
	get_overlap_stats(nrows, ncols, thresh=107)

main()

