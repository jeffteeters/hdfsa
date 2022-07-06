
# script to predict sizes using simple equations, without integrating a distribution

from scipy.stats import norm
import math
import numpy as np

found_binarized_bundle_sizes = [
	# output for bundle sizes:
	# 	Bundle sizes, k=1000, d=100:
	[1, 24002],
	[2, 40503],
	[3, 55649],
	[4, 70239],
	[5, 84572],
	[6, 98790],
	[7, 112965],
	[8, 127134],
	[9, 141311]
]

found_sdm_8_bit_counter_threshold_sizes = [
	# sdm_anl.Sdm_error_analytical
	# SDM sizes, nact:
	[1, 51, 1],
	[2, 86, 2],
	[3, 125, 2],
	[4, 168, 2],
	[5, 196, 3],
	[6, 239, 3],
	[7, 285, 3],
	[8, 312, 4],
	[9, 349, 4]
]

found_sdm_jaeckel_sizes = [
	# output from sdm_jaeckel:
	# SDM sizes, nact
	[1, 51, 1],
	[2, 88, 2],
	[3, 122, 2],
	[4, 155, 2],
	[5, 188, 3],
	[6, 221, 3],
	[7, 255, 3],
	[8, 288, 3],
	[9, 323, 4],
]
def bundle_length(k, perf, d, binarized=True):
	# k - number if items stored in bundle
	# perf - desired error (fraction, final) when recalling one item and comparing it to d-1 other items
	# d - number of items in item memory (d-1 are compared)
	# binarized - True if counters are binarized (using hamming distance for matching),
	#             False if not (use dot product for matching)
	per = perf / (d - 1)
	const = (3.125 * k - 2.375) if binarized else (2 * k - 1)
	# norm.ppf is inverse of cdf
	# https://stackoverflow.com/questions/20626994/how-to-calculate-the-inverse-of-the-normal-cumulative-distribution-function-in-p
	ncols = norm.ppf(per) ** 2 * const
	return round(ncols)

def sdm_nrows(perf, nact=1, k=1000, d=100, ncols=512, binarized=True):
	# k - number if items stored
	# perf - desired error (fraction, final) when recalling one item and comparing it to d-1 other items
	# d - number of items in item memory (d-1 are compared)
	# binarized - True if sum of counters are binarized (using hamming distance for matching),
	#             False if not (use dot product for matching)
	# this derived from equations in Pentti's book chapter
	per = perf / (d - 1)
	pp = norm.ppf(per)
	if binarized:
		delta = 0.5 - pp / math.sqrt(2 * (ncols - pp))
		fp = norm.ppf(delta)
		a = nact-(nact/fp)**2
		b = (k-1)*nact**2
		c = (k-1)*nact**4
	else:
		pp2 = pp**2
		# a = pp2*nact-nact**2*ncols
		# b = pp2*nact**2
		# c = pp2*nact**4
		a = nact - (nact**2 * ncols/pp**2)
		b = nact**2
		c = nact**4
	b2ac = b**2 - 4*a*c
	if b2ac < 0:
		print("b2ac=%s" % b2ac)
		import pdb; pdb.set_trace()
	sbac = math.sqrt(b2ac)
	m1 = (-b + sbac) / (2*a)
	m2 = (-b - sbac) / (2*a)
	# if m1 < 0:
	# 	if m2 > 0:
	# 		return round(m2)
	if not binarized:
		print("m1=%s, m2=%s" % (m1, m2))
	return round(m2)
	
def sdm_optimum_size(perf, k=1000, d=100, ncols=512, binarized=True):
	# find nact giving minimum number of rows
	max_nact = 11
	nrows = np.empty(max_nact, dtype=np.int16)
	for i in range(max_nact):
		nact = i+1
		nrows[i] = sdm_nrows(perf,nact,k,d,ncols, binarized)
	mini = np.argmin(nrows)
	optSize = "%s/%s" % (nrows[mini], mini+1)
	return optSize



def main():
	k = 1000
	d = 100
	print("i\tbinLen\tgalLen\tratio\tfoundBinLen\tratio2\tsdmOptSize\tsdm_anl_size\tsdm_jaeckel_size\tsdm_dot")
	for i in range(1, 10):
		perf = 10**(-i)
		binLen = bundle_length(k, perf, d, binarized=True)
		galLen = bundle_length(k, perf, d, binarized=False)
		ratio = binLen / galLen
		found_binLen = found_binarized_bundle_sizes[i-1][1]
		ratio2 = binLen / found_binLen
		# sdmLen = sdm_nrows(k, perf, d)
		sdmOptSize = sdm_optimum_size(perf, k=k, d=d, ncols=512, binarized=True)
		sdmOptSize_dot = sdm_optimum_size(perf, k=k, d=d, ncols=512, binarized=False)
		sdm_anl_size = "%s/%s" % (found_sdm_8_bit_counter_threshold_sizes[i-1][1],
			found_sdm_8_bit_counter_threshold_sizes[i-1][2])
		sdm_jaeckel_size = "%s/%s" % (found_sdm_jaeckel_sizes[i-1][1], found_sdm_jaeckel_sizes[i-1][2])
		print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (i, binLen, galLen, ratio, found_binLen, ratio2, sdmOptSize,
			sdm_anl_size, sdm_jaeckel_size, sdmOptSize_dot))

if __name__ == "__main__":
	main()