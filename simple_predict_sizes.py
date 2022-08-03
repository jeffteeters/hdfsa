
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

found_dot_bundle_sizes = [
	[1, 15221],
	[2, 25717],
	[3, 35352],
	[4, 44632],
	[5, 53750],
	[6, 62794],
	[7, 71812],
	[8, 80824],
	[9, 89843]
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
		delta = 0.5 - pp / math.sqrt(2 * (ncols + pp))
		fp = norm.ppf(delta)
		a = 1-(nact/fp**2) # tried with 0-, works better for larger perf when 1-
		b = (k-1)*nact
		c = (k-1)*nact**3
	else:
		pp2 = pp**2
		a = (ncols*nact/pp2 - 1)/(2*k-1)
		b = -nact
		c = -nact**3
		# pp2*nact-nact**2*ncols
		# b = pp2*nact**2
		# c = pp2*nact**4
		# a = nact - (nact**2 * ncols/pp**2)
		# b = nact**2
		# c = nact**4
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
		# print("m1=%s, m2=%s" % (m1, m2))
		return round(m1)
	else:
		return round(m2)

def sdm_perr(nrows, nact, k, d, ncols, binarized):
	# return predicted error rate using formula derived from difference of match and distractor distributions
	if binarized:
		# first calculate single bit error rate using Poisson distribution, p. 13-14 or Pentti's book chapter
		mean = nact
		la = nact**2 / nrows  # lambda
		var = (k - 1)*(la + la**2) + mean  # variance
		delta = norm.cdf(-mean/math.sqrt(var))  # single bit error rate
		# now create combined distribution, subtracting this distribution (for match) from distractor
		cm = 0.5 - delta  # combined mean
		cv = (delta*(1-delta) + 0.5*(1 - 0.5)) / ncols
		perr = norm.cdf(-cm/math.sqrt(cv))
	else:
		# using dot product to compute distances
		cm = ncols * nact  # combined distribution mean
		la = nact**2 / nrows
		cv = ncols * (2*k-1)*(la + la**2) + nact  # combined distribution variance
		perr = norm.cdf(-cm / math.sqrt(cv))
	return perr

	
def sdm_optimum_size(perf, k=1000, d=100, ncols=512, binarized=True):
	# find nact giving minimum number of rows
	max_nact = 11
	nrows = np.empty(max_nact, dtype=np.int16)
	# print("optimum nacts found for binarized=%s:" % binarized)
	for i in range(max_nact):
		nact = i+1
		nrows[i] = sdm_nrows(perf,nact,k,d,ncols, binarized)
		opt_nact = optimum_nact(k, nrows[i])
		# if opt_nact == nact:
		# 	print("%s/%s" %(nrows[i], nact))
	mini = np.argmin(nrows)
	optSize = (nrows[mini], mini+1, "%s/%s" % (nrows[mini], mini+1))  # return ints and string, eg (134, 7, "134/7")
	return optSize

def optimum_nact(k, m):
	# return optimum nact given number items stored (k) and number of rows (m)
	# from formula given in Pentti paper
	fraction_rows_activated =  1.0 / ((2*m*k)**(1/3))
	nact = round(m * fraction_rows_activated)
	return nact

def main():
	k = 1000
	d = 100
	ncols = 512
	print("Legend:")
	print("p_bun1 - predicted bundle length, binarized=True (hamming used for match to item memory)")
	print("p_bun8 - predicted bundle length, binarized=False (dot product used for match to item memory)")
	print("bun8   - found bundle length, binarized=False")
	print("pb8/pb1 - p_bun8 / p_bun1")
	print("bun1 - Found bundle length, binarized=True")
	print("bun8ov1 - bun8/f_bun1  (found bun8 / found bun_1")
	print("pb1/fb1 - predicted bundle length / found bundle length (binarized=True")
	print("sdm_ham - predicted sdm 8 bit counter, sum thresholded (hamming match)")
	print("sdham_pe - predicted error of sdm_ham")
	print("sdm_anl - sdm dimensions 8 bit counter, sum thresholded found numerically using sdm_ae script")
	print("sdm_jkl - prediced sdm size using jaeckel simple size prediction")
	print("sdm_dot - predicted sdm size using dot product for match (8 bit counter, non-thresholded sums)")
	print("sdot_pe - perdicted error using sdm_dot")
	print("ham/dot - pSDMham/sdm_dot")
	print(" OUTPUT FOR d=%s, k=%s" % (d, k))
	print("i\tp_bun1\tp_bun8\tbun8\tpb8/pb1\tbun1\tbun8ov1\tpb1/fb1\tpSDMham\tsdham_pe\tsdm_anl\tsdm_jkl\tsdm_dot\tham/dot\tsdot_pe")
	for i in range(1, 10):
		perf = 10**(-i)
		binLen = bundle_length(k, perf, d, binarized=True)
		galLen = bundle_length(k, perf, d, binarized=False)
		ratio = binLen / galLen
		bun1 = found_binarized_bundle_sizes[i-1][1]
		bun8 = found_dot_bundle_sizes[i-1][1]
		bun8ov1 = bun8/bun1
		ratio2 = binLen / bun1
		# sdmLen = sdm_nrows(k, perf, d)
		sdm_hamming = sdm_optimum_size(perf, k=k, d=d, ncols=ncols, binarized=True)
		sdham_pe = sdm_perr(sdm_hamming[0], sdm_hamming[1], k, d, ncols=ncols, binarized=True)
		sdm_dot = sdm_optimum_size(perf, k=k, d=d, ncols=ncols, binarized=False)
		sdot_pe = sdm_perr(sdm_dot[0], sdm_dot[1], k, d, ncols=ncols, binarized=False)
		dot_over_ham = sdm_dot[0] / int(sdm_hamming[0])
		sdm_anl_size = "%s/%s" % (found_sdm_8_bit_counter_threshold_sizes[i-1][1],
			found_sdm_8_bit_counter_threshold_sizes[i-1][2])
		sdm_jaeckel_size = "%s/%s" % (found_sdm_jaeckel_sizes[i-1][1], found_sdm_jaeckel_sizes[i-1][2])
		print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (i, binLen, galLen, bun8,
			round(1.0/ratio,4), bun1,
			round(bun8ov1,4),
			round(ratio2, 4), sdm_hamming[2], sdham_pe, sdm_anl_size, sdm_jaeckel_size, sdm_dot[2], round(dot_over_ham, 4),
			sdot_pe))


def compare_sdm_ham_dot():
	print("nact\tper\thamL\tdotL\tdot/ham")
	for nact in range(1,5):
		for i in range(1, 10):
			perf = 10**(-i)
			ham_len = sdm_nrows(perf, nact=nact, k=1000, d=100, ncols=512, binarized=True)
			dot_len = sdm_nrows(perf, nact=nact, k=1000, d=100, ncols=512, binarized=False)
			dot_over_ham = dot_len / ham_len
			print("%s\t%s\t%s\t%s\t%s" % (nact, i, ham_len, dot_len, dot_over_ham))



if __name__ == "__main__":
	# compare_sdm_ham_dot()
	main()