# function to calculate expected normalized hamming distance
# for binarized sdm

import numpy as np
import math
from scipy.stats import binom
from scipy.stats import poisson
import matplotlib.pyplot as plt
import sys


def binarized_delta(nrows, nact, k):
	# nrows - number rows in sdm
	# activaction count - (number rows selected when writing or reading, must be odd)
	# k - number of items stored in SDM
	# import pdb; pdb.set_trace()
	assert nact % 2 == 1, "nact must be odd"
	ao = k * nact / nrows   # average number of vectors summed in each counter (ao - average overlap)
	cd = 0.5 - (0.4 / math.sqrt(ao - 0.44))   # delta (normalized hamming distance) per counter
	dp_hit = binom.sf(nact-((nact+1)/2), nact, cd)   # probability of getting majority of counters on right side of zero
	print("nact=%s, ao=%s, cd=%s, dp_hit=%s" % (nact, ao, cd, dp_hit))
	return dp_hit

def p_error_binom (N, D, dp_hit):
	# compute error by summing values given by two binomial distributions
	# N - width of word (number of components)
	# D - number of items in item memory
	# dp_hit - normalized hamming distance of superposition vector to matching vector in item memory
	phds = np.arange(N+1)  # possible hamming distances
	match_hammings = binom.pmf(phds, N+1, dp_hit)
	distractor_hammings = binom.pmf(phds, N+1, 0.5)  # should this be N (not N+1)?
	num_distractors = D - 1
	dhg = 1.0 # fraction distractor hamming greater than match hamming
	p_err = 0.0  # probability error
	for k in phds:
		dhg -= distractor_hammings[k]
		p_err += match_hammings[k] * (1.0 - dhg ** num_distractors)
	return p_err


def binary_sdm_analytical(nrows, ncols, nact, k, d):
	# estimate error for binary sdm assuming poission distribution for overlaps
	mean_overlap = nact * k / nrows
	# import pdb; pdb.set_trace()
	possible_overlaps = np.arange(1,(((mean_overlap*3)+1)//2)*2+1)  # 3 std, make sure even number of entries
	prob_overlap = poisson.pmf(possible_overlaps, mean_overlap)
	delt_overlap =  0.5 - (0.4 / np.sqrt(possible_overlaps - 0.44))   # delta (normalized hamming distance) per counter
	delt_overlap[0] = 0
	odd_delt_overlap = delt_overlap[0::2]  # select only odd number overlaps
	prob_overlap_odd = prob_overlap[0::2] + prob_overlap[1::2]  # probability of odd number overlaps
	dp_hit_new = np.dot(odd_delt_overlap, prob_overlap_odd)
	dp_hit_old = binarized_delta(nrows, nact, k)
	print ("dp_hit_new=%s, dp_hit_old=%s" % (dp_hit_new, dp_hit_old))
	p_err = p_error_binom(ncols, d, dp_hit_new)
	return p_err


def main():
	# test predicted error

	d=100; k=1000; ncols=512
	nacts = [1,3,5,7]
	rows_to_test = list(range(50, 380, 10))  # list(range(50, 101, 20)) # 
	p_err=np.empty((len(nacts), len(rows_to_test)), dtype=np.float64);

	for i in range(len(rows_to_test)):
		nrows = rows_to_test[i]
		for j in range(len(nacts)):
			nact = nacts[j]
			p_err[j,i] = binary_sdm_analytical(nrows, ncols, nact, k, d)
			print("nrows=%s, nact=%s, p_err=%s" % (nrows, nact, p_err[j,i]))

	# sys.exit("not plotting for now")
	# plot info
	# make plots
	plots_info = [
		{"subplot": 121, "scale":"linear"},
		{"subplot": 122, "scale":"log"},
	]
	xvals = rows_to_test
	for pi in plots_info:
		plt.subplot(pi["subplot"])
		for j in range(len(nacts)):
			plt.errorbar(rows_to_test, p_err[j], yerr=None, fmt="-o", label="nact=%s" % nacts[j])
		title = "Error rate vs nrows for binarized sdm"
		if pi["scale"] == "log":
			plt.yscale('log')
			title += " (log scale)"
		# plt.xticks(xvals, xticks)
		plt.title(title)
		plt.xlabel("SDM nrows")
		plt.ylabel("fraction error")
		plt.grid()
		plt.legend(loc="upper right")
	plt.show()

main()

