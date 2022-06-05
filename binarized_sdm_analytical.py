# function to calculate expected normalized hamming distance
# for binarized sdm

import numpy as np
import math
from scipy.stats import binom
from scipy.stats import poisson
import matplotlib.pyplot as plt
import sys

class Binarized_sdm_analytical():

	def __init__(self, nrows, ncols, nact, k, d):
		# nrows is number of rows (hard locations) in the SDM
		# ncols is the number of columns
		# nact is activaction count
		# k - number of items stored
		# d - size of item memory
		# mean_overlap = nact * k / nrows
		# import pdb; pdb.set_trace()
		# num_possible_overlaps = (round(5*mean_overlap)//2)*2  # 5 std, make sure even number so len(possible_overlap)
		# is odd so final delt_overlap (odd) won't need next item
		num_possible_overlaps = (k // 2) * 2  # if all items happend to activate same address (assuming nact == 1)
		# make sure even number so len(possible_overlap) is odd so final delt_overlap (odd) won't need next item
		possible_overlaps = np.arange(1,num_possible_overlaps)
		prob_one_trial_overlap = 1 / nrows
		prob_overlap = binom.pmf(possible_overlaps, num_possible_overlaps, prob_one_trial_overlap )
		# prob_overlap = poisson.pmf(possible_overlaps, mean_overlap)
		# print("sum_before=%s" % prob_overlap.sum())
		# prob_overlap = prob_overlap / prob_overlap.sum()  # normalize to take into account 0 overlaps not included
		# delt_overlap =  0.5 - (0.4 / np.sqrt(possible_overlaps - 0.44))   # delta (normalized hamming distance) per counter
		delt_overlap = binom.cdf((possible_overlaps-3)/2, possible_overlaps-1, 0.5)
		delt_overlap[1:-1:2]=delt_overlap[2::2] # replace even by next odd (if even overlap add random vector to break ties)
		# delt_overlap[0] = 0  # overlap 1 has 0 hamming distance
		hdist = np.empty(ncols + 1)  # probability mass function
		for h in range(len(hdist)):  # hamming distance
			phk = binom.pmf(h, ncols, delt_overlap)
			hdist[h] = np.dot(phk, prob_overlap)
		# print("hdist (pmf) sum is %s (should be close to 1)" % np.sum(hdist))
		# assert math.isclose(np.sum(pmf), 1.0), "hdist sum is not equal to 1, is: %s" % np.sum(pmf)
		self.match_hamming_distribution = hdist
		self.prob_overlap = prob_overlap
		self.delt_overlap = delt_overlap
		print("sum match_hamming_distribution=%s" % sum(self.match_hamming_distribution))
		p_err = self.compute_overall_perr(self.match_hamming_distribution, d)
		# self.match_hamming_distribution = match_hamming_distribution
		self.p_err = p_err

		# # import pdb; pdb.set_trace()
		# # odd_delt_overlap = delt_overlap[0::2]  # select only odd number overlaps
		# # prob_overlap_odd = prob_overlap[0::2] + prob_overlap[1::2]  # probability of odd number overlaps
		# match_hamming_distribution = np.zeros(ncols+1, dtype=np.float64)
		# for i in range(len(prob_overlap)):
		# 	overlap = possible_overlaps[i]
		# 	delta = delt_overlap[i] if overlap % 2 ==1 else delt_overlap[i+1]
		# 	hamming = round(delta * ncols)
		# 	match_hamming_distribution[hamming] += prob_overlap[i]
		# print("sum match_hamming_distribution=%s" % sum(match_hamming_distribution))
		# p_err = self.compute_overall_perr(match_hamming_distribution, d)
		# self.match_hamming_distribution = match_hamming_distribution
		# self.p_err = p_err

	def compute_hamming_dist(self, ncols):
		# compute distribution of probability of each hamming distance
		# print("start compute_hamming_dist for ncols=%s" % ncols)
		hdist = np.empty(ncols + 1)  # probability mass function
		for h in range(len(hdist)):  # hamming distance
			phk = binom.pmf(h, ncols, self.cop_err)
			hdist[h] = np.dot(phk, self.cop_prb)
		# print("hdist (pmf) sum is %s (should be close to 1)" % np.sum(hdist))
		# assert math.isclose(np.sum(pmf), 1.0), "hdist sum is not equal to 1, is: %s" % np.sum(pmf)
		self.hdist = hdist
		# possible_overlaps = np.arange(1,(((mean_overlap*3)+1)//2)*2+1)  # 3 std, make sure even number of entries
		# prob_overlap = poisson.pmf(possible_overlaps, mean_overlap)
		# delt_overlap =  0.5 - (0.4 / np.sqrt(possible_overlaps - 0.44))   # delta (normalized hamming distance) per counter
		# delt_overlap[0] = 0
		# odd_delt_overlap = delt_overlap[0::2]  # select only odd number overlaps
		# prob_overlap_odd = prob_overlap[0::2] + prob_overlap[1::2]  # probability of odd number overlaps
		# match_hamming_distribution = np.zeros(ncols+1, dtype=np.float64)
		# for i in range(len(odd_delt_overlap)):
		# 	match_hamming_distribution[round(odd_delt_overlap[i]*ncols)] += prob_overlap_odd[i]
		# print("sum match_hamming_distribution=%s" % sum(match_hamming_distribution))
		# p_err = self.compute_overall_perr(match_hamming_distribution, d)
		# self.match_hamming_distribution = match_hamming_distribution
		# self.p_err = p_err

# def binarized_delta(nrows, nact, k):
# 	# nrows - number rows in sdm
# 	# activaction count - (number rows selected when writing or reading, must be odd)
# 	# k - number of items stored in SDM
# 	# import pdb; pdb.set_trace()
# 	assert nact % 2 == 1, "nact must be odd"
# 	ao = k * nact / nrows   # average number of vectors summed in each counter (ao - average overlap)
# 	cd = 0.5 - (0.4 / math.sqrt(ao - 0.44))   # delta (normalized hamming distance) per counter
# 	dp_hit = binom.sf(nact-((nact+1)/2), nact, cd)   # probability of getting majority of counters on right side of zero
# 	print("nact=%s, ao=%s, cd=%s, dp_hit=%s" % (nact, ao, cd, dp_hit))
# 	return dp_hit

# def p_error_binom (N, D, dp_hit):
# 	# compute error by summing values given by two binomial distributions
# 	# N - width of word (number of components)
# 	# D - number of items in item memory
# 	# dp_hit - normalized hamming distance of superposition vector to matching vector in item memory
# 	phds = np.arange(N+1)  # possible hamming distances
# 	match_hammings = binom.pmf(phds, N+1, dp_hit)
# 	distractor_hammings = binom.pmf(phds, N+1, 0.5)  # should this be N (not N+1)?
# 	num_distractors = D - 1
# 	dhg = 1.0 # fraction distractor hamming greater than match hamming
# 	p_err = 0.0  # probability error
# 	for k in phds:
# 		dhg -= distractor_hammings[k]
# 		p_err += match_hammings[k] * (1.0 - dhg ** num_distractors)
# 	return p_err


	def compute_overall_perr(self, match_hamming_distribution, d):
		# compute overall error rate, by integrating over all hamming distances with distractor distribution
		# ncols is the number of columns in each row of the sdm
		# d is number of item in item memory
		ncols = len(match_hamming_distribution)
		n = ncols
		hdist = match_hamming_distribution
		h = np.arange(len(hdist+1))
		distractor_pmf = binom.pmf(h, n, 0.5)
		# self.plot(binom.pmf(h, n, 0.5), "distractor pmf", "hamming distance", "probability")
		ph_corr = binom.sf(h, n, 0.5) ** (d-1)
		# ph_corr = binom.sf(h, n, 0.5) ** (d-1)
		# ph_corr = (binom.sf(h, n, 0.5) + match_hammings_area) ** (d-1)
		# self.plot(ph_corr, "probability correct", "hamming distance", "fraction correct")
		# self.plot(ph_corr * hdist, "p_corr weighted by hdist", "hamming distance", "weighted p_corr")
		hdist = hdist / np.sum(hdist)  # renormalize to increase due to loss of terms
		p_corr = np.dot(ph_corr, hdist)
		perr = 1 - p_corr
		return perr

# def binary_sdm_analytical(nrows, ncols, nact, k, d):
# 	# estimate error for binary sdm assuming poission distribution for overlaps
# 	mean_overlap = nact * k / nrows
# 	# import pdb; pdb.set_trace()
# 	possible_overlaps = np.arange(1,(((mean_overlap*3)+1)//2)*2+1)  # 3 std, make sure even number of entries
# 	prob_overlap = poisson.pmf(possible_overlaps, mean_overlap)
# 	delt_overlap =  0.5 - (0.4 / np.sqrt(possible_overlaps - 0.44))   # delta (normalized hamming distance) per counter
# 	delt_overlap[0] = 0
# 	odd_delt_overlap = delt_overlap[0::2]  # select only odd number overlaps
# 	prob_overlap_odd = prob_overlap[0::2] + prob_overlap[1::2]  # probability of odd number overlaps
# 	match_hamming_distribution = np.zeros(ncols, dtype=np.float64)
# 	for i in range(len(odd_delt_overlap)):
# 		match_hamming_distribution[round(odd_delt_overlap[i]*ncols)] += prob_overlap_odd[i]
# 	print("sum match_hamming_distribution=%s" % sum(match_hamming_distribution))
# 	p_err = compute_overall_perr(match_hamming_distribution, d)
# 	return p_err

	# dp_hit_new = np.dot(odd_delt_overlap, prob_overlap_odd)
	# dp_hit_old = binarized_delta(nrows, nact, k)
	# print ("dp_hit_new=%s, dp_hit_old=%s" % (dp_hit_new, dp_hit_old))
	# p_err = p_error_binom(ncols, d, dp_hit_new)
	# return p_err


def main():
	# test predicted error

	d=100; k=1000; ncols=512
	nacts = [1,3] # ,5,7]
	rows_to_test = list(range(50, 180, 10))  # list(range(50, 101, 20)) # 
	p_err=np.empty((len(nacts), len(rows_to_test)), dtype=np.float64);

	for i in range(len(rows_to_test)):
		nrows = rows_to_test[i]
		for j in range(len(nacts)):
			nact = nacts[j]
			bsm = Binarized_sdm_analytical(nrows, ncols, nact, k, d)
			p_err[j,i] = bsm.p_err
			# print("nrows=%s, nact=%s, p_err=%s" % (nrows, nact, p_err[j,i]))

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


if __name__ == "__main__":
	# test sdm and bundle
	main()


