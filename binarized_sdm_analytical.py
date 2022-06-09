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
		num_possible_overlaps = k - 1 # number of possible overlaps onto target item (1 row)
		possible_overlaps = np.arange(0,num_possible_overlaps)  # may need to fix, change to n+1
		prob_one_trial_overlap = 1 / nrows
		prob_overlap = binom.pmf(possible_overlaps, num_possible_overlaps, prob_one_trial_overlap )
		# for experiment with uniform overlaps
		# import pdb; pdb.set_trace()
		# prob_overlap[:] = 0.0
		# avg_overlap = (k-1)/nrows
		# prob_overlap[round(avg_overlap)] = 0.5
		# prob_overlap[round(avg_overlap)+1] = 0.5
		# delt_overlap =  0.5 - (0.4 / np.sqrt(possible_overlaps - 0.44))   # delta (normalized hamming distance) per counter
		oddup = np.floor((possible_overlaps+1)/2)*2  # round odd values up to next even integer, e.g.: [ 0.,  2.,  2.,  4.,  4.,  6.,  6., ...
		delt_overlap = binom.cdf(oddup/2-1, oddup, 0.5) # normalized hamming distance for each overlap,
		# like: [0., 0.25, 0.25, 0.3125, 0.3125, 0.34375, 0.34375 (if odd overlap on top of target 1, add random vector to break ties)
		hdist = np.empty(ncols + 1)  # probability mass function
		for h in range(len(hdist)):  # hamming distance
			phk = binom.pmf(h, ncols, delt_overlap)
			hdist[h] = np.dot(phk, prob_overlap)
		# print("hdist (pmf) sum is %s (should be close to 1)" % np.sum(hdist))
		# assert math.isclose(np.sum(pmf), 1.0), "hdist sum is not equal to 1, is: %s" % np.sum(pmf)
		self.match_hamming_distribution = hdist
		self.prob_overlap = prob_overlap
		self.delt_overlap = delt_overlap
		# print("sum match_hamming_distribution=%s" % sum(self.match_hamming_distribution))
		p_err = self.compute_overall_perr(self.match_hamming_distribution, d)
		self.p_err = p_err

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


class Bsa_sample():
	# compute analytical fraction error of binarized sdm using sampling which should work for nact > 1
	# This is unlike the Binarized_sdm_analytical class (above) which only works when nact is 1.

	def __init__(self, nrows, ncols, nact, k, d, epochs=100):
		# nrows is number of rows (hard locations) in the SDM
		# ncols is the number of columns
		# nact is activaction count
		# k - number of items stored
		# d - size of item memory
		assert nact==1, "only implemented for nact==3 now"
		num_possible_overlaps = k - 1 # number of possible overlaps onto target item (1 row)
		possible_overlaps = np.arange(num_possible_overlaps+1)
		prob_one_trial_overlap = 1 / nrows
		# nai = np.arange(nact)
		# prob_one_trial_overlap = 1-np.prod((nrows-nact-nai)/nrows) # e.g. 1-((nrows-nact)/nrows * (nrows-nact-1)/nrows * (nrows-nact-2)/nrows)
		prob_overlap = binom.pmf(possible_overlaps, num_possible_overlaps, prob_one_trial_overlap )
		# delt_overlap =  0.5 - (0.4 / np.sqrt(possible_overlaps - 0.44))   # delta (normalized hamming distance) per counter
		oddup = np.floor((possible_overlaps+1)/2)*2  # round odd values up to next even integer, e.g.: [ 0.,  2.,  2.,  4.,  4.,  6.,  6., ...
		delt_overlap = binom.cdf(oddup/2-1, oddup, 0.5) # normalized hamming distance for each overlap,
		# like: [0., 0.25, 0.25, 0.3125, 0.3125, 0.34375, 0.34375 (if odd overlap on top of target 1, add random vector to break ties)
		hdist = np.zeros(ncols + 1)  # probability mass function
		rng = np.random.default_rng()
		possible_hammings = np.arange(ncols+1)
		for i in range(epochs*1000):
			samples = rng.choice(num_possible_overlaps+1, size=nact, replace=False, p=prob_overlap, shuffle=False)
			# pc = 1 - delt_overlap[samples] # probability correct
			# er = delt_overlap[samples]  # probability error
			# pvote = pc[0]*pc[1]*pc[2] + er[0]*pc[1]*pc[2] + pc[0]*er[1]*pc[2] + pc[0]*pc[1]*er[2]
			# delta = 1 - pvote
			delta = delt_overlap[samples][0]
			hdist += binom.pmf(possible_hammings, ncols, delta)
		hdist = hdist / hdist.sum()  # normalize
		# print("hdist (pmf) sum is %s (should be close to 1)" % np.sum(hdist))
		# assert math.isclose(np.sum(pmf), 1.0), "hdist sum is not equal to 1, is: %s" % np.sum(pmf)
		self.match_hamming_distribution = hdist
		self.prob_overlap = prob_overlap
		self.delt_overlap = delt_overlap
		# print("sum match_hamming_distribution=%s" % sum(self.match_hamming_distribution))
		p_err = self.compute_overall_perr(self.match_hamming_distribution, d)
		self.p_err = p_err

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

def test_bsa_sample():
	d=100; k=1000; ncols=512; nact = 3; nrows = 300
	bas = Bsa_sample(nrows, ncols, nact, k, d)




if __name__ == "__main__":
	# test sdm and bundle
	test_bsa_sample()
	# main()


