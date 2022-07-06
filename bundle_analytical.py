## Functions for bundle analytical accuracy
import scipy
from scipy.stats import binom
import numpy as np
from scipy.stats import norm
import math
from scipy import special


#Compute expected Hamming distance


def BundleErrorAnalytical(N,D,K,ber=0.0):
	# N - width (number of components) in bundle vector
	# D - number of items in item memory that must be matched
	# K - number of vectors stored in bundle
	# ber - bit error rate (rate of bits flipped).  0 for none.
	deltaHam=  expectedHamming (K) #expected Hamming distance
	deltaBER=(1-2*deltaHam)*ber # contribution of noise to the expected Hamming distance
	dp_hit= deltaHam+deltaBER # total expected Hamming distance
	# est_acc=p_corr (N, D, dp_hit) # expected accuracy # DOES NOT WORK IF N IS LARGE, E.G. 200K OR ABOVE
	# est_error = 1 - est_acc  # expected error
	est_error = p_error_binom(N, D, dp_hit) # expected error
	return est_error

def  expectedHamming (K):
	if (K % 2) == 0: # If even number then break ties so add 1
		K+=1
	deltaHam = 0.5 - (scipy.special.binom(K-1, 0.5*(K-1)))/2**K  # Pentti's formula for the expected Hamming distance
	return deltaHam

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


def gallen(s, per):
	# calculate required length using equation in Gallant paper
	# solved to give length.  (Page 39: "This is equivalent to the probability
	# of a random vector ["per"] being negative when selected from
	# NormalDistibution(D, √((2S-1)D)" solved for D in terms of per and s 
	return round(2*(-1 + 2*s)*special.erfcinv(2*per)**2)

def gallenf(d, k, perf):
	# bundle length from formula in Gallant paper from final probability error
	# (perf) taking into account number of items in bundle (k) and size of item memory (d)
	per = perf/(k*(d-1))
	return gallen(k, per)

def galerr(d, k, w):
	# return expected error of Gallant bundle (not thresholded).
	# d - number of items in item memory (related to "n" in gallant paper, d = n + 1)
	# k - number of items stored in bundle ("s" in gallant paper)
	# w - width (number of components) in bundle, e.g. 10,000
	# calculated from page 39
	# T (Z) = T (√ (D/(2S-1))).
	tz = norm.cdf(-math.sqrt(w/(2*k-1)))
	perr = np.exp((d-1)*k*np.log(1-tz))
	return perr


	# d items combined into trace vector and n items 
