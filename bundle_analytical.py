## Functions for bundle analytical accuracy
import scipy
from scipy.stats import binom
import numpy as np


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
