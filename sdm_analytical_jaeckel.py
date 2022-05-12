# functions for sdm accuracy using formula developed by Jaeckel (that uses poisson distribution)

from fractions import Fraction
from scipy.stats import norm
import math

## begin functions for sdm analytical accuracy

# Calculate analytical accuracy of the encoding according to the equation for p_corr from 2017 IEEE Tran paper
# but use python Fractions and the binonomal distribution to reduce roundoff error
# return error rather than p_correct for best accuracy

def p_error_Fraction (N, D, dp_hit):
	wl = N    # word length
	delta = dp_hit
	num_distractors = D - 1
	pflip = Fraction(int(delta*10000), 10000)  # probability of bit flip
	pdist = Fraction(1, 2) # probability of distractor bit matching
	match_hamming = []
	distractor_hamming = []
	for k in range(wl):
		ncomb = Fraction(math.comb(wl, k))
		match_hamming.append(ncomb * pflip ** k * (1-pflip) ** (wl-k))
		distractor_hamming.append(ncomb * pdist ** k * (1-pdist) ** (wl-k))
	# print("sum match_hamming=%s, distractor_hamming=%s" % (sum(match_hamming), sum(distractor_hamming)))
	# if self.env.pvals["show_histograms"]:
	# 	fig = Figure(title="match and distractor hamming distributions for %s%% bit flips" % xval,
	# 		xvals=range(wl), grid=False,
	# 		xlabel="Hamming distance", ylabel="Probability", xaxis_labels=None,
	# 		legend_location="upper right",
	# 		yvals=match_hamming, ebar=None, legend="match_hamming", fmt="-g")
	# 	fig.add_line(distractor_hamming, legend="distractor_hamming", fmt="-m")
	# 	self.figures.append(fig)
	dhg = Fraction(1) # fraction distractor hamming greater than match hamming
	pcor = Fraction(0)  # probability correct
	for k in range(wl):
		dhg -= distractor_hamming[k]
		pcor += match_hamming[k] * dhg ** num_distractors
	# error = (float((1 - pcor) * 100))  # convert to percent
	error = float(1 - pcor) # * 100))  # convert to percent
	return error
	# return float(pcor)

def SdmErrorAnalytical(R,K,D,nact=None,word_length=512,ber=0.0):
	# R - number rows in sdm
	# K - number of vectors stored in sdm
	# D - number of items in item memory that must be matched
	# nact - activation count.  None to compute optimal activaction count based on number of rows (R)
	# word_length - width (number of components) in each word
	# ber - bit error rate (rate of counters flipped).  0 for none.
	# following from page 15 of Pentti's book chapter
	sdm_num_rows = R
	num_entities_stored = K
	sdm_activation_count = get_sdm_activation_count(sdm_num_rows, num_entities_stored) if nact is None else nact 
	pact = sdm_activation_count / sdm_num_rows
	mean = sdm_activation_count
	# following from Pentti's chapter
	standard_deviation = math.sqrt(mean * (1 + pact * num_entities_stored * (1.0 + pact*pact * sdm_num_rows)))
	# average_overlap = ((num_entities_stored - 1) * sdm_activation_count) * (sdm_activation_count / sdm_num_rows)
	# standard_deviation = math.sqrt(average_overlap * 0.5 * (1 - 0.5)) # compute variance assuming binomonal distribution
	probability_single_bit_failure = norm(0, 1).cdf(-mean/standard_deviation)
	# probability_single_bit_failure = norm(mean, standard_deviation).cdf(0.0)
	deltaHam = probability_single_bit_failure
	# print("sdm num_rows=%s,act_count=%s,pact=%s,average_overlap=%s,mean=%s,std=%s,probability_single_bit_failure=%s" % (
	# 	sdm_num_rows, sdm_activation_count, pact, average_overlap, mean, standard_deviation, probability_single_bit_failure))
	# bl = 512  # 512 bit length words used for sdm
	# deltaBER=(1-2*deltaHam)*ber # contribution of noise to the expected Hamming distance
	# dp_hit= deltaHam+deltaBER # total expected Hamming distance
	# perr = 1 - p_corr (word_length, D, dp_hit)
	# perr = 1 - p_corr_Fraction(word_length, D, deltaHam)
	perr = p_error_Fraction(word_length, D, deltaHam)
	return perr

def get_sdm_activation_count(nrows, number_items_stored):
	nact = round(fraction_rows_activated(nrows, number_items_stored)*nrows)
	if nact < 1:
		nact = 1
	return nact

def fraction_rows_activated(m, k):
	# compute optimal fraction rows to activate in sdm
	# m is number rows, k is number items stored in sdm
	return 1.0 / ((2*m*k)**(1/3))
