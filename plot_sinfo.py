# script to parse generated file
import re
import sys
import os.path
from fractions import Fraction
import pprint
pp = pprint.PrettyPrinter(indent=4)
import statistics

from scipy.stats import norm
from scipy.stats import binom
from scipy.integrate import quad
import math
import numpy as np
import scipy.integrate as integrate
import warnings
warnings.filterwarnings("error")



def load_data(file_name, xvar):
	assert xvar in ("storage", "pflip")

	fp = open(file_name, 'r')
	while True:
		# scan for line indicating start of data
		line = fp.readline()
		if not line:
			sys.exit("start not found")
		if line == "-----Data starts here-----\n":
			break

	header = fp.readline()
	# assert header == "rid\t%s\tmtype\tmem_len\terror_count\tfraction_error\tprobability_of_error\ttotal_storage_required\n" % xvar
	assert header == ("rid\t%s\tmtype\tmem_len\terror_count\tfraction_error\tprobability_of_error\t"
		"mean_bit_error_count\tstdev_bit_error_count\tmean_dhd\tstdev_dhd\ttotal_storage_required\n" % xvar)

	sdata = {}
	pattern = r"(\d+\.\d+)\t([^\t]+)\t(\w+)\t(\d+)\t(\d+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t(\d+)\n"
	# 1.1	100000	bind	7207	641	0.641	0.6433422825527839	99996
	# 2.2     2.5     sdm     1300    0       0.0     4.427697670830096e-06   672640 # for fdata.txt (bit flips)
	while True:
		# Get next line from file
		line = fp.readline()
		if not line:
			break
		match = re.match(pattern, line)
		if not match:
			sys.exit("Match failed on:\n%s" % line)
		rid, xval, mtype, mlen, error_count, fraction_error, perror, mean_bit_error_count, \
			stdev_bit_error_count, mean_dhd, stdev_dhd, storage_required = match.group(1,2,3,4,5,6,7,8,9,10,11,12)

		# add new line to sdata dictionary

		if mtype not in sdata:
			sdata[mtype] = {}
		xval = int(xval) if xvar == "storage" else float(xval)   # xval is either storage (bytes) or bits flipped (%)
		perror = float(perror)
		mean_bit_error_count = float(mean_bit_error_count)
		stdev_bit_error_count = float(stdev_bit_error_count)
		mean_dhd = float(mean_dhd)
		stdev_dhd = float(stdev_dhd)
		if xval not in sdata[mtype]:
			sdata[mtype][xval] = {"pmin":perror, "pmax":perror, "recs":[], "pelist": [perror],
			"mean_bit_error_counts": [mean_bit_error_count], "stdev_bit_error_counts": [stdev_bit_error_count],
			"mean_dhds":[mean_dhd], "stdev_dhds":[stdev_dhd]}
		else:
			if perror < sdata[mtype][xval]["pmin"]:
				sdata[mtype][xval]["pmin"] = perror
			elif perror > sdata[mtype][xval]["pmax"]:
				sdata[mtype][xval]["pmax"] = perror
			sdata[mtype][xval]["pelist"].append(perror)
			sdata[mtype][xval]["mean_bit_error_counts"].append(mean_bit_error_count)
			sdata[mtype][xval]["stdev_bit_error_counts"].append(stdev_bit_error_count)
			sdata[mtype][xval]["mean_dhds"].append(mean_dhd)
			sdata[mtype][xval]["stdev_dhds"].append(stdev_dhd)
		info = {"error_count":int(error_count), "fraction_error":float(fraction_error), "perror":perror,
		   "storage_required":int(storage_required), "mlen":int(mlen), "error_count":int(error_count),
		   "mean_bit_error_count": mean_bit_error_count, "stdev_bit_error_count":stdev_bit_error_count,
		   "mean_dhd":mean_dhd,"stdev_dhd":stdev_dhd }
		sdata[mtype][xval]["recs"].append(info)
	fp.close()
	return sdata


# pp.pprint(sdata)

def compute_plotting_data(sdata, xvar):
	# create arrays used for plotting
	xvals = {}
	yvals = {}
	ebar = {}
	add_theoretical_storage = xvar == "storage"
	add_theoretical_pflips = xvar == "pflip"
	bit_error_counts = {}
	bit_error_counts_ebar = {}
	mean_dhds = {}
	stdev_dhds = {}
	theory_bundle_bit_error_counts = None
	theoretical_bundle_err = None
	for mtype in sdata:
		xvals[mtype] = sorted(list(sdata[mtype].keys()))  # will be sorted storage (or pflips)
		yvals[mtype] = []
		ebar[mtype] = []
		bit_error_counts[mtype] = []
		bit_error_counts_ebar[mtype] = []
		mean_dhds[mtype] = []
		stdev_dhds[mtype] = []
		for xval in xvals[mtype]:
			# pmid = (sdata[mtype][xval]["pmin"] + sdata[mtype][xval]["pmax"]) / 2.0
			pmid = statistics.mean(sdata[mtype][xval]["pelist"])
			yvals[mtype].append(pmid)
			# prange = sdata[mtype][xval]["pmax"] - sdata[mtype][xval]["pmin"]
			prange = statistics.stdev(sdata[mtype][xval]["pelist"])
			ebar[mtype].append(prange / 2)
			# save bit_count_errors
			mean_bit_error_count = statistics.mean(sdata[mtype][xval]["mean_bit_error_counts"])
			stdev_bit_error_count = statistics.mean(sdata[mtype][xval]["stdev_bit_error_counts"])
			mean_dhd = statistics.mean(sdata[mtype][xval]["mean_dhds"])
			stdev_dhd = statistics.mean(sdata[mtype][xval]["stdev_dhds"])
			bit_error_counts[mtype].append(mean_bit_error_count)
			bit_error_counts_ebar[mtype].append(stdev_bit_error_count / 2)
			mean_dhds[mtype].append(mean_dhd)
			stdev_dhds[mtype].append(stdev_dhd / 2)
		if add_theoretical_storage:
			storage_lengths = get_storage_lengths(xvals, mtype, sdata)
			if mtype == "bind":
				theoretical_bundle_err = [ compute_theoretical_error(sl, mtype) for sl in storage_lengths]
				theory_sdm_bit_error_counts = None
				theory_bundle_bit_error_counts = [compute_theoretical_bundle_bit_error_count(sl) for sl in storage_lengths]
			else:
				assert mtype == "sdm"
				print("** == sdm storage lengths=%s" % storage_lengths)
				print("xvals['sdm']=%s" % xvals["sdm"])
				print("yvals['sdm']=%s" % yvals["sdm"])
				print("ebars['sdm']=%s" % ebar["sdm"])
				# theoretical_sdm_err = [ compute_theoretical_error(sl, mtype) for sl in storage_lengths]
				theoretical_sdm_err = [ compute_theoretical_sdm_error(sl) for sl in storage_lengths]
				theory_sdm_bit_error_counts = [compute_theoretical_sdm_bit_error_count(sl) for sl in storage_lengths]
		elif add_theoretical_pflips and mtype == "sdm":
			# pflips = get_storage_lengths(xvals, mtype, sdata)
			pflips = xvals[mtype]
			# import pdb; pdb.set_trace()
			theoretical_sdm_err = None
			theory_sdm_bit_error_counts = None #[compute_theoretical_sdm_bit_error_count_from_pflips(pf) for pf in pflips]
		elif add_theoretical_pflips and mtype == "bind":
			# get length of bundle
			xval0 = xvals[mtype][0]  # first xval
			bl = sdata[mtype][xval0]["recs"][0]["mlen"]  # get length of superposition vector
			pflips = xvals[mtype]
			theoretical_bundle_err = [ compute_theoretical_error(bl, mtype, pf) for pf in pflips]
		else:
			theoretical_sdm_err = None
			theory_sdm_bit_error_counts = None

	plotting_data = {"xvals": xvals, "yvals":yvals, "ebar":ebar,
		"theoretical_bundle_err":theoretical_bundle_err,
		"theoretical_sdm_err":theoretical_sdm_err,
		"bit_error_counts":bit_error_counts,
		"bit_error_counts_ebar":bit_error_counts_ebar,
		"theory_sdm_bit_error_counts": theory_sdm_bit_error_counts,
		"theory_bundle_bit_error_counts": theory_bundle_bit_error_counts,
		"mean_dhd":mean_dhds, "stdev_dhd":stdev_dhds,
		}
	return plotting_data



def normal_dist_wrong(x , mean , sd):
	# from https://www.askpython.com/python/normal-distribution
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density

def normal_dist(x , mean , sd):
	# from https://www.askpython.com/python/normal-distribution
	# but fixed
    prob_density = (1.0 / math.sqrt(2.0*np.pi*sd*sd)) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density

def test_pdf_integrate(h, mean, sd):
	# test integrating pdf
	weight = normal_dist(h, mean, sd)
	return weight

def frady_eq_integrand(h, mean_hamming_match, sd_hamming_match, mean_hamming_distractor, sd_hamming_distractor):
	# equation based on Frady, 2018 2.12
	# h is the variable for integeration.  Here I think it's related to hamming distance, but not in the Frady paper
	# weight = normal_dist(h, mean_hamming_match, sd_hamming_match)
	weight = norm(mean_hamming_match, sd_hamming_match).pdf(h)
	sum_distractors_less = norm(mean_hamming_distractor, sd_hamming_distractor).cdf(h)
	prob_correct_one = 1.0 - sum_distractors_less
	num_distractors = 99
	prob_correct = weight * (prob_correct_one ** num_distractors)
	return prob_correct

def compute_theoretical_sdm_bit_error_count(sl):
	# compute theoretical sdm bit error based on Pentti's book chapter
	# sl is the storage length, number of rows in the SDM
	k = 1000  # number of items stored in SDM
	word_length = 512
	sdm_num_rows = sl
	sdm_activation_count = get_sdm_activation_count(sl, k)  # round(sdm_num_rows / 100)  # used in hdfsa.py
	pact = sdm_activation_count / sdm_num_rows
	mean = sdm_activation_count
	num_entities_stored = k
	# following from Pentti's chapter - gives error a little more than found
	standard_deviation = math.sqrt(mean * (1 + pact * num_entities_stored * (1.0 + pact*pact * sdm_num_rows)))
	# following replaces T with T -1
	# standard_deviation = math.sqrt(mean * (1 + pact * (num_entities_stored - 1) * (1.0 + pact*pact * sdm_num_rows)))
	# following same as above, but with the 1.0 replaced by zero, for variance of sum zero - DOES NOT WORK
	# standard_deviation = math.sqrt(mean * (1 + pact * num_entities_stored * (0.0 + pact*pact * sdm_num_rows)))
	# average_overlap = ((num_entities_stored - 1) * sdm_activation_count) * (sdm_activation_count / sdm_num_rows)
		# standard_deviation = math.sqrt(average_overlap * 0.5 * (1 - 0.5)) # compute variance assuming binomonal distribution
	probability_single_bit_failure = norm(0, 1).cdf(-mean/standard_deviation)
	expected_error_count = probability_single_bit_failure * word_length
	print("sl=%s, probability_single_bit_failure=%s, expected_error_count=%s" % (sl, 
		probability_single_bit_failure, expected_error_count))
	return expected_error_count

def compute_theoretical_sdm_bit_error_count_from_pflips(pf):
	# compute theoretical bit count error based on difference of two distributions
	# this does not work.
	k = 1000  # number of items stored in SDM
	word_length = 512
	sdm_num_rows = 1724  # should get from file or assert
	sdm_activation_count = get_sdm_activation_count(m, k) # round(sdm_num_rows / 100)  # used in hdfsa.py
	noise_percent = pf
	fraction_flipped = noise_percent / 100.0
	fraction_upright = 1.0 - fraction_flipped
	nact_flipped = sdm_activation_count * fraction_flipped
	nact_upright = sdm_activation_count * fraction_upright
	print("pf=%s, fraction_flipped=%s, fraction_upright=%s, nact_flipped=%s, nact_upright=%s" % (pf,
		fraction_flipped, fraction_upright, nact_flipped, nact_upright))
	mean_flipped, variance_flipped = compute_distribution_for_specific_nact(nact_flipped, sdm_num_rows, k)
	mean_upright, variance_upright = compute_distribution_for_specific_nact(nact_upright, sdm_num_rows, k)
	mean_combined = fraction_upright * mean_upright - fraction_flipped * mean_flipped
	# following is not sufficient
	variance_combined = (fraction_upright * variance_upright + fraction_flipped * variance_flipped)
	# from: https://stats.stackexchange.com/questions/205126/standard-deviation-for-weighted-sum-of-normal-distributions
	variance_combined += fraction_flipped * fraction_upright * (mean_flipped - mean_upright)**2
	print("mean_flipped=%s, variance_flipped=%s, mean_upright=%s, variance_upright=%s, mean_combined=%s, variance_combined=%s" % (
		mean_flipped, variance_flipped, mean_upright, variance_upright, mean_combined, variance_combined))
	standard_deviation_combined = math.sqrt(variance_combined)
	probability_single_bit_failure = norm(0, 1).cdf(-mean_combined/standard_deviation_combined)
	# percent_expected_correct = round((1 - probability_single_bit_failure) * 100.0, 1)
	# return percent_expected_correct
	expected_error_count = probability_single_bit_failure * word_length
	return expected_error_count

def compute_distribution_for_specific_nact(nact, sdm_num_rows, num_items_stored):
	# compute mean and standard distribution for a specific activation count
	pact = nact / sdm_num_rows
	mean = nact
	sdm_num_rows = sdm_num_rows
	variance = mean * (1 + pact * num_items_stored * (1.0 + pact*pact * sdm_num_rows))
	return (mean, variance)

def get_sdm_activation_count(m, k):
	# compute number of rows in sdm to be active
	# m is number of rows in sdm.  k is number of items being stored (1000)
	# nact = round(sdm_num_rows / 100)  # originally used in hdfsa.py
	nact = round( m/((2*m*k)**(1/3)) )
	return nact

def compute_theoretical_error(sl, mtype, pflip=0):
	# compute theoretical error based on storage length.  if mtype bind (bundle) sl is bundle length (bl)
	# if mtype is sdm, sl is number of rows in SDM
	# pflip is the probability (percent) of a bit flip (used for bundle only)
	k = 1000  # number of items stored in bundle
	assert mtype in ("sdm", "bind")
	num_distractors = 99
	if mtype == "bind":
		delta = 0.5 - 0.4 / math.sqrt(k - 0.44)  # from Pentti's paper
		if pflip > 0:
			delta += (1-2*delta)*pflip/100.0
		bl = sl
		print("bundle, bl=%s, delta=%s" % (bl, delta))
	else:
		# following from page 15 of Pentti's book chapter
		sdm_num_rows = sl
		sdm_activation_count = get_sdm_activation_count(sdm_num_rows, k) # round(sdm_num_rows / 100)  # used in hdfsa.py
		pact = sdm_activation_count / sdm_num_rows
		mean = sdm_activation_count
		num_entities_stored = k
		# following from Pentti's chapter
		standard_deviation = math.sqrt(mean * (1 + pact * num_entities_stored * (1.0 + pact*pact * sdm_num_rows)))
		average_overlap = ((num_entities_stored - 1) * sdm_activation_count) * (sdm_activation_count / sdm_num_rows)
		# standard_deviation = math.sqrt(average_overlap * 0.5 * (1 - 0.5)) # compute variance assuming binomonal distribution
		probability_single_bit_failure = norm(0, 1).cdf(-mean/standard_deviation)
		# probability_single_bit_failure = norm(mean, standard_deviation).cdf(0.0)
		delta = probability_single_bit_failure
		print("sdm num_rows=%s,act_count=%s,pact=%s,average_overlap=%s,mean=%s,std=%s,probability_single_bit_failure=%s" % (
			sdm_num_rows, sdm_activation_count, pact, average_overlap, mean, standard_deviation, probability_single_bit_failure))
		bl = 512  # 512 bit length words used for sdm

	"""
	# <<<following use mean and sd distributions without explicit arrays>>>
	mean_hamming_match = delta * bl
	variance_hamming_match = delta * (1 - delta) * bl
	sd_hamming_match = math.sqrt(variance_hamming_match)
	mean_hamming_distractor = 0.5 * bl
	variance_distractor = 0.5 * (1 - 0.5) * bl
	sd_hamming_distractor = math.sqrt(variance_distractor)
	match_sum = quad(test_pdf_integrate, -bl/4, bl, args=(mean_hamming_match, sd_hamming_match ))
	distractor_sum = quad(test_pdf_integrate, 0, bl, args=(mean_hamming_distractor, sd_hamming_distractor))
	print("mean_hamming_match=%s, sd_hamming_match=%s, mean_hamming_distractor=%s, sd_hamming_distractor=%s, match_sum=%s"
		", distractor_sum=%s" % (
		mean_hamming_match, sd_hamming_match, mean_hamming_distractor, sd_hamming_distractor, match_sum[0], distractor_sum[0]))
	I = quad(frady_eq_integrand, -bl/4, bl, args=(
			mean_hamming_match, sd_hamming_match, mean_hamming_distractor, sd_hamming_distractor))
	prob_correct = I[0]
	prob_error = 1.0 - prob_correct
	# import pdb; pdb.set_trace()
	return prob_error
	"""

	# following uses delta only with explicit arrays and binom distribution
	# match_hamming_distribution = np.empty(bl, dtype=np.float64)
	# distractor_hamming_distribution = np.empty(bl, dtype=np.float64)
	match_hamming_distribution = np.empty(bl, dtype=np.double)
	distractor_hamming_distribution = np.empty(bl, dtype=np.double)
	if mtype == "bind":
		# to save time over binom
		mean_hamming_match = delta * bl
		variance_hamming_match = delta * (1 - delta) * bl
		sd_hamming_match = math.sqrt(variance_hamming_match)
		mean_hamming_distractor = 0.5 * bl
		variance_distractor = 0.5 * (1 - 0.5) * bl
		sd_hamming_distractor = math.sqrt(variance_distractor)
	for h in range(bl):
		if mtype == "bind":
			# use normal_dist to save time, faster than binom for long vectors
			match_hamming_distribution[h] = normal_dist(h, mean_hamming_match, sd_hamming_match)
			distractor_hamming_distribution[h] = normal_dist(h, mean_hamming_distractor, sd_hamming_distractor)
		else:
			match_hamming_distribution[h] = binom.pmf(h, bl, delta)
			distractor_hamming_distribution[h] = binom.pmf(h, bl, 0.5)
	show_distributions = False
	if mtype == "sdm" and show_distributions:
		plot_dist(match_hamming_distribution, "match_hamming_distribution, delta=%s" % delta)
		plot_dist(distractor_hamming_distribution, "distractor_hamming_distribution, delta=%s" % delta)
	# now calculate total error
	sum_distractors_less = 0.0
	prob_correct = 0.0
	for h in range(0, bl):
		try:
			sum_distractors_less += distractor_hamming_distribution[h]
			prob_correct_one = 1.0 - sum_distractors_less
			prob_correct += match_hamming_distribution[h] * (prob_correct_one ** num_distractors)
		except RuntimeWarning as err:
			print ('Warning at h=%s, %s' % (h, err))
			print("sum_distractors_less=%s" % sum_distractors_less)
			print("prob_correct_one=%s" % prob_correct_one)
			print("match_hamming_distribution[h] = %s" % match_hamming_distribution[h])
			import pdb; pdb.set_trace()

			sys.exit("aborting")
	prob_error = 1.0 - prob_correct
	# import pdb; pdb.set_trace()
	return prob_error

def compute_theoretical_sdm_error(sl):
	# this routine is for convience for trying different methods
	method = "fraction"
	if method == "frady":
		return compute_theoretical_sdm_frady_error(sl)
	elif method == "fraction":
		return compute_theoretical_sdm_fraction_error(sl)
	elif method == "original":
		mtype = "sdm"
		return compute_theoretical_error(sl, mtype)
	else:
		sys.exit("Invalid method specified for compute_theoretical_sdm_error")


def compute_theoretical_sdm_frady_error(sl):
	# compute theoretical sdm error based on Frady sequence paper equations
	k = 1000  # number of items stored in bundle
	codebook_length = 100
	# following from page 15 of Pentti's book chapter
	sdm_num_rows = sl
	sdm_activation_count = get_sdm_activation_count(sdm_num_rows, k) # round(sdm_num_rows / 100)  # used in hdfsa.py
	pact = sdm_activation_count / sdm_num_rows
	mean = sdm_activation_count
	num_entities_stored = k
	# following from Pentti's chapter
	standard_deviation = math.sqrt(mean * (1 + pact * num_entities_stored * (1.0 + pact*pact * sdm_num_rows)))
	average_overlap = ((num_entities_stored - 1) * sdm_activation_count) * (sdm_activation_count / sdm_num_rows)
	# standard_deviation = math.sqrt(average_overlap * 0.5 * (1 - 0.5)) # compute variance assuming binomonal distribution
	probability_single_bit_failure = norm(0, 1).cdf(-mean/standard_deviation)
	# probability_single_bit_failure = norm(mean, standard_deviation).cdf(0.0)
	delta = probability_single_bit_failure
	print("sdm num_rows=%s,act_count=%s,pact=%s,average_overlap=%s,mean=%s,std=%s,probability_single_bit_failure=%s" % (
		sdm_num_rows, sdm_activation_count, pact, average_overlap, mean, standard_deviation, probability_single_bit_failure))
	bl = 512  # 512 bit length words used for sdm
	prob_correct = p_corr(bl, codebook_length, delta)
	return 1-prob_correct


def compute_theoretical_sdm_fraction_error(sl):
	# compute theoretical recall error using equation in Frady paper implemented using Fractions
	# sl is the sdm_num_rows
	sdm_num_rows = sl
	k = 1000  # number of items stored in bundle
	codebook_length = 100
	sdm_activation_count = get_sdm_activation_count(sdm_num_rows, k) # round(sdm_num_rows / 100)  # used in hdfsa.py
	pact = sdm_activation_count / sdm_num_rows
	mean = sdm_activation_count
	num_entities_stored = k
	# following from Pentti's chapter
	standard_deviation = math.sqrt(mean * (1 + pact * num_entities_stored * (1.0 + pact*pact * sdm_num_rows)))
	average_overlap = ((num_entities_stored - 1) * sdm_activation_count) * (sdm_activation_count / sdm_num_rows)
	# standard_deviation = math.sqrt(average_overlap * 0.5 * (1 - 0.5)) # compute variance assuming binomonal distribution
	probability_single_bit_failure = norm(0, 1).cdf(-mean/standard_deviation)
	# probability_single_bit_failure = norm(mean, standard_deviation).cdf(0.0)
	delta = probability_single_bit_failure
	print("sdm num_rows=%s,act_count=%s,pact=%s,average_overlap=%s,mean=%s,std=%s,probability_single_bit_failure=%s" % (
		sdm_num_rows, sdm_activation_count, pact, average_overlap, mean, standard_deviation, probability_single_bit_failure))
	bl = 512  # 512 bit length words used for sdm
	wl = bl  # word length
	num_distractors = codebook_length - 1
	pflip = Fraction(int(delta*100), 100)  # probability of bit flip
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
	error = (float((1 - pcor) * 100))  # convert to percent
	return error


	# Calculate analytical accuracy of the encoding according to the equation for p_corr from 2017 IEEE Tran paper
def  p_corr (N, D, dp_hit):
    dp_rej=0.5
    var_hit = 0.25*N
    var_rej=var_hit
    range_var=10 # number of std to take into account
    fun = lambda u: (1/(np.sqrt(2*np.pi*var_hit)))*np.exp(-((u-N*(dp_rej-dp_hit) )**2)/(2*(var_hit)))*((norm.cdf((u/np.sqrt(var_rej))))**(D-1) ) # analytical equation to calculate the accuracy  

    acc = integrate.quad(fun, (-range_var)*np.sqrt(var_hit), N*dp_rej+range_var*np.sqrt(var_hit)) # integrate over a range of possible values
    print("p_corr, N=%s, D=%s, dp_hit=%s, return=%s" % (N, D, dp_hit, acc[0]))
    return acc[0]

""" <<old version>>>
	# create arrays of distribution for both
	# match_hamming_distribution = np.empty(bl, dtype=np.float64)
	# distractor_hamming_distribution = np.empty(bl, dtype=np.float64)
	match_hamming_distribution = np.empty(bl, dtype=np.double)
	distractor_hamming_distribution = np.empty(bl, dtype=np.double)
	for h in range(bl):
		match_hamming_distribution[h] = normal_dist(h, mean_hamming_match, sd_hamming_match)
		distractor_hamming_distribution[h] = normal_dist(h, mean_hamming_distractor, sd_hamming_distractor)
		# match_hamming_distribution[h] = binom.pmf(h, bl, delta)
		# distractor_hamming_distribution[h] = binom.pmf(h, bl, 0.5)
	# now calculate total error
	sum_distractors_less = 0.0
	prob_correct = 0.0
	for h in range(1, bl):
		try:
			sum_distractors_less += distractor_hamming_distribution[h-1]
			prob_correct_one = 1.0 - sum_distractors_less
			prob_correct += match_hamming_distribution[h] * (prob_correct_one ** num_distractors)
		except RuntimeWarning as err:
			print ('Warning at h=%s, %s' % (h, err))
			print("sum_distractors_less=%s" % sum_distractors_less)
			print("prob_correct_one=%s" % prob_correct_one)
			print("match_hamming_distribution[h] = %s" % match_hamming_distribution[h])
			import pdb; pdb.set_trace()

			sys.exit("aborting")
	prob_error = 1.0 - prob_correct
	# import pdb; pdb.set_trace()
	return prob_error
	<<< end old version>>>"""


def compute_theoretical_bundle_error(bl):
	# compute theoretical error based on bundle length (bl)
	k = 1000  # number of items stored in bundle
	num_distractors = 99
	delta = 0.5 - 0.4 / math.sqrt(k - 0.44)  # from Pentti's paper
	mean_hamming_match = delta * bl
	variance_hamming_match = delta * (1 - delta) * bl
	sd_hamming_match = math.sqrt(variance_hamming_match)
	mean_hamming_distractor = 0.5 * bl
	variance_distractor = 0.5 * (1 - 0.5) * bl
	sd_hamming_distractor = math.sqrt(variance_distractor)
	# create arrays of distribution for both
	match_hamming_distribution = np.empty(bl, dtype=np.float64)
	distractor_hamming_distribution = np.empty(bl, dtype=np.float64)
	for h in range(bl):
		match_hamming_distribution[h] = normal_dist(h, mean_hamming_match, sd_hamming_match)
		distractor_hamming_distribution[h] = normal_dist(h, mean_hamming_distractor, sd_hamming_distractor)
	# now calculate total error
	sum_distractors_less = 0.0
	prob_correct = 0.0
	for h in range(1, bl):
		try:
			sum_distractors_less += distractor_hamming_distribution[h-1]
			prob_correct_one = 1.0 - sum_distractors_less
			prob_correct += match_hamming_distribution[h] * (prob_correct_one ** num_distractors)
		except RuntimeWarning as err:
			print ('Warning at h=%s, %s' % (h, err))
			print("sum_distractors_less=%s" % sum_distractors_less)
			print("prob_correct_one=%s" % prob_correct_one)
			print("match_hamming_distribution[h] = %s" % match_hamming_distribution[h])
			import pdb; pdb.set_trace()

			sys.exit("aborting")
	prob_error = 1.0 - prob_correct
	# import pdb; pdb.set_trace()
	return prob_error


def compute_theoretical_bundle_error_simple_1(bl):
	# compute theoretical error based on bundle length (bl)
	# *Version is incorrect*
	# Assumes that prob error with one can be raised to power number of distractors, to
	# get probability error for all (is incorrect)
	k = 1000  # number of items stored in bundle
	delta = 0.5 - 0.4 / math.sqrt(k - 0.44)  # from Pentti's paper
	mean_hamming_match = delta * bl
	variance_hamming_match = delta * (1 - delta) * bl
	mean_hamming_distractor = 0.5 * bl
	variance_distractor = 0.5 * (1 - 0.5) * bl
	# make mean and variance of difference distribution
	diff_dist_mean = mean_hamming_distractor - mean_hamming_match
	diff_dist_variance = variance_hamming_match + variance_distractor
	# want probability diff_dist distribution is less than zero (meaning, hamming distance of distractor less than
	# hamming distance of match)
	loc = diff_dist_mean
	scale = math.sqrt(diff_dist_variance)
	eperr = norm.pdf(0, loc, scale)
	pcorr_one = 1.0 - eperr 
	pcorr_all = pcorr_one ** 99
	perror_all = 1 - pcorr_all
	return perror_all

def compute_theoretical_bundle_bit_error_count(sl):
	# use pentti equation to get single bit error then multiply by bundle length
	k = 1000  # number of items stored in bundle
	delta = 0.5 - 0.4 / math.sqrt(k - 0.44)  # from Pentti's paper
	return delta * sl

def get_storage_lengths(xvals, mtype, sdata):
	bundle_lengths = []
	for xval in xvals[mtype]:
		bundle_lengths.append(sdata[mtype][xval]["recs"][0]["mlen"])
	return(bundle_lengths)


# import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def make_plots(plotting_data, xvar, scale):
	xvals = plotting_data["xvals"]
	yvals = plotting_data["yvals"]
	ebar = plotting_data["ebar"]
	theoretical_bundle_err = plotting_data["theoretical_bundle_err"]
	theoretical_sdm_err = plotting_data["theoretical_sdm_err"]
	bit_error_counts = plotting_data["bit_error_counts"]
	bit_error_counts_ebar = plotting_data["bit_error_counts_ebar"]
	theory_sdm_bit_error_counts = plotting_data["theory_sdm_bit_error_counts"]
	theory_bundle_bit_error_counts = plotting_data["theory_bundle_bit_error_counts"]
	mean_dhd = plotting_data["mean_dhd"]
	stdev_dhd = plotting_data["stdev_dhd"]


	# plot bit_error_counts separately
	for mtype in xvals:
		fig = plt.figure()
		label = "bit_error_counts for %s" % mtype
		# bit error count is hamming distance to correct item in item memory
		plt.errorbar(xvals[mtype], bit_error_counts[mtype], yerr=bit_error_counts_ebar[mtype], label="found", fmt="-o")
		# print("%s xvals: %s" % (mtype, xvals[mtype]))
		# print("%s be_counts: %s" % (mtype, bit_error_counts[mtype]))	
		# plt.errorbar(xvals[mtype], mean_dhd[mtype], yerr=stdev_dhd[mtype], label="found", fmt="-o")
		if mtype == "sdm" and xvar == "pflip":
			overlap_vals = [55.8, 62.97, 78.8, 92.6, 105.4, 130.5, 149.5, 172.0, 199.1, 227.3, 243.7]
			plt.errorbar(xvals[mtype], overlap_vals, label="overlap_output", fmt="-o")
		if mtype == "sdm" and theory_sdm_bit_error_counts is not None:
			plt.errorbar(xvals[mtype], theory_sdm_bit_error_counts, label="sdm theory", fmt="-o")
		if mtype == "bind" and theory_bundle_bit_error_counts is not None:
			print ("in make plots: theory_bundle_bit_error_counts=%s" % theory_bundle_bit_error_counts)
			plt.errorbar(xvals[mtype], theory_bundle_bit_error_counts, label="bundle bit count theory", fmt="-o")
		plt.title(label)
		plt.legend(loc='upper left')
		plt.grid(which='both', axis='both')
		if mtype == "sdm":
			plt.yticks(np.arange(50.0, 275.0, 25.0))
		plt.show

	# fig, ax = plt.subplots()
	fig = plt.figure()
	for mtype in xvals:
		label = "Superposition vector" if mtype == "bind" else "SDM"
		yerr = None # ebar[mtype] to include error bars
		plt.errorbar(xvals[mtype], yvals[mtype], yerr=yerr, label=label, fmt="-o")
		if theoretical_bundle_err and mtype == "bind":
			plt.errorbar(xvals["bind"], theoretical_bundle_err, label="Superposition theory", fmt="-",
				linestyle=':', linewidth=6, alpha=0.7)
		if theoretical_sdm_err and mtype == "sdm":
			plt.errorbar(xvals["bind"], theoretical_sdm_err, label="SDM theory", fmt="-o", linestyle='dashed')


	title = "Fraction error vs. storage size (bytes)" if xvar == "storage" else "Fraction error vs. percent bits (or counters) flipped" 
	# plt.title(title)
	xlabel = "Storage (bytes)" if xvar == "storage" else '% bits (or counters) flipped'
	plt.xlabel(xlabel)
	if xvar == "storage":
		xaxis_labels = ["100k", "200k", "300k", "400k", "500k", "600k", "700k", "800k", "900k", "10^6" ]
		plt.xticks(xvals[mtype],xaxis_labels)
	ylabel = "Fraction error"
	plt.ylabel(ylabel)

	# plt.axes().yaxis.get_ticklocs(minor=True)

	# Initialize minor ticks
	# plt.axes().yaxis.minorticks_on()

	log_plot = scale == "log"
	if log_plot:
		plt.yscale('log')
		title += " (log scale)"
		loc = 'lower left' if xvar == "storage" else 'lower right'
	else:
		loc = 'upper right' if xvar == "storage" else 'upper left'
	plt.title(title)
	plt.legend(loc=loc)
	plt.grid()
	plt.show()

def plot_dist(dist, title):
	plt.plot(dist)
	plt.title(title)
	plt.show()

def display_usage_and_exit():
	sys.exit("Usage %s <file_name.txt> (linear | log)" % sys.argv[0])

def main():
	if len(sys.argv) != 3:
		display_usage_and_exit()
	file_name = sys.argv[1]
	assert os.path.isfile(file_name), "file %s not found" % file_name
	basename = os.path.basename(file_name)
	if basename[0] == "s":
		# assume 'sdata*.txt' - storage vs error
		xvar = "storage"
	elif basename[0] == "f":
		# assume 'fdata*.txt' - bit flips vs error
		xvar = "pflip"  # percent flip
	else:
		sys.exit("File name first character should be 'f' or 's': found: %s" % basename[0])
	scale = sys.argv[2]
	if scale not in ("linear", "log"):
		display_usage_and_exit()
	sdata = load_data(file_name, xvar)
	plotting_data = compute_plotting_data(sdata, xvar)
	make_plots(plotting_data, xvar, scale)


main()
