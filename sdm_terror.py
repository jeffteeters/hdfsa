# show sdm theoretical error

# displays empirical error found by hdfsa.py and also plots theoretical error using equation
# in Frady Sequence paper.  The theoretical error is not matching because this script didn't
# use the python Fractions module to compute the error exactly.

# this script is probably not needed any more.  Saving it, just in case it's useful later.

import statistics

from scipy.stats import norm
from scipy.stats import binom
from scipy.integrate import quad
import math
import numpy as np
import decimal as dc

import warnings
warnings.filterwarnings("error")

def compute_theoretical_error(total_storage, sl, mtype):
	# compute theoretical error based on storage length.  if mtype bind (bundle) sl is bundle length (bl)
	# if mtype is sdm, sl is number of rows in SDM
	print("Total storage=%s, sl=%s" % (total_storage, sl))
	k = 1000  # number of items stored in bundle
	assert mtype in ("sdm", "bind")
	num_distractors = 99
	if mtype == "bind":
		delta = 0.5 - 0.4 / math.sqrt(k - 0.44)  # from Pentti's paper
		bl = sl
		print("bundle, bl=%s, delta=%s" % (bl, delta))
	else:
		# following from page 15 of Pentti's book chapter
		sdm_num_rows = sl
		sdm_activation_count = round(sdm_num_rows / 100)  # used in hdfsa.py
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
	match_hamming_distribution = [] # np.empty(bl, dtype=np.double)
	distractor_hamming_distribution = [] # np.empty(bl, dtype=np.double)
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
			# match_hamming_distribution[h] = binom.pmf(h, bl, delta)
			# distractor_hamming_distribution[h] = binom.pmf(h, bl, 0.5)
			match_hamming_distribution.append(dc.Decimal(binom.pmf(h, bl, delta)))
			distractor_hamming_distribution.append(dc.Decimal(binom.pmf(h, bl, 0.5)))
	show_distributions = False
	if mtype == "sdm" and show_distributions:
		plot_dist(match_hamming_distribution, "match_hamming_distribution, delta=%s" % delta)
		plot_dist(distractor_hamming_distribution, "distractor_hamming_distribution, delta=%s" % delta)
	# now calculate total error
	sum_distractors_less = dc.Decimal(0.0)
	prob_correct = dc.Decimal(0.0)
	one = dc.Decimal(1.0)
	ctx = dc.getcontext()
	# print("ctx=%s" % ctx)
	for h in range(0, bl):
		try:
			# sum_distractors_less += distractor_hamming_distribution[h]
			sum_distractors_less = ctx.add(sum_distractors_less, distractor_hamming_distribution[h])
			# prob_correct_one = 1 - sum_distractors_less
			prob_correct_one = ctx.subtract(one, sum_distractors_less)
			# prob_correct += match_hamming_distribution[h] * (prob_correct_one ** num_distractors)
			prob_correct = ctx.add(prob_correct, ctx.multiply(match_hamming_distribution[h], ctx.power(prob_correct_one, num_distractors)))
		except RuntimeWarning as err:
			print ('Warning at h=%s, %s' % (h, err))
			print("sum_distractors_less=%s" % sum_distractors_less)
			print("prob_correct_one=%s" % prob_correct_one)
			print("match_hamming_distribution[h] = %s" % match_hamming_distribution[h])
			import pdb; pdb.set_trace()

			sys.exit("aborting")
	# prob_error = 1.0 - prob_correct
	prob_error = float(ctx.subtract(one, prob_correct))
	# import pdb; pdb.set_trace()
	return prob_error

def compute_plotting_data():
	# storage lengths in sdata_7.txt
	storage_lengths = [182, 377, 572, 768, 963, 1158, 1353, 1549, 1744, 1939]
	xvals={}
	xvals["sdm"] = [100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]
	mtype = "sdm"
	theoretical_sdm_err = [ compute_theoretical_error(xvals[mtype][i], storage_lengths[i], mtype)
		for i in range(len(storage_lengths))]
	yvals={}
	yvals["sdm"] = [0.00015494224973963938, 1.2461288379968406e-09, 3.4223387577581987e-15, 
		8.884086712156747e-22, 4.625889709542001e-29, 2.9855085634270426e-37, 1.1036213886560766e-41,
		4.492548644599037e-50, 5.233630523674144e-58, 2.3449316690573875e-60]
	ebar={}
	ebar["sdm"] = [1.7062488488909926e-05, 5.259037387679861e-10, 1.7475411979827154e-15, 7.583086336970683e-22,
		2.539007697473842e-29, 1.4548924878836518e-37, 9.557461057310374e-42, 3.851357529423032e-50,
		4.511283777435027e-58, 2.030770395493569e-60]
	plotting_data = {"xvals": xvals, "yvals":yvals, "ebar":ebar,
		"theoretical_sdm_err":theoretical_sdm_err,
		}
	return plotting_data


import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def make_plots(plotting_data, xvar):
	xvals = plotting_data["xvals"]
	yvals = plotting_data["yvals"]
	ebar = plotting_data["ebar"]
	theoretical_sdm_err = plotting_data["theoretical_sdm_err"]


	# fig, ax = plt.subplots()
	fig = plt.figure()
	plt.yscale('log')
	for mtype in xvals:
		label = "superposition vector" if mtype == "bind" else "SDM"
		plt.errorbar(xvals[mtype], yvals[mtype], yerr=ebar[mtype], label=label, fmt="-o")
		plt.errorbar(xvals[mtype], theoretical_sdm_err, label="sdm theory", fmt="-o")


	title = "Fraction error vs storage size (bytes)" if xvar == "storage" else "Fraction error vs percent bits flipped" 
	plt.title(title)
	xlabel = "Storage (bytes)" if xvar == "storage" else '% bits flipped'
	plt.xlabel(xlabel)
	if False and xvar == "storage":
		xaxis_labels = ["100k", "200k", "300k", "400k", "500k", "600k", "700k", "800k", "900k", "1e6" ]
		plt.xticks(xvals[mtype],xaxis_labels)
	ylabel = "Fraction error"
	plt.ylabel(ylabel)

	# plt.axes().yaxis.get_ticklocs(minor=True)

	# Initialize minor ticks
	# plt.axes().yaxis.minorticks_on()

	loc = 'upper right' if xvar == "storage" else 'lower right'
	plt.legend(loc=loc)
	plt.grid()
	plt.show()

def plot_dist(dist, title):
	plt.plot(dist)
	plt.title(title)
	plt.show()


def main():
	xvar = "storage"
	plotting_data = compute_plotting_data()
	make_plots(plotting_data, xvar)


main()
