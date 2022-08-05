# plot empirical fit of error to bundle and sdm

import fast_sdm_empirical
import fast_bundle_empirical
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import linregress
from curve_fit import mdims

mem_info = [
	{
		"name": "bun_k1000_d100_c1#S1",
		"short_name": "S1",
 		"mtype": "bundle",
 		"binarize_counters": True,  # use hamming match
  	},
  	{
		"name": "bun_k1000_d100_c8#S2",
		"short_name": "S2",
 		"mtype": "bundle",
 		"binarize_counters": False,  # used dot product match
  	},
  	{
  		"name": "sdm_k1000_d100_c8_ham#A1",
		"short_name": "A1",
 		"mtype": "sdm",
 		"bits_per_counter": 8,
 		"match_method": "hamming",
 	},
 	{
  		"name": "sdm_k1000_d100_c1_ham#A2",
		"short_name": "A2",
 		"mtype": "sdm",
 		"bits_per_counter": 1,
 		"match_method": "hamming",
 	},
 	{
  		"name": "sdm_k1000_d100_c1_dot#A3",
		"short_name": "A3",
 		"mtype": "sdm",
 		"bits_per_counter": 1,
 		"match_method": "dot",
 	},
 	{
  		"name": "sdm_k1000_d100_c8_dot#A4",
		"short_name": "A4",
 		"mtype": "sdm",
 		"bits_per_counter": 8,
 		"match_method": "dot",
 	},
]

def get_bundle_perr(mi, ie):
	# return mean and standard deviation
	# mi entry in mem_info array (key-value pairs)
	# ip is desired error, range 1 to 9 (10^(-ie))
	global mdims
	epochs = get_epochs(ie, bundle=True)
	if epochs is None:
		# requires too many epochs, don't run
		mean_error = math.nan
		std_error = math.nan
	else:
		assert mi["mtype"] == "bundle"
		name = mi["name"]
		dims = mdims[name]
		size = dims[ie - 1]
		assert size[0] == ie, "First component of %s dims does not match error %s" % (name, ie)
		ncols = size[1]
		binarize_counters = mi["binarize_counters"]
		fbe = Fast_bundle_empirical.Fast_bundle_empirical(ncols, epochs=epochs, binarize_counters=binarize_counters)
		mean_error = fbe.mean_error
		std_error = fbe.std_error
	return (mean_error, std_error)

def get_sdm_perr(mi, ie):
	# return mean and standard deviation of error
	# mi entry in mem_info array (key-value pairs)
	# ie is desired error, range 1 to 9 (10^(-ie))
	global mdims
	epochs = get_epochs(ie, bundle=False)
	if epochs is None:
		# requires too many epochs, don't run
		mean_error = math.nan
		std_error = math.nan
	else:
		assert mi["mtype"] == "sdm"
		name = mi["name"]
		dims = mdims[name]
		size = dims[ie - 1]
		assert size[0] == ie, "First component of %s dims does not match error %s" % (name, ie)
		nrows = size[1]
		nact = size[2]
		bits_per_counter = mi["bits_per_counter"]
		assert mi["match_method"] in ("hamming", "dot")
		threshold_sum = mi["match_method"] == "hamming"
		fse = Fast_sdm_empirical.Fast_sdm_empirical(nrows, ncols, nact, epochs=epochs,
			bits_per_counter=bits_per_counter, threshold_sum=threshold_sum)
		mean_error = fse.mean_error
		std_error = fse.std_error
	return (mean_error, std_error)

def get_epochs(ie, bundle=False):
	# ie is expected error rate, range 1 to 9 (10^(-ie))
	# return number of epochs required or None if not running this one because would require too many epochs
	num_transitions = 1000  # 100 states, 10 choices per state
	desired_fail_count = 100
	minimum_fail_count = 40
	epochs_max = 1000 if not bundle else 200  # bundle takes longer, so use fewer epochs 
	expected_perr = 10^(-ie)  # expected probability of error
	desired_epochs = round(desired_fail_count / (expected_perr *num_transitions))
	if desired_epochs <= epochs_max:
		return desired_epochs
	minimum_epochs = round(minimum_fail_count / (expected_perr *num_transitions))
	if minimum_epochs <= epochs_max:
		return epochs_max
	# requires too many epochs
	return None

def plot_fit(mtype="sdm"):
	assert mtype in ("sdm", "bundle")
	get_perr = get_sdm_perr if mtype=="sdm" else get_bundle_perr
	for mi in mem_info:
		if mi["mtype"] == mtype:
			results = [get_perr(mi, ie) for ie in range(1, 9)]
			mi["results"] = results
	# now make plot





	epochs_needed = round(minimum_fail_count / (expected_perr *num_transitions))
			if epochs_needed > epochs_start:
				if epochs_needed > epochs_max:
					print("num_steps=%s, nrows=%s, expected_perr=%s, epochs_needed=%s; stopping because epochs_max=%s" % (
						num_steps, nrows, expected_perr, epochs_needed, epochs_max))
					break
				epochs = epochs_needed



def optimum_nact(k, m):
	# return optimum nact given number items stored (k) and number of rows (m)
	# from formula given in Pentti paper
	fraction_rows_activated =  1.0 / ((2*m*k)**(1/3))
	nact = round(m * fraction_rows_activated)
	return nact

def predict_sdm_dims(kcon, mcon, k):
	# predict sdm dimensions for each perr power of 10 assuming perr = kcon*10**(mcon*x),
	# or log10(perr) = log10(kcon) + mcon*x
	# or x = nrows = (log10(perr) - log10(kcon)) / mcon
	# k is number of items stored in the sdm (used to compute nact from nrows)
	dims = []
	for i in range(1,10):
		log_perr = -i
		log_kcon = np.log10(kcon)
		nrows = round((log_perr - log_kcon) / mcon)
		nact = optimum_nact(k, nrows)
		dims.append([i, nrows, nact])
	return dims


def main():
	ncols = 512; actions=10; choices=10; states=100
	k = 1000
	d = 100
	num_transitions = choices * states
	epochs=50
	threshold_sum = False
	bits_per_counter=8
	row_start = 20
	row_step = 10
	epochs_start = 40
	minimum_fail_count = 40
	epochs_max = 1000
	# wanted_rows = range(40, 121, 20)
	max_num_steps = 100
	perr = np.empty(max_num_steps, dtype=np.float64)
	rows = np.empty(max_num_steps, dtype=np.float64)
	nacts = np.empty(max_num_steps, dtype=np.int16)
	confidence = np.empty(max_num_steps, dtype=np.float64)  # confidence interval
	nrows = row_start
	expected_perr = None
	for num_steps in range(max_num_steps):  # number of steps saved
		if num_steps >= 2:
			expected_perr = perr[num_steps-1]**2 / perr[num_steps -2]  # expected next perr
			epochs_needed = round(minimum_fail_count / (expected_perr *num_transitions))
			if epochs_needed > epochs_start:
				if epochs_needed > epochs_max:
					print("num_steps=%s, nrows=%s, expected_perr=%s, epochs_needed=%s; stopping because epochs_max=%s" % (
						num_steps, nrows, expected_perr, epochs_needed, epochs_max))
					break
				epochs = epochs_needed
		nact = optimum_nact(k, nrows)
		if nact > 1:
			print("stopping because nact > 1, nrows=%s" % nrows)
			break
		print("num_steps=%s, nrows=%s, nact=%s, expected_perr=%s, epochs=%s" % (num_steps,
			nrows, nact, expected_perr, epochs))
		nacts[num_steps] = nact
		rows[num_steps] = nrows
		fse = fast_sdm_empirical.Fast_sdm_empirical(nrows, ncols, nact, actions=actions, states=states, choices=choices,
			threshold_sum=threshold_sum, bits_per_counter=bits_per_counter, epochs=epochs)
		perr[num_steps] = fse.mean_error
		confidence[num_steps] = (fse.std_error / math.sqrt(epochs)) * 1.96  # 95% confidence interval for the mean (CLM)
		nrows += row_step
		print("epochs=%s, sdm size=%s/%s, perr=%s, std=%s" %(epochs, nrows, nact, fse.mean_error, fse.std_error))

	# fit a line, log10(y) = kcon * 10 ** (mcon * nrows).  Find kcon and mcon  ("con" means constant)
	x = rows[0:num_steps]
	y = perr[0:num_steps]
	ylog = np.log10(y)
	result = linregress(x, y=ylog)
	kcon = 10.0**(result.intercept)
	mcon = result.slope
	assert mcon < 1
	yreg = kcon * 10.0**(mcon *x)
	print("linregress for err=k*exp(-m*x), k=%s, m=%s" % (kcon, mcon))
	dims = predict_sdm_dims(kcon, mcon, k)
	print("predicted sdm_dims for different perr powers of 10 is:\n%s" % dims)


	# make plot
	plt.errorbar(x, y, yerr=confidence[0:num_steps], fmt="-o", label="non-thresholded")
	plt.errorbar(x, yreg, yerr=None, fmt="-", label="line_fit")
	plt.title("sdm error vs size, threshold_sum=%s, bits_per_counter=%s" % (threshold_sum, bits_per_counter))
	plt.xlabel("sdm num rows")
	plt.ylabel("Fraction error")
	plt.yscale('log')
	# xlabels = ["%s/%s" % (rows[i], nacts[i]) for i in range(num_steps)]
	# plt.xticks(rows[0:num_steps], xlabels)
	plt.grid()
	plt.legend(loc='upper right')
	plt.show()


if __name__ == "__main__":
	# compare_sdm_ham_dot()
	main()

