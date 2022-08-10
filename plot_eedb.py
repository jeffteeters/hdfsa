# plot data in empirical_error.db; that is, plot empirical and
# predicted error rate vs size for both bundle and SDM

import numpy as np
import matplotlib.pyplot as plt
# import sqlite3
import math
# import os.path
# from scipy.stats import linregress
# import pprint
# pp = pprint.PrettyPrinter(indent=4)
from build_eedb import Empirical_error_db


def plot_error_vs_dimension(mtype="sdm"):
	# dimenion is ncols (width) of bundle or nrows
	assert mtype in ("sdm", "bundle")
	edb = Empirical_error_db()
	names = edb.get_memory_names(mtype=mtype)
	for name in names:
		mi = edb.get_minfo(name)
		bits_per_counter = mi["bits_per_counter"]
		match_method = mi["match_method"]
		ndims = len(mi["dims"])
		sizes = np.empty(ndims, dtype=np.uint32)
		empirical_error = np.empty(ndims, dtype=np.float64)
		empirical_clm = np.empty(ndims, dtype=np.float64)
		predicted_error = np.empty(ndims, dtype=np.float64)
		for i in range(ndims):
			dim = mi["dims"][i]
			if mtype == "sdm":
				dim_id, ie, size, ncols, nact, pe, epochs, mean, std = dim
			else:
				dim_id, ie, size, pe, epochs, mean, std = dim
			sizes[i] = size
			predicted_error[i] = pe
			if mean is not None:
				empirical_error[i] = mean
				empirical_clm[i] = std / math.sqrt(epochs) * 1.96  # 95% confidence interval for the mean (CLM)
			else:
				empirical_error[i] = np.nan
				empirical_clm[i] = np.nan
		# plot arrays filled by above
		plt.errorbar(sizes, empirical_error, yerr=empirical_clm, fmt="-o", label=name)
		plt.errorbar(sizes, predicted_error, yerr=None, fmt="o", label="%s - predicted error" % name)
	plt.title("%s empirical vs. predicted error" % mtype)
	xlabel = "SDM num rows" if mtype == "sdm" else "Superposition vector width"
	plt.xlabel(xlabel)
	plt.ylabel("Fraction error")
	plt.yscale('log')
	# xlabels = ["%s/%s" % (rows[i], nacts[i]) for i in range(num_steps)]
	# plt.xticks(rows[0:num_steps], xlabels)
	plt.grid()
	plt.legend(loc='upper right')
	plt.show()


def plot_size_vs_error(fimp=0.0):
	# size is number of bits
	# fimp is fraction of item memory present
	# plot for both bundle and sdm
	edb = Empirical_error_db()
	names = edb.get_memory_names()
	item_memory_len = 110
	for name in names:
		mi = edb.get_minfo(name)
		mtype = mi["mtype"]
		bits_per_counter = mi["bits_per_counter"]
		match_method = mi["match_method"]
		ndims = len(mi["dims"])
		sizes = np.empty(ndims, dtype=np.float64)
		# empirical_error = np.empty(ndims, dtype=np.float64)
		# empirical_clm = np.empty(ndims, dtype=np.float64)
		predicted_error = np.empty(ndims, dtype=np.float64)
		for i in range(ndims):
			dim = mi["dims"][i]
			if mtype == "sdm":
				dim_id, ie, nrows, ncols, nact, pe, epochs, mean, std = dim
				size = (nrows * ncols * bits_per_counter) + fimp*(nrows*ncols + item_memory_len*ncols) # address memory plus item memory
			else:
				dim_id, ie, ncols, pe, epochs, mean, std = dim
				bits_per_counter = 1 if match_method == "hamming" else 8  # match_method indicates of bundle binarized or not
				size = ncols * bits_per_counter + fimp*(ncols * item_memory_len)  # bundle + item memory
			sizes[i] = size
			predicted_error[i] = -math.log10(pe)
		# plot arrays filled by above
		plt.errorbar(predicted_error, sizes, yerr=None, fmt="-o", label=name)

	# following does not work
	# change order of x-axis so error rate goes from 1 (10**(-1)) to 9 (10**(-9))
	# from: https://stackoverflow.com/questions/23330484/descend-x-axis-values-using-matplotlib
	# grab a reference to the current axes
	# ax = plt.gca()
	# # set the xlimits to be the reverse of the current xlimits
	# ax.set_xlim(ax.get_xlim()[::-1])
	# # call `draw` to re-render the graph
	# plt.draw()

	plt.title("Size (bits) vs error with fimp=%s" % fimp)
	xlabel = "Error rate (10^-n)"
	plt.xlabel(xlabel)
	plt.ylabel("Size in bits")
	# plt.xscale('log')
	# xlabels = ["%s/%s" % (rows[i], nacts[i]) for i in range(num_steps)]
	# plt.xticks(rows[0:num_steps], xlabels)
	plt.grid()
	plt.legend(loc='upper left')
	plt.show()

def main():
	# plot_error_vs_dimension("bundle")
	# plot_error_vs_dimension("sdm")
	plot_size_vs_error(fimp=1.0/64.0)


if __name__ == "__main__":
	# compare_sdm_ham_dot()
	main()

