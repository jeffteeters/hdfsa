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

def plot_operations_vs_error(parallel=False):
	# operations is number if byte operations, or parallel byte operations (if parallel is True)
	# plot for both bundle and sdm
	edb = Empirical_error_db()
	names = edb.get_memory_names()
	item_memory_len = 100
	for name in names:
		mi = edb.get_minfo(name)
		mtype = mi["mtype"]
		bits_per_counter = mi["bits_per_counter"]
		match_method = mi["match_method"]
		ndims = len(mi["dims"])
		operations = np.empty(ndims, dtype=np.float64)
		# empirical_error = np.empty(ndims, dtype=np.float64)
		# empirical_clm = np.empty(ndims, dtype=np.float64)
		predicted_error = np.empty(ndims, dtype=np.float64)
		for i in range(ndims):
			dim = mi["dims"][i]
			if mtype == "sdm":
				dim_id, ie, nrows, ncols, nact, pe, epochs, mean, std = dim
				if match_method == "hamming":
					# threshold, use hamming
					ops_address_compare = ncols * nrows / 8
					ops_select_active = nrows
					ops_add_counters = ncols * nact
					ops_threshold = ncols
					ops_item_memory_compare = item_memory_len * ncols / 8
					ops_sum_to_make_hamming = item_memory_len * ncols / 8
					ops_select_smallest = item_memory_len
					ops = (ops_address_compare + ops_select_active + ops_add_counters + ops_threshold + ops_item_memory_compare
						+ ops_sum_to_make_hamming + ops_select_smallest)
				else:
					# don't threshold, match using dot product
					ops_address_compare = ncols * nrows / 8
					ops_select_active = nrows
					ops_add_counters = ncols * nact
					ops_threshold = 0  # was ncols for thresholding
					ops_item_memory_compare = item_memory_len * ncols  # don't divide by 8 because of dot product
					ops_sum_to_make_hamming = item_memory_len * ncols  # "                "
					ops_select_smallest = item_memory_len
					ops = (ops_address_compare + ops_select_active + ops_add_counters + ops_threshold + ops_item_memory_compare
						+ ops_sum_to_make_hamming + ops_select_smallest)
				# size = (nrows * ncols * bits_per_counter) + fimp*(nrows*ncols + item_memory_len*ncols) # address memory plus item memory
			else:
				dim_id, ie, ncols, pe, epochs, mean, std = dim
				# bits_per_counter = 1 if match_method == "hamming" else 8  # match_method indicates of bundle binarized or not
				ops = ncols * (item_memory_len + 2)/8 + item_memory_len if not parallel else ncols / 8 + item_memory_len
				# size = ncols * bits_per_counter + fimp*(ncols * item_memory_len)  # bundle + item memory
			operations[i] = ops
			predicted_error[i] = -math.log10(pe)
		# plot arrays filled by above
		plt.errorbar(predicted_error, operations, yerr=None, fmt="-o", label=name)
	plt.title("Byte operations vs error with parallel=%s" % parallel)
	xlabel = "Error rate (10^-n)"
	plt.xlabel(xlabel)
	plt.ylabel("Number byte operations")
	# plt.xscale('log')
	# xlabels = ["%s/%s" % (rows[i], nacts[i]) for i in range(num_steps)]
	# plt.xticks(rows[0:num_steps], xlabels)
	plt.grid()
	plt.legend(loc='upper left')
	plt.show()

def main():
	# plot_error_vs_dimension("bundle")
	# plot_error_vs_dimension("sdm")
	# plot_size_vs_error(fimp=1.0/64.0)
	plot_operations_vs_error()


if __name__ == "__main__":
	# compare_sdm_ham_dot()
	main()

