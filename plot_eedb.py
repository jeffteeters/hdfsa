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
import sdm_analytical_jaeckel as sdm_jaeckel
import warnings

pfmt = {
	"S1": { # hamming match, solid line thin
			"predict": {"fmt": "o-k", "dashes": None, "lw": 2, "linestyle":"solid", "label":"Prediction", "alpha":1},
			 "epmf":{"fmt": "+r", "dashes":(0, (2,2)), "lw":6, "label":"empirical pmf", "alpha":0.6},
			 "ecount":{"fmt":"*y", "dashes":(2,(2,2)), "lw":6, "label":"empirical count", "alpha":0.6},
			 "annotate":{"xy":(5.1e4, 2.3e-3), "xytext":(5.6e4, 8.0e-3), "text":"S1"}
			 },
	"S2": { # dot match, solid line thin
			"predict": {"fmt": "o-k", "dashes": None, "lw": 2, "linestyle":"solid", "label":None, "alpha":1},
			 "epmf":{"fmt":":+m", "dashes":(0, (1,1)), "lw":6, "label":None, "alpha":0.6},
			 "ecount":{"fmt":"--*y", "dashes":(1,(1,1)), "lw":6, "label":None, "alpha":0.6},
			 "annotate":{"xy":(3.9e4, 3.0e-4), "xytext":(2.95e4, 5.5e-5), "text":"S2"}  # "arrow_start": (3.5e4, 1.5e-5)
			 },
	"A1": { # full counter, threshold sum
			"predict": {"fmt": ".-k", "dashes": None, "lw": 1, "linestyle":"solid", "label":"Prediction"
				, "alpha":1}, # [12,6,12,6,3,6]
			 "epmf":{"fmt":":+m", "dashes":None, "lw":None, "label":"empirical pmf", "alpha":1},
			 "ecount":{"fmt":"--*k", "dashes":None, "lw":None, "label":"empirical count", "alpha":1},
			 "annotate":{"xy":(156, 2.4e-4), "xytext":(191, 1.9e-3), "text":"A1 & A3"}
			 },
	"A2": { # Binary counter, threshold sum
			"predict": {"fmt": ".-k", "dashes": None, "lw": 1, "linestyle":"solid", "label":None, "alpha":1},  # [12,6,3,6,3,6]
			 "epmf":{"fmt":":+m", "dashes":None, "lw":None, "label":None , "alpha":1},
			 "ecount":{"fmt":"--*k", "dashes":None, "lw":None, "label":None, "alpha":1},
			 "annotate":{"xy":(249, 2.8e-5), "xytext":(264, 1.0e-4), "text":"A2"}
			 },
	"A3": { # binary counter, non-thresholded sum
			"predict": {"fmt": None, "dashes": None, "lw": 1, "linestyle":"solid", "label":None, "alpha":1},  # [12,6,12,6,3,6]
			 "epmf":{"fmt":":+g", "dashes":None, "lw":None, "label": "A3 empirical pmf" , "alpha":1},
			 "ecount":{"fmt":"--*g", "dashes":None, "lw":None, "label": "A3 empirical count", "alpha":1},
			 },
	"A4": { # full counter, non-thresholded sum
			"predict": {"fmt": ".-k", "dashes": None, "lw": 1, "linestyle":"solid", "label":None, "alpha":1},  # [12,6,12,6,3,6]
			 "epmf":{"fmt":":+m", "dashes":None, "lw":None, "label":None, "alpha":1},
			 "ecount":{"fmt":"--*k", "dashes":None, "lw":None,"label":None, "alpha":1},
			 "annotate":{"xy":(95, 1.0e-4), "xytext":(62, 2.5e-5), "text":"A4"}
			 },

}

		# if theoretical_bundle_err and mtype == "bind":
		# 	plt.errorbar(xvals["bind"], theoretical_bundle_err, label="Superposition theory", fmt="-",
		# 		linestyle=':', linewidth=6, alpha=0.7)
		# if theoretical_sdm_err and mtype == "sdm":
		# 	plt.errorbar(xvals["bind"], theoretical_sdm_err, label="SDM theory", fmt="-o", linestyle='dashed')

def plot_error_vs_dimension(mtype="sdm", include_jaeckel=False):
	# dimenion is ncols (width) of bundle or nrows
	global pfmt
	assert mtype in ("sdm", "bundle")
	edb = Empirical_error_db()
	names = edb.get_memory_names(mtype=mtype)
	fig, ax = plt.subplots()
	for name in names:
		mi = edb.get_minfo(name)
		bits_per_counter = mi["bits_per_counter"]
		match_method = mi["match_method"]
		ndims = len(mi["dims"])
		sizes = np.empty(ndims, dtype=np.uint32)
		empirical_error = np.empty(ndims, dtype=np.float64)
		empirical_clm = np.empty(ndims, dtype=np.float64)
		predicted_error = np.empty(ndims, dtype=np.float64)
		pmf_error = np.empty(ndims, dtype=np.float64)
		jaeckel_error = np.empty(ndims, dtype=np.float64) if (name == "sdm_k1000_d100_c8_ham#A1" and
				include_jaeckel) else None
		for i in range(ndims):
			dim = mi["dims"][i]
			if mtype == "sdm":
				# dim_id, ie, size, ncols, nact, pe, epochs, mean, std = dim
				dim_id, ie, size, ncols, nact, pe, epochs, mean, std, match_counts, distract_counts, pmf_err = dim
				if jaeckel_error is not None:
					k = 1000; d=100
					jaeckel_error[i] = sdm_jaeckel.SdmErrorAnalytical(size,k,d,nact=nact,word_length=512)
			else:
				# dim_id, ie, size, pe, epochs, mean, std = dim
				dim_id, ie, size, pe, epochs, mean, std, match_counts, distract_counts, pmf_err = dim
			sizes[i] = size
			predicted_error[i] = pe
			if False: # mi["short_name"] in ("S1", "S2") and ie == 6:
				print("%s, ie=%s - len(match_counts)=%s, len(distract_counts)=%s" % (name, ie, len(match_counts), len(distract_counts)))
				print("match_counts=%s..." % match_counts[0:80])
				print("distract_counts=%s..." % distract_counts[0:80])
				import pdb; pdb.set_trace()
			pmf_error[i] = pmf_err if pmf_err is not None else np.nan
			if mean is not None and mean > 0:
				empirical_error[i] = mean
				empirical_clm[i] = std / math.sqrt(epochs) * 1.96  # 95% confidence interval for the mean (CLM)
			else:
				empirical_error[i] = np.nan
				empirical_clm[i] = np.nan
		# plot arrays filled by above
		# print("plotting")
		# print("sizes=%s" % sizes)
		# print("empirical_error=%s" % empirical_error)
		# print("empirical_clm=%s" % empirical_clm)
		# plt.errorbar(sizes, empirical_error, yerr=empirical_clm, fmt="-o", label=name)
		short_name = mi["short_name"]
		pf = pfmt[short_name]
		if pf["predict"]["fmt"] is not None:
			ax.errorbar(sizes, predicted_error, yerr=None, label=pf["predict"]["label"],
				fmt=pf["predict"]["fmt"], dashes=pf["predict"]["dashes"], lw=pf["predict"]["lw"],
				linestyle=pf["predict"]["linestyle"], alpha=pf["predict"]["alpha"])
		ax.errorbar(sizes, empirical_error, yerr=None, #empirical_clm,
				label=pf["ecount"]["label"],
				fmt=pf["ecount"]["fmt"], linestyle=pf["ecount"]["dashes"], lw=pf["ecount"]["lw"],
				alpha=pf["ecount"]["alpha"])
		ax.errorbar(sizes, pmf_error, yerr=None, label=pf["epmf"]["label"],
				fmt=pf["epmf"]["fmt"], linestyle=pf["epmf"]["dashes"], lw=pf["epmf"]["lw"],
				alpha=pf["epmf"]["alpha"])
		if "annotate" in pf:
			if "arrow_start" in pf["annotate"]:
				# draw arrow and text separately to allow better control of arrow start
				# draw text
				ax.annotate(pf["annotate"]["text"], xy=pf["annotate"]["xy"],  xycoords='data',
    		        xytext=pf["annotate"]["xytext"], textcoords='data',
        		    arrowprops=None, # dict(facecolor='black', shrink=0.05, width=.5, headwidth=7,),
        	    	fontsize='large', fontweight='bold',
            		# horizontalalignment='right', verticalalignment='top',
            		)
				# draw arrow
				ax.annotate("", xy=pf["annotate"]["xy"],  xycoords='data',
    		        xytext=pf["annotate"]["xytext"], textcoords='data',
        		    arrowprops=dict(facecolor='black', shrink=0.05, width=.5, headwidth=7,),
        		    # arrowstyle="->"
        	    	# fontsize='large', fontweight='bold',
            		# horizontalalignment='right', verticalalignment='top',
            		)
			else:
				# draw arrow and text with one call
				ax.annotate(pf["annotate"]["text"], xy=pf["annotate"]["xy"],  xycoords='data',
    		        xytext=pf["annotate"]["xytext"], textcoords='data',
        		    arrowprops=dict(facecolor='black', shrink=0.05, width=.5, headwidth=7,),
        		    # arrowstyle="->"
        	    	fontsize='large', fontweight='bold',
            		# horizontalalignment='right', verticalalignment='top',
            		)
		if jaeckel_error is not None:
			plt.errorbar(sizes, jaeckel_error, yerr=None, fmt="8m", label="jaeckel_error")
	plt.title("%s empirical vs. predicted error rw1_noroll_v3" % mtype)
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
				# dim_id, ie, nrows, ncols, nact, pe, epochs, mean, std = dim
				dim_id, ie, nrows, ncols, nact, pe, epochs, mean, std, match_counts, distract_counts, pmf_err = dim
				size = (nrows * ncols * bits_per_counter) + fimp*(nrows*ncols + item_memory_len*ncols) # address memory plus item memory
			else:
				# dim_id, ie, ncols, pe, epochs, mean, std = dim
				dim_id, ie, ncols, pe, epochs, mean, std, match_counts, distract_counts, pmf_err = dim
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


def plot_memory_size_and_efficiency_vs_fimp(ie, plot_type="size"):
	# ie is a fixed error rate, range(1,10); error rate is 10**(-ie)
	# plot_type = "size" or "memory_efficiency"
	# plot for both bundle and sdm
	assert plot_type in ("size", "memory_efficiency")
	assert ie in range(1,10)
	if plot_type == "size":
		title = "Size required vs fimp for error rate=10**(-%s)" % ie
		ylabel = "Size (bits)"
	else:
		title = "Memory efficiency vs fimp for error rate=10**(-%s)" % ie
		ylabel = "Memory efficiency (%)"
	edb = Empirical_error_db()
	names = edb.get_memory_names()
	item_memory_len = 110
	max_bits_used_in_counters = 8
	for name in names:
		mi = edb.get_minfo(name)
		mtype = mi["mtype"]
		bits_per_counter = mi["bits_per_counter"]
		match_method = mi["match_method"]
		# determine bits_used_in_counters, for sdm differently than bundle (for bundle, bits_per_counter is always 8)
		if mtype == "sdm":
			bits_used_in_counters = min(max_bits_used_in_counters, bits_per_counter)
		else:
			bits_used_in_counters = 1 if match_method == "hamming" else max_bits_used_in_counters
		num_fimps = 50
		fimps = np.linspace(0, 1.0, num=num_fimps, endpoint=True)
		yvals = np.empty(num_fimps, dtype=np.float64)  # stores either size or efficiency
		dim = mi["dims"][ie-1]
		assert dim[1] == ie  # make sure retrieved specified dimension
		# empirical_error = np.empty(ndims, dtype=np.float64)
		for i in range(num_fimps):
			fimp = fimps[i]
			if mtype == "sdm":
				# dim_id, ie, nrows, ncols, nact, pe, epochs, mean, std = dim
				dim_id, ie, nrows, ncols, nact, pe, epochs, mean, std, match_counts, distract_counts, pmf_err = dim
				# size = (nrows * ncols * bits_per_counter) + fimp*(nrows*ncols + item_memory_len*ncols) # address memory plus item memory
				size = (nrows * ncols * bits_used_in_counters) + fimp*(nrows*ncols + item_memory_len*ncols) # address memory plus item memory
				changing_bits_size = (nrows * ncols * bits_used_in_counters)
			else:
				# dim_id, ie, ncols, pe, epochs, mean, std = dim
				dim_id, ie, ncols, pe, epochs, mean, std, match_counts, distract_counts, pmf_err = dim
				# bits_per_counter = 1 if match_method == "hamming" else 8  # match_method indicates of bundle binarized or not
				# size = ncols * bits_per_counter + fimp*(ncols * item_memory_len)  # bundle + item memory
				size = ncols * bits_used_in_counters + fimp*(ncols * item_memory_len)  # bundle + item memory
				changing_bits_size = (ncols * bits_used_in_counters)
			yvals[i] = size if plot_type == "size" else (changing_bits_size *100.0 / size)
		# plot arrays filled by above
		plt.errorbar(fimps, yvals, yerr=None, fmt="-", label=name)

	plt.title(title)
	plt.xlabel("Fraction item memory present (fimp)")
	plt.ylabel(ylabel)
	# plt.xscale('log')
	# xlabels = ["%s/%s" % (rows[i], nacts[i]) for i in range(num_steps)]
	# plt.xticks(rows[0:num_steps], xlabels)
	plt.grid()
	plt.legend(loc='upper right')
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
				# dim_id, ie, nrows, ncols, nact, pe, epochs, mean, std = dim
				dim_id, ie, nrows, ncols, nact, pe, epochs, mean, std, match_counts, distract_counts, pmf_err = dim
				if match_method == "hamming":
					# threshold, use hamming
					ops_address_compare = ncols * nrows / 8 if not parallel else ncols / 8
					ops_select_active = nrows
					ops_add_counters = ncols * nact if not parallel else ncols
					ops_threshold = ncols
					ops_item_memory_compare = item_memory_len * ncols / 8 if not parallel else ncols / 8
					ops_sum_to_make_hamming = item_memory_len * ncols / 8 if not parallel else ncols / 8
					ops_select_smallest = item_memory_len
					ops = (ops_address_compare + ops_select_active + ops_add_counters + ops_threshold + ops_item_memory_compare
						+ ops_sum_to_make_hamming + ops_select_smallest)
				else:
					# don't threshold, match using dot product
					ops_address_compare = ncols * nrows / 8 if not parallel else ncols / 8
					ops_select_active = nrows
					ops_add_counters = ncols * nact if not parallel else ncols
					ops_threshold = 0  # was ncols for thresholding
					ops_item_memory_compare = item_memory_len * ncols if not parallel else ncols  # don't divide by 8 because of dot product
					ops_sum_to_make_hamming = item_memory_len * ncols if not parallel else ncols  # "                "
					ops_select_smallest = item_memory_len
					ops = (ops_address_compare + ops_select_active + ops_add_counters + ops_threshold + ops_item_memory_compare
						+ ops_sum_to_make_hamming + ops_select_smallest)
				# size = (nrows * ncols * bits_per_counter) + fimp*(nrows*ncols + item_memory_len*ncols) # address memory plus item memory
			else:
				# dim_id, ie, ncols, pe, epochs, mean, std = dim
				dim_id, ie, ncols, pe, epochs, mean, std, match_counts, distract_counts, pmf_err = dim
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
	plot_error_vs_dimension("bundle")
	plot_error_vs_dimension("sdm")
	# plot_size_vs_error(fimp=0) # 1.0/64.0)
	# plot_size_vs_error(fimp=1) # 1.0/64.0)
	# plot_operations_vs_error(parallel=True)
	# ie = 6
	# plot_memory_size_and_efficiency_vs_fimp(ie, plot_type="size")
	# plot_memory_size_and_efficiency_vs_fimp(ie, plot_type="memory_efficiency")


if __name__ == "__main__":
	# compare_sdm_ham_dot()
	warnings.simplefilter('error', UserWarning)
	main()

