
# compare bundle analytical and empirical error

import numpy as np
import matplotlib.pyplot as plt

import bundle_analytical
import fast_bundle_empirical
import math


dims = {
	"bun_k1000_d100_c8":
		# gallant bundle (8 bit counters), computed using curve_fit.py
		# must divide by 8 to calculate width of vector
		[[1, 45129, ],
		[2, 54000, 200],
		[3, 62919, 300],
		[4, 71872, 400],
		[5, 80854, 0],
		[6, 89857, 0],
		[7, 98879, 0],
		# [8, 107916],
		# [9, 116966]
		],

	"bun_k1000_d100_c1":
		# output for bundle sizes (from find_sizes.py)
		# Bundle sizes, k=1000, d=100:
		[[1, 24002, 50],
		[2, 40503, 50],
		[3, 55649, 50],
		[4, 70239, 400],
		[5, 84572, 4000],
		[6, 98790, 0],
		[7, 112965, 0],
		# [8, 127134, 0],
		# [9, 141311, 0],
		],

# verified above ("bun_k1000_d100_c1") empirically, output is:
# (base) Jeffs-MacBook:hdfsa jt$ time python bundle_ae_test.py 
# perr=1, ncols=24002, epochs=50, empirical mean=0.09872, stdev=0.008165880234242969, predicted=0.09999738539779227 
# perr=2, ncols=40503, epochs=50, empirical mean=0.01088, stdev=0.0030635926622186575, predicted=0.009999615412897213 
# perr=3, ncols=55649, epochs=50, empirical mean=0.00082, stdev=0.001033247308247159, predicted=0.0010000267689206422 
# perr=4, ncols=70239, epochs=400, empirical mean=0.00013, stdev=0.00035085609585697663, predicted=9.999498413910984e-05 
# perr=5, ncols=84572, epochs=4000, empirical mean=8.750000000000001e-06, stdev=9.577806377245264e-05, predicted=9.99944186034052e-06 
# perr=6, ncols=98790, epochs=0, empirical mean=nan, stdev=None, predicted=1.0000040731419814e-06 
# perr=7, ncols=112965, epochs=0, empirical mean=nan, stdev=None, predicted=1.0000718433472203e-07 
# objc[42113]: Class FIFinderSyncExtensionHost is implemented in both /System/Library/PrivateFrameworks/FinderKit.framework/Versions/A/FinderKit (0x7fff8d08e3f0) and /System/Library/PrivateFrameworks/FileProvider.framework/OverrideBundles/FinderSyncCollaborationFileProviderOverride.bundle/Contents/MacOS/FinderSyncCollaborationFileProviderOverride (0x12ebb7f50). One of the two will be used. Which one is undefined.

# real	1277m20.293s
# user	1094m51.038s
# sys	113m40.107s
	}

def bun_ae():
	k = 1000; d=100
	wanted_dims = "bun_k1000_d100_c1"
	binarize_counters = wanted_dims.endswith("_c1")
	widths = []
	eerr = []  # empirical error
	econ = []  # empirical confidence interval
	pred = []  # analytical (predicted) error
	for dim in dims[wanted_dims]:
		perr, ncols, epochs = dim
		widths.append(ncols)
		if epochs > 0:
			fbe = fast_bundle_empirical.Fast_bundle_empirical(ncols, epochs=epochs,
				count_multiple_matches_as_error=True, roll_address=True,
				binarize_counters=binarize_counters)
			empirical_error = fbe.mean_error
			empirical_std = fbe.std_error
			empirical_con = empirical_std / math.sqrt(epochs) * 1.96 # 95% confidence interval for the mean (CLM)
		else:
			empirical_error = math.nan
			empirical_con = math.nan
			empirical_std = None
			empirical_con = math.nan
		eerr.append(empirical_error )
		econ.append(empirical_con)
		pred_err = bundle_analytical.BundleErrorAnalytical(ncols,d,k)  # predicted error
		pred.append(pred_err)
		print("perr=%s, ncols=%s, epochs=%s, empirical mean=%s, stdev=%s, predicted=%s " % (perr,
			ncols, epochs, empirical_error, empirical_std, pred_err))


	# make plots
	plots_info = [
		{"subplot": 121, "scale":"linear"},
		{"subplot": 122, "scale":"log"},
	]
	for pi in plots_info:
		plt.subplot(pi["subplot"])
		plt.errorbar(widths, eerr, yerr=econ, fmt="-o", label="empirical")
		plt.plot(widths, pred, "-o", label="analytical")
		# plt.title("Bundle 8-bit (Gallant) empirical vs analytical")
		title = "%s empirical vs analytical" % wanted_dims
		plt.xlabel("width of bundle")
		plt.ylabel("Fraction error")
		if pi["scale"] == "log":
			plt.yscale('log')
			yticks = (10.0**-(np.arange(perr, 0, -1.0)))
			ylabels = [10.0**(-i) for i in range(perr, 0, -1)]
			plt.yticks(yticks, ylabels)
			title += " (log scale)"
		plt.title(title)
		plt.grid()
		plt.legend(loc='upper right')
	plt.show()


def main():
	bun_ae()


main()