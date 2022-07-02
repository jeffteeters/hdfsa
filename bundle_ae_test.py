
# compare bundle analytical and empirical error

import numpy as np
import matplotlib.pyplot as plt

import bundle_analytical
import fast_bundle_empirical
import math


dims = {
	"bun_k1000_d100_c8":
		# gallant bundle (8 bit counters), computed using curve_fit.py
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
		[[1, 24002, 5],
		[2, 40503, 5],
		[3, 55649, 20],
		[4, 70239, 100],
		[5, 84572, 0],
		[6, 98790, 0],
		[7, 112965, 0],
		# [8, 127134, 0],
		# [9, 141311, 0],
		],
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