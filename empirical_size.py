# estimate sdm size empirically

import fast_sdm_empirical
import numpy as np
import matplotlib.pyplot as plt
import math

def optimum_nact(k, m):
	# return optimum nact given number items stored (k) and number of rows (m)
	# from formula given in Pentti paper
	fraction_rows_activated =  1.0 / ((2*m*k)**(1/3))
	nact = round(m * fraction_rows_activated)
	return nact


def main():
	ncols = 512; actions=10; choices=10; states=100
	k = 1000
	d = 100
	epochs=50
	threshold_sum = False
	wanted_rows = range(40, 121, 20)
	perr = np.empty(len(wanted_rows), dtype=np.float64)
	nacts = np.empty(len(wanted_rows), dtype=np.int16)
	clm = np.empty(len(wanted_rows), dtype=np.float64)
	for i in range(len(wanted_rows)):
		nrows = wanted_rows[i]
		epochs = 50 if nrows < 100 else 200
		nact = optimum_nact(k, nrows)
		nacts[i] = nact
		fse = fast_sdm_empirical.Fast_sdm_empirical(nrows, ncols, nact, actions=actions, states=states, choices=choices,
			threshold_sum=threshold_sum, epochs=epochs)
		perr[i] = fse.mean_error
		clm[i] = (fse.std_error / math.sqrt(epochs)) * 1.96  # 95% confidence interval for the mean (CLM)

		print("epochs=%s, sdm size=%s/%s, perr=%s, std=%s" %(epochs, nrows, nact, fse.mean_error, fse.std_error))

	# make plot
	plt.errorbar(wanted_rows, perr, yerr=clm, fmt="-o", label="non-thresholded")
	plt.title("Error vs sdm size for non-thresholded sdm with 8-bit counters")
	plt.xlabel("sdm num rows")
	plt.ylabel("Fraction error")
	plt.yscale('log')
	xlabels = ["%s/%s" % (wanted_rows[i], nacts[i]) for i in range(len(wanted_rows))]
	plt.xticks(wanted_rows, xlabels)
	plt.grid()
	plt.legend(loc='upper right')
	plt.show()


if __name__ == "__main__":
	# compare_sdm_ham_dot()
	main()

