# make plot of SDM error vs activaction count for different number of rows

# for the different accuracy

import fast_sdm_empirical
import binarized_sdm_analytical
# import sdm_ae as sdm_anl
# import sdm_analytical_jaeckel as sdm_jaeckel
import numpy as np
import matplotlib.pyplot as plt
import math

actions=10; states=100; choices=10; d=100; k=1000; ncols=512
nacts = [1] # [3,5,7]
bits_per_counter=1
rows_to_test = list(range(40, 200, 20))

emp_mean=np.empty((len(nacts), len(rows_to_test)), dtype=np.float64)
emp_clm=np.empty((len(nacts), len(rows_to_test)), dtype=np.float64)
bsm_err=np.empty((len(nacts), len(rows_to_test)), dtype=np.float64)

# jaeckel=np.empty(len(nrows))
# numerical=np.empty(len(nrows))

for i in range(len(rows_to_test)):
	nrows = rows_to_test[i]
	for j in range(len(nacts)):
		nact = nacts[j]
		epochs=100
		print("nrows=%s, nact=%s, epochs=%s, " % (nrows, nact, epochs), end='')
		fast_emp = fast_sdm_empirical.Fast_sdm_empirical(nrows, ncols, nact, actions=actions,
			hl_selection_method="random",
			states=states, choices=choices, epochs=epochs, bits_per_counter=bits_per_counter) #100000)
		emp_mean[j,i] = fast_emp.mean_error
		# compute 95% confidence interval as described in:
		# https://blogs.sas.com/content/iml/2019/10/09/statistic-error-bars-mean.html
		ntrials = epochs * k
		emp_clm[j,i] = (fast_emp.std_error / math.sqrt(ntrials)) * 1.96  # 95% confidence interval for the mean (CLM)
		bsm = binarized_sdm_analytical.Binarized_sdm_analytical(nrows, ncols, nact, k, d)
		bsm_err[j,i] = bsm.p_err 
		print("bsm_err=%s, emp_err=%s, std=%s" % (bsm.p_err, fast_emp.mean_error, fast_emp.std_error))

# plot info
# make plots
plots_info = [
	{"subplot": 121, "scale":"linear"},
	{"subplot": 122, "scale":"log"},
]
xvals = rows_to_test
for pi in plots_info:
	plt.subplot(pi["subplot"])
	for j in range(len(nacts)):
		plt.errorbar(rows_to_test, emp_mean[j], yerr=emp_clm[j], fmt="-o", label="empirical nact=%s" % nacts[j])
		plt.errorbar(rows_to_test, bsm_err[j], yerr=None, fmt="-o", label="bsm predicted nact=%s" % nacts[j])
	title = "SDM error rate vs nrows, bits_per_counter=%s" % bits_per_counter
	if pi["scale"] == "log":
		plt.yscale('log')
		title += " (log scale)"
	# plt.xticks(xvals, xticks)
	plt.title(title)
	plt.xlabel("SDM nrows")
	plt.ylabel("fraction error")
	plt.grid()
	plt.legend(loc="upper right")
plt.show()