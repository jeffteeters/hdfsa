# make plot of SDM error vs number of rows for sizes that are predicted
# for the different accuracy

import fast_sdm_empirical
import sdm_ae as sdm_anl
import sdm_analytical_jaeckel as sdm_jaeckel
import numpy as np
import matplotlib.pyplot as plt
import math


sdm_dims = [
#  perror, nrows, nact, epochs;  perror is predicted error in 10e-n
  [1, 51, 1, 200],
  [2, 86, 2, 500],
  [3, 125, 2, 1000],
  [4, 168, 2, 5000],
  [5, 196, 3, 10000],
  [6, 239, 3, 30000],
  [7, 285, 3, 60000],
#  [8, 312, 4],
#  [9, 349, 4]
]

actions=10; states=100; choices=10; d=100; k=1000; ncols=512
xticks=[]; xvals=[]; emp_mean=[]; emp_std=[]; jaeckel=[]; numerical=[]

# tol = 1e-9
for i in range(len(sdm_dims)):
	print("starting %s" % i)
	xvals.append(i)
	perr, nrows, nact, epochs = sdm_dims[i]
	xticks.append("%s/%s" % (nrows, nact))
	if epochs > 0:
		# epochs = 100 if perr < 3 else 1000  # int(max((10**perr)*100/1000, 100))
		print("perr=%s, calling fast_emp with epochs=%s" % (perr, epochs))
		fast_emp = fast_sdm_empirical.Fast_sdm_empirical(nrows, ncols, nact, actions=actions,
			states=states, choices=choices, epochs=epochs) #100000)
		emp_mean.append(fast_emp.mean_error)
		mean_error = fast_emp.mean_error
		std_error = fast_emp.std_error
		# compute 95% confidence interval as described in:
		# https://blogs.sas.com/content/iml/2019/10/09/statistic-error-bars-mean.html
		ntrials = epochs * k
		clm = (fast_emp.std_error / math.sqrt(ntrials)) * 1.96  # 95% confidence interval for the mean (CLM)
		emp_std.append(clm)
		# mean_error = fast_emp.mean_error
		# std_error = fast_emp.std_error / 2
		# # from: https://stackoverflow.com/questions/56433933/why-error-bars-in-log-scale-matplotlib-bar-plot-are-lopsided
		# std = std_error if mean_error > std_error else mean_error - tol
		# emp_std.append(fast_emp.std_error/2)
		print("Done calling fast empirical")
	else:
		emp_mean.append(math.nan)
		emp_std.append(math.nan)
	jaeckel.append(sdm_jaeckel.SdmErrorAnalytical(nrows,k,d,nact,word_length=ncols))
	anl = sdm_anl.Sdm_error_analytical(nrows, nact, k, ncols, d)
	numerical.append(anl.perr)

# make plots
plots_info = [
	{"subplot": 121, "scale":"linear"},
	{"subplot": 122, "scale":"log"},
]
for pi in plots_info:
	plt.subplot(pi["subplot"])
	plt.errorbar(xvals, emp_mean, yerr=emp_std, fmt="-o", label="Empirical")
	plt.errorbar(xvals, numerical, yerr=None, fmt="-o", label="Predicted numerically")
	plt.errorbar(xvals, jaeckel, yerr=None, fmt="-o", label="Predicted Jackel")
	title = "Found and predicted error rate vs SDM dimensions"
	if pi["scale"] == "log":
		plt.yscale('log')
		title += " (log scale)"
	plt.xticks(xvals, xticks)
	plt.title(title)
	plt.xlabel("SDM nrows/nact")
	plt.ylabel("fraction error")
	plt.grid()
	plt.legend(loc="upper right")
plt.show()


# output from sdm_anl.Sdm_error_analytical:
		# SDM sizes, nact:
		# 1 - 51, 1
		# 2 - 86, 2
		# 3 - 125, 2
		# 4 - 168, 2
		# 5 - 196, 3
		# 6 - 239, 3
		# 7 - 285, 3
		# 8 - 312, 4
		# 9 - 349, 4

# output of run overnight, 5/31/2022

#  time python plot_sdm_err_vs_rows.py 
# starting 0
# perr=1, calling fast_emp with epochs=200
# starting Fast_sdm_empirical, nrows=51, ncols=512, nact=1, actions=10, states=100, choices=10
# fast_sdm_empirical epochs=200, perr=0.099345, std_error=0.009092633007000776
# Done calling fast empirical
# starting 1
# perr=2, calling fast_emp with epochs=500
# starting Fast_sdm_empirical, nrows=86, ncols=512, nact=2, actions=10, states=100, choices=10
# fast_sdm_empirical epochs=500, perr=0.010352, std_error=0.003239767892920726
# Done calling fast empirical
# starting 2
# perr=3, calling fast_emp with epochs=1000
# starting Fast_sdm_empirical, nrows=125, ncols=512, nact=2, actions=10, states=100, choices=10
# fast_sdm_empirical epochs=1000, perr=0.001003, std_error=0.0009710772368869532
# Done calling fast empirical
# starting 3
# perr=4, calling fast_emp with epochs=5000
# starting Fast_sdm_empirical, nrows=168, ncols=512, nact=2, actions=10, states=100, choices=10
# fast_sdm_empirical epochs=5000, perr=0.0001018, std_error=0.0003277754719316258
# Done calling fast empirical
# starting 4
# perr=5, calling fast_emp with epochs=10000
# starting Fast_sdm_empirical, nrows=196, ncols=512, nact=3, actions=10, states=100, choices=10
# fast_sdm_empirical epochs=10000, perr=9.3e-06, std_error=9.598703037389997e-05
# Done calling fast empirical
# starting 5
# perr=6, calling fast_emp with epochs=30000
# starting Fast_sdm_empirical, nrows=239, ncols=512, nact=3, actions=10, states=100, choices=10
# fast_sdm_empirical epochs=30000, perr=1.2333333333333333e-06, std_error=3.509718253966011e-05
# Done calling fast empirical
# starting 6
# perr=7, calling fast_emp with epochs=60000
# starting Fast_sdm_empirical, nrows=285, ncols=512, nact=3, actions=10, states=100, choices=10
# fast_sdm_empirical epochs=60000, perr=1e-07, std_error=9.999499987499376e-06
# Done calling fast empirical