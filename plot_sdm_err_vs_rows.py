# make plot of SDM error vs number of rows for sizes that are predicted
# for the different accuracy

# this function does empirical calls to calculate the error before plotting
# so it takes a long time.  It's also not complete (can't do all error rates).
# This has been replaced by plot_eedb.py, which makes plots by reading the
# empirical error from an sqlite3 databae; so the empirical calculations
# can be done separately from the plotting.


import fast_sdm_empirical
import sdm_ae as sdm_anl
import sdm_analytical_jaeckel as sdm_jaeckel
import binarized_sdm_analytical
import numpy as np
import matplotlib.pyplot as plt
import math

# from curve_fit import mdims # can't do this for this routine because need to manually include epochs


cf_dims = {
	# from curve_fit.py
	"bun_k1000_d100_c1#S1":
		[[1,24002],
		[2, 40503],
		[3, 55649],
		[4, 70239],
		[5, 84572],
		[6, 98790],
		[7, 112965],
		[8, 127134],
		[9, 141311]],

	"bun_k1000_d100_c8#S2":
	[[1, 15221],
	[2, 25717],
	[3, 35352],
	[4, 44632],
	[5, 53750],
	[6, 62794],
	[7, 71812],
	[8, 80824],
	[9, 89843]],

	"sdm_k1000_d100_c1_ham#A2": # sdm_binarized_nact_optimum_dims
		[[1, 50, 1,], # 200],  # comment out epochs
		[2, 97, 1,], #  200],
		[3, 158, 3,], #  300],
		[4, 208, 3,], #  400],
		[5, 262, 3,], # 3000],
		[6, 315, 5,], # 10000],
		[7, 368, 5,], # 40000]]
		[8, 424, 7,],
		[9, 476, 7]],
	"sdm_k1000_d100_c8_ham#A1":
		#output from sdm_anl.Sdm_error_analytical:
		# SDM sizes, nact:
		# sdm_8bit_counter_dims = 
		# from file: sdm_c8_ham_thres1m.txt
		[[1, 50, 2, 100], # 51, 1],
		[2, 86, 2, 200],
		[3, 121, 3, 300], # 125, 2],
		[4, 157, 3, 400], # 168, 2],
		[5, 192, 4, 0], # #196, 3],
		[6, 228, 5, 0], #239, 3],
		[7, 265, 5, 0], # 285, 3],
		[8, 303, 5, 0], # 312, 4],
		[9, 340, 6, 0]], # 349, 4]],

	"sdm_k1000_d100_c1_dot#A3":
	# from empirical_size, non-thresholded sum, binarized counters
		# new, same as above
		[[1, 50, 2], # 51, 1],
		[2, 86, 2],
		[3, 121, 3], # 125, 2],
		[4, 157, 3], # 168, 2],
		[5, 192, 4], # #196, 3],
		[6, 228, 5], #239, 3],
		[7, 265, 5], # 285, 3],
		[8, 303, 5], # 312, 4],
		[9, 340, 6]], # 349, 4]],
		# original
		# [[1, 51, 1],
		# [2, 86, 2],
		# [3, 127, 2],
		# [4, 163, 2],
		# [5, 199, 3],
		# [6, 236, 3],
		# [7, 272, 3],
		# [8, 309, 4],
		# [9, 345, 4]],
	"sdm_k1000_d100_c8_dot#A4":
		# updatted; from sdm_c8_dot_threshold_10M
		[[1, 30, 1],  # was 1, 31, 2
		[2, 54, 2],
		[3, 77, 2],
		[4, 100, 3],  # was 102, 2]
		[5, 123, 3],  # 129, 2],  # was 129, 2
		[6, 147, 4], # was 160, 2], # try with nact=4?
		[7, 172, 4], # was 178, 3],
		[8, 198, 5], # was 205, 3],
		[9, 224, 5]], # was 238, 3]],
}



sdm_dims = [
#  perror, nrows, nact, epochs;  perror is predicted error in 10e-n
	[1, 51, 1, 200],
	[2, 86, 2, 200],
	[3, 125, 2, 500],
	[4, 168, 2, 500],
	[5, 196, 3, 1000],
	[6, 239, 3, 0],
	[7, 285, 3, 0],
#  [8, 312, 4],
#  [9, 349, 4]
]
sdm_binarized_nact_1_dims = [
	[1, 50, 1, 200],
	[2, 97, 1, 200],
	[3, 161, 1, 300],
	[4, 254, 1, 400],
	[5, 396, 1, 500],
	[6, 619, 1, 0],
	[7, 984, 1, 0]
]

sdm_binarized_nact_optimum_dims = [
	# [1, 50, 1, 200],
	# [2, 97, 1, 200],
	# [3, 158, 3, 300],
	# [4, 208, 3, 400],
	[5, 262, 3, 3000], # 500
	[6, 315, 5, 10000], # 1000
	[7, 368, 5, 40000] # 4000
]  # 6500 takes 1 hour
   # 55000 should take 8 hours
	#    (base) Jeffs-MacBook:hdfsa jt$ time python plot_sdm_err_vs_rows.py 
	# perr=5, starting nrows=262, nact=3, epochs=3000, 
	# bsm_err=9.903994211261537e-06, emp_err=1.2333333333333334e-05, std=0.00011036857443231846

	# perr=6, starting nrows=315, nact=5, epochs=10000, 
	# bsm_err=1.011641044571517e-06, emp_err=1.0000000000000002e-06, std=3.160696125855823e-05

	# perr=7, starting nrows=368, nact=5, epochs=40000, 
	# bsm_err=1.0171451195259692e-07, emp_err=1.5e-07, std=1.2246530120813818e-05
	# real	1014m15.520s
	# user	520m59.679s
	# sys	6m14.097s




# gallant bundle (8 bit counters), computed using curve_fit.py
bun_k1000_d100_c8 = [
	[1, 45129, 200],
	[2, 54000, 200],
	[3, 62919, 300],
	[4, 71872, 400],
	[5, 80854, 0],
	[6, 89857, 0],
	[7, 98879, 0],
	# [8, 107916],
	# [9, 116966]
	]

actions=10; states=100; choices=10; d=100; k=1000; ncols=512
xticks=[]; xvals=[]; emp_mean=[]; emp_std=[]; jaeckel=[]; numerical=[]

# using_binarized_sdm = True
# if using_binarized_sdm:
# 	sdm_dims = sdm_binarized_nact_optimum_dims # sdm_binarized_nact_1_dims
# else:
sdm_dims = cf_dims["sdm_k1000_d100_c8_ham#A1"]
bits_per_counter = 8
# tol = 1e-9
for i in range(len(sdm_dims)):
	# print("starting %s" % i)
	xvals.append(i)
	perr, nrows, nact, epochs = sdm_dims[i]
	xticks.append("%s/%s" % (nrows, nact))
	if epochs > 0:
		# epochs = 100 if perr < 3 else 1000  # int(max((10**perr)*100/1000, 100))
		print("perr=%s, starting nrows=%s, nact=%s, epochs=%s, " % (perr, nrows, nact, epochs))
		fast_emp = fast_sdm_empirical.Fast_sdm_empirical(nrows, ncols, nact, actions=actions,
			states=states, choices=choices, epochs=epochs, bits_per_counter=bits_per_counter) #100000)
		emp_mean.append(fast_emp.mean_error)
		mean_error = fast_emp.mean_error
		std_error = fast_emp.std_error
		# compute 95% confidence interval as described in:
		# https://blogs.sas.com/content/iml/2019/10/09/statistic-error-bars-mean.html
		ntrials = epochs * k
		clm = (fast_emp.std_error / math.sqrt(ntrials)) * 1.96  # 95% confidence interval for the mean (CLM)
		emp_std.append(clm)
		# print("Done calling fast empirical")
	else:
		emp_mean.append(math.nan)
		emp_std.append(math.nan)
	# jaeckel.append(sdm_jaeckel.SdmErrorAnalytical(nrows,k,d,nact,word_length=ncols))
	# anl = sdm_anl.Sdm_error_analytical(nrows, nact, k, ncols, d)
	# numerical.append(anl.perr)

	# following was in latest version
	# if nact == 1:
	# 	bsm = binarized_sdm_analytical.Binarized_sdm_analytical(nrows, ncols, nact, k, d)
	# else:
	# 	bsm = binarized_sdm_analytical.Bsa_sample(nrows, ncols, nact, k, d)
	# numerical.append(bsm.p_err)
	# print("bsm_err=%s, emp_err=%s, std=%s\n" % (bsm.p_err, fast_emp.mean_error, fast_emp.std_error))
	desired_error = 10.0**(-perr)
	numerical.append(desired_error)


# make plots
plots_info = [
	{"subplot": 121, "scale":"linear"},
	{"subplot": 122, "scale":"log"},
]
for pi in plots_info:
	plt.subplot(pi["subplot"])
	plt.errorbar(xvals, emp_mean, yerr=emp_std, fmt="-o", label="Empirical")
	# plt.errorbar(xvals, numerical, yerr=None, fmt="-o", label="Predicted numerically")
	plt.errorbar(xvals, numerical, yerr=None, fmt="-o", label="Desired error")
	# plt.errorbar(xvals, jaeckel, yerr=None, fmt="-o", label="Predicted Jackel")
	# title = "Found and predicted error rate vs binarized SDM dimensions"
	title = "Found and desired error rate vs SDM dimensions"
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
#
# real	2312m51.654s
# user	380m40.024s
# sys	3m43.139s

# works well with smaller epoch counts:
# (base) Jeffs-MacBook:hdfsa jt$ time python plot_sdm_err_vs_rows.py 
# starting 0
# perr=1, calling fast_emp with epochs=200
# starting Fast_sdm_empirical, nrows=51, ncols=512, nact=1, actions=10, states=100, choices=10
# fast_sdm_empirical epochs=200, perr=0.099795, std_error=0.00942777677928365
# Done calling fast empirical
# starting 1
# perr=2, calling fast_emp with epochs=200
# starting Fast_sdm_empirical, nrows=86, ncols=512, nact=2, actions=10, states=100, choices=10
# fast_sdm_empirical epochs=200, perr=0.010215, std_error=0.0032555759859047985
# Done calling fast empirical
# starting 2
# perr=3, calling fast_emp with epochs=500
# starting Fast_sdm_empirical, nrows=125, ncols=512, nact=2, actions=10, states=100, choices=10
# fast_sdm_empirical epochs=500, perr=0.00111, std_error=0.0010324243313676793
# Done calling fast empirical
# starting 3
# perr=4, calling fast_emp with epochs=500
# starting Fast_sdm_empirical, nrows=168, ncols=512, nact=2, actions=10, states=100, choices=10
# fast_sdm_empirical epochs=500, perr=0.000114, std_error=0.00033616067586795455
# Done calling fast empirical
# starting 4
# perr=5, calling fast_emp with epochs=1000
# starting Fast_sdm_empirical, nrows=196, ncols=512, nact=3, actions=10, states=100, choices=10
# fast_sdm_empirical epochs=1000, perr=1e-05, std_error=9.949874371066199e-05
# Done calling fast empirical
# starting 5
# starting 6

# real	17m43.876s
# user	16m0.467s
# sys	0m12.667s

# Now try with binary counters
