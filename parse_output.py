
# script to parse output of plot_sdm_err_vs_nact and plot values
# allows selecting which values of nact to show / plot
# and displaying ratio of binary_sdm prediction to empirical

import re
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
bits_per_counter=1

file_name = "archive/2022_06_plots/2022_06_10/sdm_bc1_nrows50_to_400-nact_1,3,5,7_vs.txt"
with open(file_name) as f:
	output = f.read()

# pattern = re.compile(r'[^\n]*\nstarting nrows=(\d+), nact=(\d), epochs=(\d+),\n(?:prob[^\n]+\n)?bsm_err=([^,]+),'
# 	' emp_err=([^,]+), std=([^\n]+)\n')

# pat =re.compile(r'[^\n]*\nstarting nrows=(\d+), nact=(\d), epochs=(\d+),[^\n]*?\nbsm_err=([^,]+), emp_err=([^,]+), std=([^\n]+)\n')
# pat3 =re.compile(r'[^\n]*\nstarting nrows=(\d+), nact=(\d), epochs=(\d+),\nbsm_err=([^,]+), emp_err=([^,]+), std=([^\n]+)\n')
pat2=re.compile(r'[^\n]*\nstarting nrows=(\d+), nact=(\d), epochs=(\d+),\s*\n(?:prob[^\n]+\n)?bsm_err=([^,]+), emp_err=([^,]+), std=([^\n]+)\n')

nacts = [1,3,5,7]
k=1000
rows_to_test = list(range(50, 421, 20))
emp_mean=np.empty((len(nacts), len(rows_to_test)), dtype=np.float64)
emp_clm=np.empty((len(nacts), len(rows_to_test)), dtype=np.float64)
bsm_err=np.empty((len(nacts), len(rows_to_test)), dtype=np.float64)
anl_err=np.empty((len(nacts), len(rows_to_test)), dtype=np.float64)

for (nrows, nact, epochs, bsm_er, emp_err, std) in re.findall(pat2, output):
	nrows = int(nrows)
	nact = int(nact)
	epochs = int(epochs)
	bsm_er = float(bsm_er)
	emp_err=float(emp_err)
	std=float(std)

	nact_idx = nacts.index(nact)
	nrows_idx = rows_to_test.index(nrows)
	emp_mean[nact_idx, nrows_idx] = emp_err
	emp_clm[nact_idx, nrows_idx] = (std / math.sqrt(epochs)) * 1.96
	bsm_err[nact_idx, nrows_idx] = bsm_er


# display ratio of predicted to empirical
for i in range(len(nacts)):
	print("nact=%s" % nacts[i])
	for j in range(len(rows_to_test)):
		print("rows=%s, empirical=%s, predicted=%s, ratio=%s" % (rows_to_test[j], emp_mean[i,j], bsm_err[i,j],
			round(bsm_err[i,j]/emp_mean[i,j], 3)))

sys.exit("done for now")

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
		if nacts[j] in (7,):
			plt.errorbar(rows_to_test, emp_mean[j], yerr=emp_clm[j], fmt="-o", label="empirical nact=%s" % nacts[j])
			plt.errorbar(rows_to_test, bsm_err[j], yerr=None, fmt="-o", label="bsm predicted nact=%s" % nacts[j])
		# plt.errorbar(rows_to_test, anl_err[j], yerr=None, fmt="-o", label="anl predicted nact=%s" % nacts[j])
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


    # print("nrows=%s, nact=%s, epochs=%s, bsm_err=%s, emp_err=%s, std=%s" % (
    # 	nrows, nact, epochs, bsm_err, emp_err, std))

"""
(base) Jeffs-MacBook:hdfsa jt$ time python plot_sdm_err_vs_nact.py
starting nrows=50, nact=1, epochs=400,
bsm_err=0.09744423457541718, emp_err=0.10477, std=0.009384940063740417

starting nrows=50, nact=3, epochs=400,
prob_one_trial_overlap=0.06, peak_num_overlaps=59
bsm_err=0.18385813235924686, emp_err=0.21645000000000003, std=0.01183353286216758

starting nrows=50, nact=5, epochs=400,
prob_one_trial_overlap=0.1, peak_num_overlaps=100
bsm_err=0.21097715330018152, emp_err=0.3125775, std=0.013006498135547486

starting nrows=50, nact=7, epochs=400,
prob_one_trial_overlap=0.14, peak_num_overlaps=139
bsm_err=0.2233438542899311, emp_err=0.4227575, std=0.01584199147045598

starting nrows=70, nact=1, epochs=400,
bsm_err=0.03434493679186701, emp_err=0.03728, std=0.006688916205186009

starting nrows=70, nact=3, epochs=400,
prob_one_trial_overlap=0.04285714285714286, peak_num_overlaps=42
bsm_err=0.07146743000055866, emp_err=0.087455, std=0.008223622985035246
"""

