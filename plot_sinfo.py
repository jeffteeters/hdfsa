# script to parse generated file
import re
import sys
import os.path

import pprint
pp = pprint.PrettyPrinter(indent=4)
import statistics


if len(sys.argv) != 2:
	sys.exit("Usage %s <file_name.txt>" % sys.argv[0])
file_name = sys.argv[1]
assert os.path.isfile(file_name), "file %s not found" % file_name
if file_name[0] == "s":
	# assume 'sdata*.txt' - storage vs error
	xvar = "storage"
elif file_name[0] == "f":
	# assume 'fdata*.txt' - bit flips vs error
	xvar = "pflip"  # percent flip
else:
	sys.exit("File name first character should be 'f' or 's': found: %s" % file_name[0])


fp = open(file_name, 'r')
while True:
	# scan for line indicating start of data
	line = fp.readline()
	if not line:
		sys.exit("start not found")
	if line == "-----Data starts here-----\n":
		break

header = fp.readline()
assert header == "rid\t%s\tmtype\tmem_len\terror_count\tfraction_error\tprobability_of_error\ttotal_storage_required\n" % xvar

sdata = {}
pattern = r"(\d+\.\d+)\t([^\t]+)\t(\w+)\t(\d+)\t(\d+)\t([^\t]+)\t([^\t]+)\t(\d+)\n"
# 1.1	100000	bind	7207	641	0.641	0.6433422825527839	99996
# 2.2     2.5     sdm     1300    0       0.0     4.427697670830096e-06   672640 # for fdata.txt (bit flips)
while True:
	# Get next line from file
	line = fp.readline()
	if not line:
		break
	match = re.match(pattern, line)
	if not match:
		sys.exit("Match failed on:\n%s" % line)
	rid, xval, mtype, mlen, error_count, fraction_error, perror, storage_required = match.group(1,2,3,4,5,6,7,8)

	# add new line to sdata dictionary

	if mtype not in sdata:
		sdata[mtype] = {}
	xval = int(xval) if xvar == "storage" else float(xval)   # xval is either storage (bytes) or bits flipped (%)
	perror = float(perror)
	if xval not in sdata[mtype]:
		sdata[mtype][xval] = {"pmin":perror, "pmax":perror, "recs":[], "pelist": [perror]}
	else:
		if perror < sdata[mtype][xval]["pmin"]:
			sdata[mtype][xval]["pmin"] = perror
		elif perror > sdata[mtype][xval]["pmax"]:
			sdata[mtype][xval]["pmax"] = perror
		sdata[mtype][xval]["pelist"].append(perror)
	info = {"error_count":int(error_count), "fraction_error":float(fraction_error), "perror":perror,
	   "storage_required":int(storage_required), "mlen":int(mlen), "error_count":int(error_count) }
	sdata[mtype][xval]["recs"].append(info)

# pp.pprint(sdata)

# create arrays used for plotting
xvals = {}
yvals = {}
ebar = {}
for mtype in sdata:
	xvals[mtype] = sorted(list(sdata[mtype].keys()))  # will be sorted storage (or pflips)
	yvals[mtype] = []
	ebar[mtype] = []
	for xval in xvals[mtype]:
		# pmid = (sdata[mtype][xval]["pmin"] + sdata[mtype][xval]["pmax"]) / 2.0
		pmid = statistics.mean(sdata[mtype][xval]["pelist"])
		yvals[mtype].append(pmid)
		# prange = sdata[mtype][xval]["pmax"] - sdata[mtype][xval]["pmin"]
		prange = statistics.stdev(sdata[mtype][xval]["pelist"])
		ebar[mtype].append(prange / 2)


from scipy.stats import norm
import math

def compute_theoretical_bundle_error(bl):
	# compute theoretical error based on bundle length (bl)
	k = 1000  # number of items stored in bundle
	delta = 0.5 - 0.4 / math.sqrt(k - 0.44)  # from Pentti's paper
	mean_hamming_match = delta * bl
	variance_hamming_match = delta * (1 - delta) * bl
	mean_hamming_distractor = 0.5 * bl
	variance_distractor = 0.5 * (1 - 0.5) * bl
	# make mean and variance of difference distribution
	diff_dist_mean = mean_hamming_distractor - mean_hamming_match
	diff_dist_variance = variance_hamming_match + variance_distractor
	# want probability diff_dist distribution is less than zero (meaning, hamming distance of distractor less than
	# hamming distance of match)
	loc = diff_dist_mean
	scale = math.sqrt(diff_dist_variance)
	eperr = norm.pdf(0, loc, scale)
	pcorr_one = 1.0 - eperr 
	pcorr_all = pcorr_one ** 99
	perror_all = 1 - pcorr_all
	return perror_all


def get_bundle_lengths(xvals, sdata):
	mtype = "bind"
	bundle_lengths = []
	for xval in xvals[mtype]:
		bundle_lengths.append(sdata[mtype][xval]["recs"][0]["mlen"])
	return(bundle_lengths)

add_theoretical_bundle = xvar == "storage"

if add_theoretical_bundle:
	bundle_lengths = get_bundle_lengths(xvals, sdata)
	terr = [ compute_theoretical_bundle_error(bl) for bl in bundle_lengths]
	# print("bundle_lengths = %s\nterr = %s" % (bundle_lengths, terr))
	# sys.exit("done for now")

# import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# fig, ax = plt.subplots()
fig = plt.figure()
plt.yscale('log')
for mtype in xvals:
	label = "superposition vector" if mtype == "bind" else "SDM"
	plt.errorbar(xvals[mtype], yvals[mtype], yerr=ebar[mtype], label=label, fmt="-o")
if add_theoretical_bundle:
	plt.errorbar(xvals["bind"], terr, label="bundle theory", fmt="-o")

title = "Fraction error vs storage size (bytes)" if xvar == "storage" else "Fraction error vs percent bits flipped" 
plt.title(title)
xlabel = "Storage (bytes)" if xvar == "storage" else '% bits flipped'
plt.xlabel(xlabel)
if xvar == "storage":
	xaxis_labels = ["100k", "200k", "300k", "400k", "500k", "600k", "700k", "800k", "900k", "1e6" ]
	plt.xticks(xvals[mtype],xaxis_labels)
ylabel = "Fraction error"
plt.ylabel(ylabel)

# plt.axes().yaxis.get_ticklocs(minor=True)

# Initialize minor ticks
# plt.axes().yaxis.minorticks_on()

loc = 'upper right' if xvar == "storage" else 'lower right'
plt.legend(loc=loc)
plt.grid()
plt.show()

