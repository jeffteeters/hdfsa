# script to parse generated file
import re
import sys

file_name = "sdata.txt"

fp = open(file_name, 'r')
header = fp.readline()
assert header == "rid\tstorage\tmtype\tmem_len\terror_count\tfraction_error\tprobability_of_error\ttotal_storage_required\n"

sdata = {}
pattern = r"(\d+\.\d+)\t(\d+)\t(\w+)\t(\d+)\t(\d+)\t([^\t]+)\t([^\t]+)\t(\d+)\n"
# 1.1	100000	bind	7207	641	0.641	0.6433422825527839	99996
while True:
    # Get next line from file
    line = fp.readline()
    if not line:
    	break
    match = re.match(pattern, line)
    if not match:
    	sys.exit("Match failed on:\n%s" % line)
    rid, storage, mtype, mlen, error_count, fraction_error, perror, storage_required = match.group(1,2,3,4,5,6,7,8)

    # add new line to sdata dictionary

    if mtype not in sdata:
    	sdata[mtype] = {}
    storage = int(storage)
    perror = float(perror)
    if storage not in sdata[mtype]:
    	sdata[mtype][storage] = {"pmin":float(perror), "pmax":float(perror), "recs":[]}
    else:
    	if perror < sdata[mtype][storage]["pmin"]:
    		sdata[mtype][storage]["pmin"] = perror
    	elif perror > sdata[mtype][storage]["pmax"]:
    		sdata[mtype][storage]["pmax"] = perror
    info = {"error_count":int(error_count), "fraction_error":float(fraction_error), "perror":perror,
    	"storage_required":int(storage_required)}
    sdata[mtype][storage]["recs"].append(info)

# create arrays used for plotting
xvals = {}
yvals = {}
ebar = {}
for mtype in sdata:
	xvals[mtype] = sorted(list(sdata[mtype].keys()))  # will be sorted storage
	yvals[mtype] = []
	ebar[mtype] = []
	for storage in xvals[mtype]:
		pmid = (sdata[mtype][storage]["pmin"] + sdata[mtype][storage]["pmax"]) / 2.0
		yvals[mtype].append(pmid)
		prange = sdata[mtype][storage]["pmax"] - sdata[mtype][storage]["pmin"]
		ebar[mtype] = prange


import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure()
plt.yscale('log')
for mtype in xvals:
	plt.errorbar(xvals[mtype], yvals[mtype], yerr=ebar[mtype], label=mtype)


plt.legend(loc='upper right')
plt.show()

