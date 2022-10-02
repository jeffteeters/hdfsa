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
from labellines import labelLine, labelLines
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

pftypes = {
	"predict": {"fmt": "o-k", "dashes": None, "lw": 2, "linestyle":"solid", "alpha":1,
		"markersize":None, "label":None},
	"ecount_dotted_line": {"fmt": ":+g", "dashes":None, "linestyle":None, "lw":3,
		"alpha":0.6, "markersize":None, "show_clm": False, "label":None},
	"epmf_dashed_line":{"fmt": "--+r", "dashes":None, "linestyle":None, "lw":3,
		"alpha":0.6, "markersize":None, "label":None},

	"ecount_striped_line": {"fmt": ":Pc", "dashes":None, "linestyle":(0, (2,2)), "lw":5,
		"alpha":0.5, "markersize":12, "show_clm": False, "label":None},
	"epmf_striped_line":{"fmt": "--*r", "dashes":None, "linestyle":(2, (2,2)), "lw":5,
		"alpha":0.5, "markersize":12, "label":None},

	"ecount_striped_line_a3": {"fmt": ":^b", "dashes":None, "linestyle":(0, (2,2)), "lw":5,
		"alpha":0.5, "markersize":12, "show_clm": False, "label":None},
	"epmf_striped_line_a3":{"fmt": "--Xg", "dashes":None, "linestyle":(2, (2,2)), "lw":5,
		"alpha":0.5, "markersize":12, "label":None},

				# 	"predict": {"fmt": ".-k", "dashes": None, "lw": 1, "linestyle":"solid", "label":"Prediction"
				# , "alpha":1, "markersize":None}, # [12,6,12,6,3,6]
			 # "epmf":{"fmt":":+m", "dashes":None, "linestyle":None, "lw":None, "label":"empirical pmf", "alpha":1,
			 # 	"markersize":None},
			 # "ecount":{"fmt":"--*k", "dashes":None, "linestyle":None, "lw":None, "label":"empirical count", "alpha":1,
			 # 	"markersize":None,
			 # 	"show_clm": False, "max_ie":7},

}

pfmap = {
	"predict": pftypes["predict"],
	# ecount": pftypes["ecount_dotted_line" ], "epmf": pftypes["epmf_dashed_line"],
	"ecount": pftypes["ecount_striped_line"], "epmf": pftypes["epmf_striped_line"],
	"ecount_a3": pftypes["ecount_striped_line_a3"], "epmf_a3": pftypes["epmf_striped_line_a3"],
}

pfmt = {
	"S1": { # hamming match, solid line thin
			"predict": {**pfmap["predict"], "label":"Prediction"},
			# {"fmt": ".-k", "dashes": None, "lw": 1, "linestyle":"solid", "label":"Prediction", "alpha":1,
			# 		"markersize":None},
				"epmf": {**pfmap["epmf"], "label":"empirical pmf"},
				# {"fmt": ":+m",
				# 	"dashes":None,
				 # 	"linestyle":None, # (0,(1,1)),
				 # 	"lw":2,
				 # 	"label":"empirical pmf", "alpha":0.6,
				# 	"markersize":None},
				"ecount": {**pfmap["ecount"], "label":"empirical count", "max_ie": 6},
				# {"fmt":"--*c", "dashes":None,
				# 	"linestyle": None, # (1,(1,1)),
				# 	"lw":2,
				# 	"label":"empirical count", "alpha":0.6,
				# 	"markersize":None,
				# 	"show_clm": False},
			 "annotate":{"xy":(5.1e4, 2.3e-3), "xytext":(6e4, 7.0e-3), "text":"S1"}
			 },
	"S2": { # dot match, solid line thin
			"predict": pftypes["predict"],
			# {"fmt": "o-k", "dashes": None, "lw": 2, "linestyle":"solid", "label":None, "alpha":1,
			# 		"markersize":None},
			"epmf": pfmap["epmf"],
			# {"fmt":":+r", "dashes":None,
			# 	"linestyle":(0,(1,1)),
			# 	"lw":4, "label":None, "alpha":0.6,
			# 	"markersize":None},
			"ecount": {**pfmap["ecount"], "max_ie":6},
			# {"fmt":"*y", "dashes":None,
			# 	"linestyle":(1,(1,1)),
			# 	"lw":4, "label":None, "alpha":0.6,
			# 	"markersize":None,
			# 	"show_clm": False},
			# "annotate":{"xy":(3.9e4, 3.0e-4), "xytext":(2.95e4, 5.5e-5), "text":"S2"}  # "arrow_start": (3.5e4, 1.5e-5)
			"annotate":{"xy":(3.9e4, 3.0e-4), "xytext":(2.0e4, 5.0e-5), "text":"S2"}  # "arrow_start": (3.5e4, 1.5e-5)
			 },
	"A1": { # full counter, threshold sum
			"predict": {**pftypes["predict"], "label":"Prediction"},
			 "epmf": {**pfmap["epmf"], "label":"empirical pmf"},
			 "ecount": {**pfmap["ecount"], "max_ie": 7, "label":"empirical count"},
			# "predict": {"fmt": ".-k", "dashes": None, "lw": 1, "linestyle":"solid", "label":"Prediction"
			# 	, "alpha":1, "markersize":None}, # [12,6,12,6,3,6]
			#  "epmf":{"fmt":":+m", "dashes":None, "linestyle":None, "lw":None, "label":"empirical pmf", "alpha":1,
			#  	"markersize":None},
			#  "ecount":{"fmt":"--*k", "dashes":None, "linestyle":None, "lw":None, "label":"empirical count", "alpha":1,
			#  	"markersize":None,
			#  	"show_clm": False, "max_ie":7},
			 "annotate":{"xy":(156, 2.4e-4), "xytext":(220, 1.1e-3), "text":"A1 & A3"}
			 },
	"A2": { # Binary counter, threshold sum
			"predict": pftypes["predict"],
			"epmf": pfmap["epmf"],
			"ecount": {**pfmap["ecount"], "max_ie": 7},
			# "predict": {"fmt": ".-k", "dashes": None, "lw": 1, "linestyle":"solid", "label":None, "alpha":1,
			# 	"markersize":None},  # [12,6,3,6,3,6]
			#  "epmf":{"fmt":":+m", "dashes":None, "linestyle":None, "lw":None, "label":None , "alpha":1,
			#  	"markersize":None},
			#  "ecount":{"fmt":"--*k", "dashes":None, "linestyle":None, "lw":None, "label":None, "alpha":1,
			#  	"markersize":None,
			#  	"show_clm": False, "max_ie":7},
			 "annotate":{"xy":(249, 2.8e-5), "xytext":(280, 8.0e-5), "text":"A2"}
			 },
	"A3": { # binary counter, non-thresholded sum
			"predict": {"fmt": None },
			"epmf": {**pfmap["epmf_a3"], "label": "A3 empirical pmf"},
			"ecount": {**pfmap["ecount_a3"], "max_ie": 7, "label": "A3 empirical count"},
			# "predict": {"fmt": None, "dashes": None, "lw": 1, "linestyle":"solid", "label":None, "alpha":1,
			# 	"markersize":None},  # [12,6,12,6,3,6]
			#  "epmf":{"fmt":":+g", "dashes":None, "linestyle":None, "lw":None, "label": "A3 empirical pmf" , "alpha":1,
			#  	"markersize":None},
			#  "ecount":{"fmt":"--*g", "dashes":None, "linestyle":None, "lw":None, "label": "A3 empirical count", "alpha":1,
			#  	"markersize":None,
			#  	"show_clm": False, "max_ie":7},
			 },
	"A4": { # full counter, non-thresholded sum
			"predict": pftypes["predict"],
			# "predict": {"fmt": ".-k", "dashes": None, "lw": 1, "linestyle":"solid", "label":None, "alpha":1,
			# 	"markersize":None},  # [12,6,12,6,3,6]
			 "epmf": pfmap["epmf"],
			# {"fmt":":+m", "dashes":None, "linestyle":None, "lw":None, "label":None, "alpha":1,
			# 	"markersize":None},
			"ecount": {**pfmap["ecount"], "max_ie": 7},
			 # "ecount":{"fmt":"--*k", "dashes":None, "linestyle":None, "lw":None,"label":None, "alpha":1,
			 # 	"markersize":None, "show_clm": False, "max_ie":7},
			 # "annotate":{"xy":(95, 1.0e-4), "xytext":(62, 2.5e-5), "text":"A4"}
			 "annotate":{"xy":(92, 1.0e-4), "xytext":(30, 2.9e-5), "text":"A4"}
			 },

}

		# if theoretical_bundle_err and mtype == "bind":
		# 	plt.errorbar(xvals["bind"], theoretical_bundle_err, label="Superposition theory", fmt="-",
		# 		linestyle=':', linewidth=6, alpha=0.7)
		# if theoretical_sdm_err and mtype == "sdm":
		# 	plt.errorbar(xvals["bind"], theoretical_sdm_err, label="SDM theory", fmt="-o", linestyle='dashed')

def plot_error_vs_dimension(mtype="sdm", include_jaeckel=False, panel=None):
	# dimenion is ncols (width) of bundle or nrows
	# panel is axis if this is plot in a panel
	# fontsize used for axis labels
	global pfmt
	assert mtype in ("sdm", "bundle")
	plp = plt if panel is None else panel
	edb = Empirical_error_db()
	names = edb.get_memory_names(mtype=mtype)
	if panel is None:
		fig, ax = plp.subplots()
	else:
		ax = plp
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
				(dim_id, ie, size, ncols, nact, pe, epochs, mean, std,
					recall_time_mean, recall_time_std, recall_time_min, rt_nepochs,
					match_counts, distract_counts, pmf_err) = dim
				if jaeckel_error is not None:
					k = 1000; d=100
					jaeckel_error[i] = sdm_jaeckel.SdmErrorAnalytical(size,k,d,nact=nact,word_length=512)
			else:
				# dim_id, ie, size, pe, epochs, mean, std = dim
				(dim_id, ie, size, pe, epochs, mean, std,
					recall_time_mean, recall_time_std, recall_time_min, rt_nepochs,
					match_counts, distract_counts, pmf_err) = dim
			sizes[i] = size
			predicted_error[i] = pe
			if False: # mi["short_name"] in ("S1", "S2") and ie == 6:
				print("%s, ie=%s - len(match_counts)=%s, len(distract_counts)=%s" % (name, ie, len(match_counts), len(distract_counts)))
				print("match_counts=%s..." % match_counts[0:80])
				print("distract_counts=%s..." % distract_counts[0:80])
				import pdb; pdb.set_trace()
			pmf_error[i] = pmf_err if pmf_err is not None else np.nan
			clm = std / math.sqrt(epochs) * 1.96 if mean is not None else None
			if mean is not None and mean > 0: # and mean > clm:
				empirical_error[i] = mean
				empirical_clm[i] = clm  # std / math.sqrt(epochs) * 1.96  # 95% confidence interval for the mean (CLM)
			else:
				empirical_error[i] = np.nan
				empirical_clm[i] = np.nan
		# plot arrays filled by above
		short_name = mi["short_name"]
		# print("plotting %s" % short_name)
		# print("sizes=%s" % sizes)
		# print("empirical_error=%s" % empirical_error)
		# print("empirical_clm=%s" % empirical_clm)
		# plt.errorbar(sizes, empirical_error, yerr=empirical_clm, fmt="-o", label=name)
		pf = pfmt[short_name]
		if pf["predict"]["fmt"] is not None:
			ax.errorbar(sizes, predicted_error, yerr=None, label=pf["predict"]["label"],
				fmt=pf["predict"]["fmt"], dashes=pf["predict"]["dashes"], lw=pf["predict"]["lw"],
				linestyle=pf["predict"]["linestyle"], alpha=pf["predict"]["alpha"],
				markersize=pf["predict"]["markersize"],)
		empirical_error[pf["ecount"]["max_ie"]:] = np.nan  # don't show empirical ecount above max_ie
		empirical_clm[pf["ecount"]["max_ie"]:] = np.nan  # don't show empirical ecount above max_ie	
		ax.errorbar(sizes, empirical_error, yerr=empirical_clm if pf["ecount"]["show_clm"] else None,
				label=pf["ecount"]["label"], # color='tab:blue',
				fmt=pf["ecount"]["fmt"],
				linestyle=pf["ecount"]["linestyle"], lw=pf["ecount"]["lw"],
				alpha=pf["ecount"]["alpha"],markersize=pf["ecount"]["markersize"],
				markeredgecolor='brown')
		ax.errorbar(sizes, pmf_error, yerr=None, label=pf["epmf"]["label"],
				fmt=pf["epmf"]["fmt"], linestyle=pf["epmf"]["linestyle"], lw=pf["epmf"]["lw"],
				alpha=pf["epmf"]["alpha"], markersize=pf["epmf"]["markersize"], markeredgecolor='brown')
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
			plp.errorbar(sizes, jaeckel_error, yerr=None, fmt="8m", label="jaeckel_error")
	xlabel = "SDM num rows" if mtype == "sdm" else "Superposition vector width"
	ylabel = "Fraction error ($10^{-n})$"
	if panel is None:
		plp.title("%s empirical vs. predicted error rw1_noroll_v3" % mtype)
		fontsize = 14 if mtype == "sdm" else None
		plp.xlabel(xlabel, fontsize=fontsize)
		plp.ylabel(ylabel, fontsize=fontsize)
	else:
		title = "Superposition vector memories" if mtype == "bundle" else "Associative memories"
		plp.set_title(title)
		plp.set_xlabel(xlabel, fontsize=16)
		plp.set_ylabel(ylabel, fontsize=16)  # only set if on left of graph
	if panel is None:
		plp.yscale('log')
	else:
		plp.set_yscale('log')
	yticks = (10.0**-(np.arange(9.0, 0, -1.0)))
	# ylabels = [10.0**(-i) for i in range(9, 0, -1)]
	ylabels = ["%s" % i for i in range(9, 0, -1)]
	if panel is None:
		plt.yticks(yticks, ylabels)
	else:
		# import pdb; pdb.set_trace()
		if mtype == "bundle":
			# set limits so y-axis in panel are aligned
			panel.set_ylim([8.206982489549015e-11, 0.2906972567491242]) # limits from sdm, printed below
		ylow, yhi = panel.get_ylim()
		# print("%s ylow=%s, yhi=%s" % (mtype, ylow, yhi))
		panel.set_yticks(yticks)
		panel.set_yticklabels(ylabels)
		# panel.yaxis.label.set_size(16)

	# plt.yticks(yticks, ylabels)
	# xlabels = ["%s/%s" % (rows[i], nacts[i]) for i in range(num_steps)]
	# plt.xticks(rows[0:num_steps], xlabels)
	plp.grid()
	plp.legend(loc='upper right')
	# figure = plt.gcf() # get current figure
	# figure.set_size_inches(8, 6)
	if panel is None:
		plp.show()


mem_linestyles = {
	# linestyles from:
	# https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
	"S1": {"color":'tab:brown',  "linestyle":(0, (5,5))}, # loosly dashed
	"S2": {"color":'tab:orange', "linestyle":(0, (5,1))}, # densly dashed
	"A1": {"color":'tab:green', "linestyle": (0, (3, 5, 1, 5))}, # dash dotted (one dot)
	"A2": {"color":'tab:blue', "linestyle": (0, (3, 5, 1, 5, 1, 5))}, # dash dotted (two dots)
	"A3": {"color":"tab:red" , "linestyle": (0, (3, 5, 1, 5, 1, 5, 1, 5))}, # dash dotted (three dots)
	"A4": {"color":'tab:purple', "linestyle": (0, (3, 5, 1, 5, 1, 5, 1, 5, 1, 5))}, # dash dotted (four dot)
}

def plot_size_vs_error(fimp=0.0, log_scale=False, ratio_base=None, panel=None):
	# size is number of bits
	# fimp is fraction of item memory present
	# plot for both bundle and sdm
	# ratio_base - a short name (e.g. "A3") to display ratios of sizes
	# panel is plot subplot if plotting in panel
	global mem_linestyles
	plp = plt if panel is None else panel
	edb = Empirical_error_db()
	names = edb.get_memory_names()
	item_memory_len = 110
	num_dims = 9
	ratio_data = []
	short_names = []
	labelsize = None if not panel else 16
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
				(dim_id, ie, nrows, ncols, nact, pe, epochs, mean, std,
					recall_time_mean, recall_time_std, recall_time_min, rt_nepochs,
					match_counts, distract_counts, pmf_err) = dim
				size = (nrows * ncols * bits_per_counter) + fimp*(nrows*ncols + item_memory_len*ncols) # address memory plus item memory
			else:
				# dim_id, ie, ncols, pe, epochs, mean, std = dim
				(dim_id, ie, ncols, pe, epochs, mean, std,
					recall_time_mean, recall_time_std, recall_time_min, rt_nepochs,
					match_counts, distract_counts, pmf_err) = dim
				bits_per_counter = 1 if match_method == "hamming" else 8  # match_method indicates of bundle binarized or not
				size = ncols * bits_per_counter + fimp*(ncols * item_memory_len)  # bundle + item memory
			sizes[i] = size
			predicted_error[i] = -math.log10(pe)
		ratio_data.append(sizes)
		short_name = mi["short_name"]
		short_names.append(short_name)  # for displaying ratio data
		# plot arrays filled by above
		if ratio_base is None:
			plp.errorbar(predicted_error, sizes, yerr=None, fmt="-", label=short_name,
				color=mem_linestyles[short_name]["color"], # linestyle=mem_linestyles[short_name]["linestyle"]
				)

	# labelLines(plt.gca().get_lines(), zorder=2.5)
	if fimp == 0:
		xvals = [8.7, 8.5, 4.5, 6.5, 7.5, 6.5]
	else:
		assert fimp == 1.0
		xvals = [3.5, 4.5, 5.5, 7.5, 8.7, 6.5]
	if ratio_base is None:
		# normal plot, show sizes (not ratio)
		tmp_axs = plp.gca() if panel is None else panel
		# labelLines(plp.gca().get_lines(), xvals=xvals, align=False, zorder=2.5)
		labelLines(tmp_axs.get_lines(), xvals=xvals, align=False, zorder=2.5)
		fimp_msg = "$f_{imp}=%s$" % fimp
		title = "Size (bits) vs error with %s" % fimp_msg if panel is None else fimp_msg
		if panel is None:
			plp.title(title)
		else:
			plp.set_title(title, fontsize=18)
		xlabel = "Error rate ($10^{-n}$)"
		if panel is None:
			plp.xlabel(xlabel)
		# else:
			# plp.set_xlabel(xlabel)  # don't include xlabel in panel since this is in the top row
		ylabel = "Size in Mb ($10^6$ bits)"
		if panel is None:
			plp.ylabel(ylabel)
		else:
			if fimp == 0:
				# only set ylabel if fimp == 0 because that is the left most plot
				# import pdb; pdb.set_trace()
				plp.set_ylabel(ylabel, fontsize=16)
		make_y_axis_scale_10e6(panel=panel, labelsize=labelsize)
		# Following should do the same as the above call
		# display y-axis value in number, 10**6 bytes
		# ax = plt.gca()
		# ylow, yhi = ax.get_ylim()
		# assert yhi > 10**6
		# yticks_orig = ax.get_yticks()
		# assert yticks_orig[0] < 1
		# assert yticks_orig[-1] > yhi
		# yticks_new = yticks_orig[1:-1]  # strip off negative value and value greater than yhi
		# # print ("ylow=%s, yhi=%s, yticks=\n%s" % (ylow, yhi, yticks_orig))
		# # yaxis_values = [s for s in range(0, int(yhi), 200000)]
		# yaxis_labels = ["%s" % (s / 10**6) for s in yticks_new]
		# plt.yticks(yticks_new, yaxis_labels)
		# ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
		if log_scale:
			plp.yscale('log')
		# xlabels = ["%s/%s" % (rows[i], nacts[i]) for i in range(num_steps)]
		# plt.xticks(rows[0:num_steps], xlabels)
		plp.grid()
		# plt.legend(loc='upper left')
		if panel is None:
			plp.show()

	# display ratio of of size to size of A3
	if ratio_base is not None:
		# print("ratios to size of %s, fimp=%s:" % (ratio_base, fimp))
		idx_base = short_names.index(ratio_base)
		for i in range(len(ratio_data)):
			short_name = short_names[i]
			ratios = [(ratio_data[i][j]/ratio_data[idx_base][j]) for j in range(len(ratio_data[i]))]
			ratios_str = [("%.3f" % ratios[j]) for j in range(len(ratios))]
			# print("%s\t%s" % (short_names[i], "\t".join(ratios_str)))
			plp.errorbar(predicted_error, ratios, yerr=None, fmt="-", label=short_name,
				color=mem_linestyles[short_name]["color"], # linestyle=mem_linestyles[short_name]["linestyle"]
				)
		tmp_axs = plp.gca() if panel is None else panel
		# labelLines(plt.gca().get_lines(), xvals=xvals, align=False, zorder=2.5)
		labelLines(tmp_axs.get_lines(), xvals=xvals, align=False, zorder=2.5)
		title = "Ratio of sizes to %s size with fimp=%s" % (ratio_base,fimp)
		xlabel = "Error rate ($10^{-n}$)"
		ylabel = "Ratio to %s size" % ratio_base
		if panel is None:
			plp.title(title)
			plp.xlabel(xlabel)
			plp.ylabel(ylabel)
		else:
			# plp.set_title(title) # don't include title on panel since this is in the lower row
			plp.set_xlabel(xlabel, fontsize=16)
			if fimp == 0:
				# only include ylabel if fimp == 0 because that is on the left side of the panel
				ylabel = "Ratio to smallest size"
				plp.set_ylabel(ylabel, fontsize=16)
		if fimp == 0:
			# set ymin to zero, make step size 1 for y-axis ticks (1 to 10)
			# ax = plp.gca()
			tmp_axs.set_ylim(ymin=0)
			start, end = tmp_axs.get_ylim()
			stepsize = 1
			# tmp_axs.yaxis.set_ticks(np.arange(start, end, stepsize))
		plp.grid()
		if panel is None:
			plp.show()

def make_y_axis_scale_10e6(panel=None, labelsize=None):
	# display y-axis value in number, 10**6 bytes
	if panel is None:
		ax = plp.gca()
	else:
		ax = panel
	ylow, yhi = ax.get_ylim()
	# assert yhi > 10**6
	yticks_orig = ax.get_yticks()
	assert yticks_orig[0] < 1
	assert yticks_orig[-1] > yhi
	# print("yticks assertion failed.  ylow=%s, yhi=%s, yticks_org=\n%s" % (ylow, yhi, yticks_orig))	
	yticks_new = yticks_orig[1:-1]  # strip off negative value and value greater than yhi
	# print ("ylow=%s, yhi=%s, yticks=\n%s" % (ylow, yhi, yticks_orig))
	# yaxis_values = [s for s in range(0, int(yhi), 200000)]
	yaxis_labels = ["%g" % (s / 10**6) for s in yticks_new]
	if panel is None:
		plt.yticks(yticks_new, yaxis_labels)
	else:
		# import pdb; pdb.set_trace()
		panel.set_yticks(yticks_new)
		panel.set_yticklabels(yaxis_labels)
		panel.yaxis.label.set_size(16)

def plot_memory_size_and_efficiency_vs_fimp(ie, plot_type="size", log_scale=False, zoom=False, ratio_base=None, panel=None):
	# ie is a fixed error rate, range(1,10); error rate is 10**(-ie)
	# plot_type = "size" or "memory_efficiency"
	# displays both bundle and sdm
	# zoom is True to zoom in on lower left
	# panel is plot subplot if plotting in panel
	global mem_linestyles
	plp = plt if panel is None else panel
	assert plot_type in ("size", "memory_efficiency")
	if zoom:
		assert plot_type == "size", "zoom only implemented for plot_type size"
	assert ie in range(1,10)
	if plot_type == "size":
		zoom_msg = " (zoom)" if zoom else ""
		if panel is None:
			title = "Size required vs fimp for error rate=$10^{-%s}$%s" % (ie, zoom_msg)
		else:
			title = "$0 \leq f_{imp} \leq 1$"
		ylabel = "Size in Mb ($10^6$ bits)"
	else:
		title = "Memory efficiency vs fimp for error rate=$10^{-%s}$" % ie
		ylabel = "Memory efficiency (%)"
	edb = Empirical_error_db()
	names = edb.get_memory_names()
	item_memory_len = 110
	max_bits_used_in_counters = 8
	ratio_data = []
	short_names = []
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
		if zoom:
			num_fimps = 15
			last_fimp = 0.14
		else:
			num_fimps = 50
			last_fimp = 1.0
		fimps = np.linspace(0, last_fimp, num=num_fimps, endpoint=True)
		yvals = np.empty(num_fimps, dtype=np.float64)  # stores either size or efficiency
		dim = mi["dims"][ie-1]
		assert dim[1] == ie  # make sure retrieved specified dimension
		# empirical_error = np.empty(ndims, dtype=np.float64)
		for i in range(num_fimps):
			fimp = fimps[i]
			if mtype == "sdm":
				# dim_id, ie, nrows, ncols, nact, pe, epochs, mean, std = dim
				(dim_id, ie, nrows, ncols, nact, pe, epochs, mean, std,
					recall_time_mean, recall_time_std, recall_time_min, rt_nepochs,
					match_counts, distract_counts, pmf_err) = dim
				# size = (nrows * ncols * bits_per_counter) + fimp*(nrows*ncols + item_memory_len*ncols) # address memory plus item memory
				size = (nrows * ncols * bits_used_in_counters) + fimp*(nrows*ncols + item_memory_len*ncols) # address memory plus item memory
				changing_bits_size = (nrows * ncols * bits_used_in_counters)
			else:
				# dim_id, ie, ncols, pe, epochs, mean, std = dim
				(dim_id, ie, ncols, pe, epochs, mean, std,
					recall_time_mean, recall_time_std, recall_time_min, rt_nepochs,
					match_counts, distract_counts, pmf_err) = dim
				# bits_per_counter = 1 if match_method == "hamming" else 8  # match_method indicates of bundle binarized or not
				# size = ncols * bits_per_counter + fimp*(ncols * item_memory_len)  # bundle + item memory
				size = ncols * bits_used_in_counters + fimp*(ncols * item_memory_len)  # bundle + item memory
				changing_bits_size = (ncols * bits_used_in_counters)
			yvals[i] = size if plot_type == "size" else (changing_bits_size *100.0 / size)
			# display sizes at ends for including in paper
			if plot_type == "size" and (i == 0 or i == num_fimps - 1):
				if i == 0:
					size_at_start = size
				else:
					size_increase_ratio = round(size / size_at_start, 2)
					# ratio_txt = ", ratio=%s" % size_increase_ratio
					# following print was used to make a table of ratio changes
					# print("%s & %s & %s & %s" % (mi["short_name"], int(size_at_start), int(size), size_increase_ratio))
					# print("at fimp=%s, %s size = %s%s" % (fimp,mi["short_name"], size, ratio_txt))
		# plot arrays filled by above
		short_name = mi["short_name"]
		if ratio_base is None:
			plp.errorbar(fimps, yvals, yerr=None, fmt="-", label=short_name,
				color=mem_linestyles[short_name]["color"], # linestyle=mem_linestyles[short_name]["linestyle"]
				)
		# save info for displaying ratios
		ratio_data.append(yvals)
		short_names.append(short_name)
	assert plot_type == "size", "labeled lines xvals not setup for plot_type memory"
	if not zoom:
		xvals = [0.44, 0.7, 0.65, 0.88, 0.95, 0.75]
	else:
		# xvals = [0.03, 0.05, 0.03, 0.07, 0.11, 0.09]
		xvals = [0.135, 0.125, 0.03, 0.11, 0.13, 0.09]
	if ratio_base is None:
		tmp_axs = plp.gca() if panel is None else panel
		labelLines(tmp_axs.get_lines(), xvals=xvals, align=False, zorder=2.5)
		xlabel = "Fraction item memory present ($f_{imp}$)"
		if panel is None:
			plp.title(title)
			plp.xlabel(xlabel)
			plp.ylabel(ylabel)
		else:
			plp.set_title(title)
			# plp.set_xlabel(xlabel)  # don't include xlabel in panel since this is in the top row
			# plp.set_ylabel(ylabel)  # don't include ylabel in panel since this is not the left most
		# display y-axis value in number, 10**6 bytes
		make_y_axis_scale_10e6(panel=panel)
		# if False: # plot_type == "size":
		# 	ax = plt.gca()
		# 	ylow, yhi = ax.get_ylim()
		# 	assert yhi > 10**6
		# 	yticks_orig = ax.get_yticks()
		# 	assert yticks_orig[0] < 1
		# 	assert yticks_orig[-1] > yhi
		# 	yticks_new = yticks_orig[1:-1]  # strip off negative value and value greater than yhi
		# 	yaxis_labels = ["%s" % (s / 10**6) for s in yticks_new]
		# 	plt.yticks(yticks_new, yaxis_labels)
		if log_scale:
			plp.yscale('log')
		# xlabels = ["%s/%s" % (rows[i], nacts[i]) for i in range(num_steps)]
		# plt.xticks(rows[0:num_steps], xlabels)
		plp.grid()
		# plt.legend(loc='upper left')
		if not panel:
			plt.show()

	if ratio_base is not None:
		# display ratio of of size to size of A3
		# print("ratios to size of %s:" % ratio_base)
		idx_base = short_names.index(ratio_base)
		for i in range(len(ratio_data)):
			short_name = short_names[i]
			ratios = [(ratio_data[i][j]/ratio_data[idx_base][j]) for j in range(len(ratio_data[i]))]
			# ratios_str = [("%.3f" % ratios[j]) for j in range(len(ratios))]
			# print("%s\t%s" % (short_names[i], "\t".join(ratios_str)))
			plp.errorbar(fimps, ratios, yerr=None, fmt="-", label=short_name,
				color=mem_linestyles[short_name]["color"], # linestyle=mem_linestyles[short_name]["linestyle"]
				)
		tmp_axs = plp.gca() if panel is None else panel
		labelLines(tmp_axs.get_lines(), xvals=xvals, align=False, zorder=2.5)
		title = "Ratio of sizes to %s size with varying fimp" % (ratio_base,)
		xlabel = "Fraction item memory present ($f_{imp}$)"
		ylabel = "Ratio to size of %s" % ratio_base
		if not panel:
			plp.title(title)
			plt.xlabel(xlabel)
			plt.ylabel(ylabel)
		else:
			# plp.set_title(title) # don't include title on panel since this in the lower row
			plp.set_xlabel(xlabel, fontsize=16)
			# plp.set_ylabel(ylabel) # don't include ylabel in panel since this is not the left most
		# if fimp == 0:
		# 	# set ymin to zero, make step size 1 for y-axis ticks (1 to 10)
		# 	ax = plt.gca()
		# 	ax.set_ylim(ymin=0)
		# 	start, end = ax.get_ylim()
		# 	stepsize = 1
		# 	ax.yaxis.set_ticks(np.arange(start, end, stepsize))
		plp.grid()
		if not panel:
			plp.show()

def plot_operations_vs_error(parallel=False, log_scale=False, include_recall_times=True,
		zoom_lower=False, ratio_base=None, panel=None):
	# operations is number if byte operations, or parallel byte operations (if parallel is True)
	# plot for both bundle and sdm
	# set zoom_lower True to change labels and scale for zooming lower part of plot
	# ratio_base is short name to plot ratios of computations
	# panel is sublot axis if plotting in a panel
	global mem_linestyles
	plp = plt if panel is None else panel
	edb = Empirical_error_db()
	names = edb.get_memory_names()
	item_memory_len = 100
	scale_factor = None
	ratio_data = []
	short_names = []
	for name in names:
		mi = edb.get_minfo(name)
		mtype = mi["mtype"]
		bits_per_counter = mi["bits_per_counter"]
		match_method = mi["match_method"]
		ndims = len(mi["dims"])
		operations = np.empty(ndims, dtype=np.float64)
		recall_times_mean = np.empty(ndims, dtype=np.float64)
		recall_times_min = np.empty(ndims, dtype=np.float64)
		recall_times_clm = np.empty(ndims, dtype=np.float64)
		predicted_error = np.empty(ndims, dtype=np.float64)
		dsf = 2.2  # dot product scale factor, amount dot product time greater than hamming
		dsfq = 3.4
		for i in range(ndims):
			dim = mi["dims"][i]
			if mtype == "sdm":
				# dim_id, ie, nrows, ncols, nact, pe, epochs, mean, std = dim
				(dim_id, ie, nrows, ncols, nact, pe, epochs, mean, std,
					recall_time_mean, recall_time_std, recall_time_min, rt_nepochs,
					match_counts, distract_counts, pmf_err) = dim
				ops_form_address = ncols # current_state XOR input
				ops_address_compare = ncols * nrows if not parallel else ncols
				ops_select_active = nrows * nact  # should be log nact
				ops_add_counters = ncols * nact if not parallel else nact
				ops_unpermute = ncols
				ops_bind_address = ncols
				if match_method == "hamming":
					# threshold, use hamming
					ops_item_memory_compare = item_memory_len * ncols * 2 if not parallel else ncols * 2
				else:
					# don't threshold, match using dot product
					# ops_item_memory_compare = item_memory_len * ncols * 2 * dsf if not parallel else ncols * 2 * dsf
					ops_item_memory_compare = item_memory_len * ncols * (1+ dsfq) if not parallel else ncols * (1+ dsfq)
				ops_select_smallest = item_memory_len
				ops = (ops_form_address + ops_address_compare + ops_select_active + ops_add_counters +
						ops_unpermute + ops_bind_address + ops_item_memory_compare + ops_select_smallest)
				# size = (nrows * ncols * bits_per_counter) + fimp*(nrows*ncols + item_memory_len*ncols) # address memory plus item memory
			else:
				# calculate operations for bundle (superposition vector)
				# dim_id, ie, ncols, pe, epochs, mean, std = dim
				(dim_id, ie, ncols, pe, epochs, mean, std,
					recall_time_mean, recall_time_std, recall_time_min, rt_nepochs,
					match_counts, distract_counts, pmf_err) = dim
				ops_form_address = ncols # current_state XOR input
				ops_bind_address = ncols # superposition_vector XOR address  (bind superposition vector to accress)
				ops_rotate = ncols  # rotate bound vector left
				if match_method == "hamming":
					ops_compute_im_distances = item_memory_len * ncols *2 if not parallel else ncols *2 #  ???8 for bits to bytes * 2 # two for counting
				else:
					# match method is dot product
					# ops_compute_im_distances = item_memory_len * ncols *2 * dsf if not parallel else ncols *2 * dsf#  ???8 for bits to bytes * 2 # two for counting
					ops_compute_im_distances = item_memory_len * ncols *(1+ dsfq) if not parallel else ncols *(1+ dsfq)#
				ops_find_smallest_distance = item_memory_len
				ops = (ops_form_address + ops_bind_address + ops_rotate + ops_compute_im_distances +
						ops_find_smallest_distance)
				# else 8  # match_method indicates of bundle binarized or not
				# ops = ncols * (item_memory_len + 2)/8 + item_memory_len if not parallel else ncols / 8 + item_memory_len
				# size = ncols * bits_per_counter + fimp*(ncols * item_memory_len)  # bundle + item memory
			operations[i] = ops
			recall_times_mean[i] = recall_time_mean
			recall_times_clm = recall_time_std / math.sqrt(rt_nepochs) * 1.96
			recall_times_min[i] = recall_time_min
			np.empty(ndims, dtype=np.float64)
			predicted_error[i] = -math.log10(pe)
		# plot arrays filled by above
		short_name = mi["short_name"]
		ratio_data.append(operations)
		short_names.append(short_name)  # for displaying ratio data
		# plot arrays filled by above
		pzs = 1 if not parallel or not zoom_lower else 10**6  # change values of y-axis to 10^6
		if ratio_base is None:
			plp.errorbar(predicted_error, operations / pzs, yerr=None, fmt="-", label=short_name,
				color=mem_linestyles[short_name]["color"])
			if include_recall_times and not parallel:
				if scale_factor is None:
					#import pdb; pdb.set_trace()
					# ax = plt.gca()
					# ax2 = ax.twinx() 
					# calculate one scale factor
					scale_factor = operations[0] / recall_times_min[0]
				recall_times_min *= scale_factor
				plp.errorbar(predicted_error, recall_times_min, yerr=None, fmt="--", label=None,
					color=mem_linestyles[short_name]["color"])
				# ax2.errorbar(predicted_error, recall_times_min, yerr=None, fmt="--", label=None,
				# 	color=mem_linestyles[short_name]["color"])
	if not parallel:
		if not zoom_lower:
			# normal view
			# xvals = [3.5, 4.5, 8.5, 7.5, 5.5, 6.5] # orig
			xvals = [3.5, 4.5, 5.5, 6.5, 8.5, 7.5]
		else:
			# change location of S1 and S2 labels to front of line so show up in zoomed plot
			# xvals = [1.5, 1.15, 8.5, 7.5, 5.5, 6.5] # orig
			xvals = [1.5, 1.15, 5.5, 6.5, 8.5, 7.5]
	else:
		# parallel views
		if not zoom_lower:
			# xvals = [3.5, 4.5, 8.5, 7.5, 5.5, 6.5] # orig
			# xvals = [3.5, 4.5, 7.5, 5.5, 6.5, 8.5] # orig_2
			xvals = [3.5, 4.5, 5.5, 6.5, 8.5, 7.5]
		else:
			# change location of S1 and S2 labels to front of line so show up in zoomed plot
			# xvals = [1.1, 1.33, 8.5, 7.5, 5.5, 6.5] # orig
			xvals = [1.1, 1.33, 7.5, 5.5, 6.5, 8.5] # orig_2
			xvals = [1.1, 1.33, 5.5, 6.5, 8.5, 7.5]

	if ratio_base is None:
		xlabel = "Error rate ($10^{-n}$)"
		if panel is None:
			title = "Operations vs error; parallel=%s, log_scale=%s, zoom=%s" % (parallel, log_scale, zoom_lower)
			ylabel = "Number operations ($10^6$)" if not parallel else "Number parallel operations ($10^6$)"
			plp.title(title)
			plp.xlabel(xlabel)
			plp.ylabel(ylabel)
		else:
			title = "Serial operations" if not parallel else "Parallel operations"
			ylabel = "Number operations ($10^6$)"
			plp.set_title(title)
			# plp.set_xlabel(xlabel)  # don't include xlabel in panel since this is in the top row
			if not parallel:
				plp.set_ylabel(ylabel)  # only include ylabel in panel if this is the left most

		tmp_axs = plp.gca() if panel is None else panel
		labelLines(tmp_axs.get_lines(), xvals=xvals, align=False, zorder=2.5)

		## old code
		# if panel is None:
		# 	plt.title("Operations vs error; parallel=%s, log_scale=%s, zoom=%s" % (parallel, log_scale, zoom_lower))
		# 	xlabel = "Error rate ($10^{-n}$)"
		# 	plt.xlabel(xlabel)
		# 	if not parallel:
		# 		plt.ylabel("Number operations ($10^6$)")
		# else:
		# 	plt.ylabel("Number parallel operations ($10^6$)")

		if not zoom_lower:
			# display y-axis value in number, 10**6 bytes
			if not parallel and panel is not None:
				# specialized hack to add in ticks 10, 20, 30, 40 to panel A
				current_yticks = tmp_axs.get_yticks()
				expected_yticks = np.array([-20000000.,         0.,  20000000.,  40000000.,  60000000.])
				assert np.array_equal(current_yticks, expected_yticks)
				yticks_new = np.arange(0., 40000001., 10000000.)
				tmp_axs.set_yticks(yticks_new)
				ytick_labels = ["%g" % (s / 10**6) for s in yticks_new]
				tmp_axs.set_yticklabels(ytick_labels)
				panel.yaxis.label.set_size(16)
			else:
				make_y_axis_scale_10e6(panel=panel)
		if log_scale:
			plp.yscale('log')
		# xlabels = ["%s/%s" % (rows[i], nacts[i]) for i in range(num_steps)]
		# plt.xticks(rows[0:num_steps], xlabels)
		plp.grid()

		# add legend for dash and solid lines; from:
		# https://stackoverflow.com/questions/62705904/add-entry-to-matplotlib-legend-without-plotting-an-object
		# fig, ax = plt.subplots()
		# ax = plt.gca()
		if not parallel:
			legend_elements = [Line2D([0], [0], color='k', ls='-',lw=1, label='Number of operations'),
					   Line2D([0], [0], color='k', ls='--',lw=1, label='Empirical recall time')]
			loc = "upper left" if not zoom_lower else "upper right"
			tmp_axs.legend(handles=legend_elements, loc=loc)


		# add secondary axis on right side for time required, from:
		# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/secondary_axis.html
		if include_recall_times and not parallel:
			# print("scale_factor=%s" % scale_factor)
			scale_factor_sec = scale_factor * 10**9 # 1000 # * 1000 to convert to ms per recall of one transition
			def ops2time(ops):
				return ops/scale_factor_sec
			def time2ops(time):
				return time*scale_factor_sec
			secay = tmp_axs.secondary_yaxis('right', functions=(ops2time, time2ops))
			secay.set_ylabel('Time (ms)')

			# change resolution of scale to be int when possible, e.g. 3.0 => 3
			# from: https://stackoverflow.com/questions/61269526/customising-y-labels-on-a-secondary-y-axis-in-matplotlib-to-format-to-thousands
			secay.get_yaxis().set_major_formatter(FormatStrFormatter('%g'))
		# 	ylow, yhi = secay.get_ylim()
		# 	yticks_orig = secay.get_yticks()
		# 	# assert yticks_orig[0] < 1
		# 	# assert yticks_orig[-1] > yhi
		# 	# yticks_new = yticks_orig[1:-1]  # strip off negative value and value greater than yhi
		# 	# print ("ylow=%s, yhi=%s, yticks=\n%s" % (ylow, yhi, yticks_orig))
		# 	# yaxis_values = [s for s in range(0, int(yhi), 200000)]
		# 	yaxis_labels = ["%g" % s for s in yticks_orig]
		# 	secay.set_ticklabels(yaxis_labels)
		# 	# plt.yticks(yticks_new, yaxis_labels)

		# FormatStrFormatter
		# ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
		if panel is None:
			plp.show()

	# display ratios of computations to base
	if ratio_base is not None:
		# print("ratios to size of %s:" % ratio_base)
		idx_base = short_names.index(ratio_base)
		for i in range(len(ratio_data)):
			short_name = short_names[i]
			ratios = [(ratio_data[i][j]/ratio_data[idx_base][j]) for j in range(len(ratio_data[i]))]
			ratios_str = [("%.3f" % ratios[j]) for j in range(len(ratios))]
			# print("%s\t%s" % (short_names[i], "\t".join(ratios_str)))
			plp.errorbar(predicted_error, ratios, yerr=None, fmt="-", label=short_name,
				color=mem_linestyles[short_name]["color"], # linestyle=mem_linestyles[short_name]["linestyle"]
				)

		tmp_axs = plp.gca() if panel is None else panel
		labelLines(tmp_axs.get_lines(), xvals=xvals, align=False, zorder=2.5)
		title = "Ratio of operations to %s with parallel=%s" % (ratio_base,parallel)
		xlabel = "Error rate ($10^{-n}$)"
		ylabel = "Ratio to %s operations" % ratio_base
		if panel is None:
			plp.title(title)
			plp.xlabel(xlabel)
			plp.ylabel(ylabel)
		else:
			# don't plot title on panel because ratio is in second row
			plp.set_xlabel(xlabel, fontsize=16)
			if not parallel:
				# only show y-label if on leftmost panel
				ylabel = "Ratio to fewest operations"
				plp.set_ylabel(ylabel, fontsize=16)
		plp.grid()
		if not panel:
			plp.show()


def size_vs_error_panel():
	# create panel of plots of size vs error
	# fig, axs = plt.subplots(2,3)
	fig, axs = plt.subplots(2,3)
	plt.rcParams.update({'font.size': 12})
	plot_size_vs_error(fimp=0, log_scale=False, panel=axs[0,0])
	# axs[0,0].text(-0.1, 1.15, "A", transform=axs[0,0].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
	plot_size_vs_error(fimp=0, log_scale=False, ratio_base="S1", panel=axs[1,0]) # plot ratio
	plot_size_vs_error(fimp=1, log_scale=False, panel=axs[0,1])
	plot_size_vs_error(fimp=1, log_scale=False, ratio_base="A3", panel=axs[1,1]) # plot ratio
	ie=6
	plot_memory_size_and_efficiency_vs_fimp(ie, plot_type="size", log_scale=False, zoom=False, panel=axs[0,2])
	plot_memory_size_and_efficiency_vs_fimp(ie, plot_type="size", log_scale=False, zoom=False, ratio_base="A3", panel=axs[1,2])
	labels=[["A","C","E"], ["B","D","F"]]
	for row in range(2):
		for col in range(3):
			axs[row,col].tick_params(axis='both', which='major', labelsize=12)
			axs[row,col].tick_params(axis='both', which='minor', labelsize=12)
			label = labels[row][col]
			axs[row,col].text(-0.04, 1.10, label, transform=axs[row,col].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
	plt.show()


def operations_vs_error_panel():
	# create panel of plots of operations vs error
	fig, axs = plt.subplots(2,2)
	plt.rcParams.update({'font.size': 14})
	plot_operations_vs_error(parallel=False, log_scale=False, panel=axs[0,0])
	plot_operations_vs_error(parallel=False, log_scale=False, ratio_base="A1", panel=axs[1,0])
	plot_operations_vs_error(parallel=True, log_scale=False, panel=axs[0,1])
	plot_operations_vs_error(parallel=True, log_scale=False, ratio_base="A1", panel=axs[1,1])
	labels=[["A","C"], ["B","D"]]
	for row in range(2):
		for col in range(2):
			axs[row,col].tick_params(axis='both', which='major', labelsize=12)
			axs[row,col].tick_params(axis='both', which='minor', labelsize=12)
			label = labels[row][col]
			axs[row,col].text(-0.04, 1.10, label, transform=axs[row,col].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
	plt.show()

def error_vs_dimension_panel():
	# create panel of plot of superposition and associative memory error vs dimension
	fig, axs = plt.subplots(1,2)
	plt.rcParams.update({'font.size': 14})
	plot_error_vs_dimension("bundle", panel=axs[0])
	plot_error_vs_dimension("sdm", panel=axs[1])
	labels=["A","B"]
	for col in range(2):
		axs[col].tick_params(axis='both', which='major', labelsize=12)
		axs[col].tick_params(axis='both', which='minor', labelsize=12)
		label = labels[col]
		axs[col].text(-0.04, 1.03, label, transform=axs[col].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
	plt.show()


def main():
	if False:
		plot_error_vs_dimension("bundle")
		plot_error_vs_dimension("sdm")
	if False:
		# plot ratio of sizes
		plot_size_vs_error(fimp=0, log_scale=False, ratio_base="S1") # 1.0/64.0)
		plot_size_vs_error(fimp=1, log_scale=False, ratio_base="A3") # 1.0/64.0)
		ie=6
		plot_memory_size_and_efficiency_vs_fimp(ie, plot_type="size", log_scale=False, zoom=False, ratio_base="A3")
	if False:
		plot_size_vs_error(fimp=0, log_scale=False) # 1.0/64.0)
		plot_size_vs_error(fimp=0, log_scale=True) # 1.0/64.0)
		plot_size_vs_error(fimp=1, log_scale=False) # 1.0/64.0)
		plot_size_vs_error(fimp=1,  log_scale=True) # 1.0/64.0)
	if False:
		for ie in range(1, 10):
			# ie = 3
			print("ie=%s" % ie)
			plot_memory_size_and_efficiency_vs_fimp(ie, plot_type="size", log_scale=False, zoom=False)
			# plot_memory_size_and_efficiency_vs_fimp(ie, plot_type="size", log_scale=False, zoom=True)
		# plot_memory_size_and_efficiency_vs_fimp(ie, plot_type="size", log_scale=True)
	if False:
		plot_operations_vs_error(parallel=False, log_scale=False)
		plot_operations_vs_error(parallel=False, log_scale=False, zoom_lower=True)
		plot_operations_vs_error(parallel=False, log_scale=True, zoom_lower=True)
		plot_operations_vs_error(parallel=True, log_scale=False)
		plot_operations_vs_error(parallel=True, log_scale=False, zoom_lower=True)
		plot_operations_vs_error(parallel=True, log_scale=True, zoom_lower=True)
	# plot_memory_size_and_efficiency_vs_fimp(ie, plot_type="memory_efficiency")
	if False:
		# plot ratio of operations
		plot_operations_vs_error(parallel=False, log_scale=False, ratio_base="A1")
		plot_operations_vs_error(parallel=True, log_scale=False, ratio_base="A1")
	if False:
		size_vs_error_panel()
	if True:
		operations_vs_error_panel()
	if False:
		error_vs_dimension_panel()



if __name__ == "__main__":
	# compare_sdm_ham_dot()
	warnings.simplefilter('error', UserWarning)
	main()

