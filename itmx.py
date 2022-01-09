# item memory explorer

import numpy as np
import argparse
import matplotlib.pyplot as plt
import sys
import statistics
import math
from fractions import Fraction
from scipy import special
import random
import gmpy2
from gmpy2 import xmpz
import pprint
pp = pprint.PrettyPrinter(indent=4)
import sdm
from scipy.stats import norm


class Env:
	# stores environment settings and data arrays

	# command line arguments
	parms = [
		{ "name":"num_items", "kw":{"help":"Number of items in item memory", "type":int},
		  "flag":"s", "required_init":"i", "default":101 },
		{ "name":"word_length", "kw":{"help":"Word length for item memory, 0 to disable", "type":int},
		  "flag":"w", "required_init":"i", "default":512 },
		{ "name":"num_trials", "kw":{"help":"Number of trials to run (used when generating table)", "type":int},
		  "flag":"t", "required_init":"i", "default":3 },
		{ "name":"loop_start", "kw":{"help":"Start value for loop","type":int},
		  "flag":"x", "required_init":"", "default":0},
		{ "name":"loop_step", "kw":{"help":"Step value for loop","type":int},
		  "flag":"y", "required_init":"", "default":5},
		{ "name":"loop_stop", "kw":{"help":"Stop value for loop","type":int},
		  "flag":"z", "required_init":"", "default":50},
		{ "name":"values_to_plot", "kw":{"help":"Values to plot.  "
			"l-loop variable, e-prob error, d-dplen, g-gallen, p-pentti mean","type":str,
			"choices":["l", "e", "d", "g", "p"]}, "flag":"p", "required_init":"", "default":"l"},
		{ "name":"show_histograms", "kw":{"help":"Show histograms","type":int,
			"choices":[0, 1]}, "flag":"H", "default":0},
		{ "name":"seed", "kw":{"help":"Random number seed","type":int},
		  "flag":"e", "required_init":"i", "default":2021},
		{ "name":"debug", "kw":{"help":"Debug mode","type":int, "choices":[0, 1]},
		  "flag":"d", "required_init":"", "default":0},
		  ]

	def __init__(self):
		self.parse_command_arguments()
		self.display_settings()
		self.initialize()

	def parse_command_arguments(self):
		parser = argparse.ArgumentParser(description='Test formats for storing hard locations for a SDM.',
			formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		# also make parser for interactive updating parameters (does not include defaults)
		iparse = argparse.ArgumentParser(description='Update sdm parameters.') # exit_on_error=False)
		for p in self.parms:
			parser.add_argument("-"+p["flag"], "--"+p["name"], **p["kw"], default=p["default"])
			iparse.add_argument("-"+p["flag"], "--"+p["name"], **p["kw"])  # default not used for interactive update
		self.iparse = iparse # save for later parsing interactive input
		args = parser.parse_args()
		self.pvals = {p["name"]: getattr(args, p["name"]) for p in self.parms}

	def initialize(self):
		# initialize sdm, char_map and merge
		print("Initializing.")
		np.random.seed(self.pvals["seed"])


	def display_settings(self):
		print(self.get_settings())

	def get_settings(self):
		msg = "Arguments:\n"
		msg += " ".join(sys.argv)
		msg += "\nSettings:\n"
		for p in self.parms:
			msg += "%s %s: %s\n" % (p["flag"], p["name"], self.pvals[p["name"]])
		return msg

class Figure:
	# stores information for plot figure and creates plot

	def __init__(self, ptype="errorbar", nbins=None,
			title=None, xvals=None, grid=False, xlabel=None, ylabel=None,
			xaxis_labels=None, legend_location=None,
			yvals=None, ebar=None, legend=None, fmt="-o", logyscale=False):
		assert ptype in ("errorbar", "hist")
		self.ptype = ptype
		self.nbins = nbins
		self.title = title
		self.xvals = xvals
		self.grid = grid
		self.xlabel = xlabel
		self.ylabel = ylabel
		self.xaxis_labels = xaxis_labels
		self.legend_location = legend_location
		self.line_info = []
		self.legend = legend
		self.fmt = fmt
		self.logyscale = logyscale
		if yvals is not None:
			assert ptype == "errorbar"
			self.add_line(yvals, ebar, legend, fmt)

	def add_line(self, yvals, ebar=None, legend=None, fmt=None):
		if fmt is None:
			fmt = self.fmt
		info = {"yvals": yvals, "ebar":ebar, "legend":legend, "fmt":fmt}
		self.line_info.append(info)

	def make_plot(self, figure_number):
		fig = plt.figure(figure_number)
		plt.title(self.title)
		if self.ptype == "errorbar":
			for info in self.line_info:
				plt.errorbar(self.xvals, info["yvals"], yerr=info["ebar"], label=info["legend"], fmt=info["fmt"])
		elif self.ptype == "hist":
			n, bins, patches = plt.hist(self.xvals, self.nbins, facecolor='g', alpha=0.75, label=self.legend)
		else:
			sys.exit("unknown ptype: %s" % self.ptype)
		if self.xlabel is not None:
			plt.xlabel(self.xlabel)
		if self.ylabel is not None:
			plt.ylabel(self.ylabel)
		if self.xaxis_labels is not None:
			plt.xticks(self.xvals,self.xaxis_labels)
		if self.legend_location is not None:
			plt.legend(loc=self.legend_location)
		if self.grid:
			plt.grid()
		if self.logyscale:
			plt.yscale('log')
		plt.show()


class Calculator:
	# Calculate values to plot

	def __init__(self, env):
		self.env = env
		self.figures = []
		start = self.env.pvals["loop_start"]
		step = self.env.pvals["loop_step"]
		stop = self.env.pvals["loop_stop"]
		self.xvals = list(range(start, stop, step))
		values_to_plot = self.env.pvals["values_to_plot"]
		if values_to_plot == "l":
			# loop values
			self.calculate_loop_values()
			self.calculate_loop_values(0.5)
		elif values_to_plot == "e":
			self.calculate_item_memory_match_values()
		elif values_to_plot in ("d", "g"):
			self.test_dplen_function()
		elif values_to_plot == "p":
			self.show_petti_mean_function()
		else:
			sys.exit("values_to_plot '%s' not implemented" % values_to_plot)

	def show_plots(self):
		for i in range(len(self.figures)):
			print("plotting # %s" % i)
			self.figures[i].make_plot(i+1)

	def calculate_loop_values(self, factor=0.75):
		yvals = self.xvals
		fig = Figure(title="Loop variable %s" % factor, xvals=self.xvals, grid=True,
			xlabel="loop variable value", ylabel="loop var *factor", xaxis_labels=None,
			legend_location="lower right",
			yvals=yvals, ebar=None, legend="loop var")
		yvf = [y * factor for y in yvals]
		fig.add_line(yvf, ebar=None, legend="loop var*%s" % factor)
		self.figures.append(fig)

	def hamming(self, a, b):
		# return hamming distance between binary vectors (ndarray) a and b
		return np.count_nonzero(a!=b)

	def int_hamming(self, a, b):
		# return hamming distance between binary vectors (int) a and b
		return gmpy2.popcount(a^b)

	def get_nbins(self, xv):
		# calculate number of bins to use in histogram for values in xv
		xv_range = max(xv) - min(xv)
		if xv_range == 0:
			nbins = 3
		else:
			nbins = int(xv_range) + 1
		# if xv_range < 10:
		#   nbins = 10
		# elif xv_range > 100:
		#   nbins = 100
		# else:
		#   nbins = int(xv_range) + 1
		return nbins

	def calculate_item_memory_match_values(self):
		# display plot of recall error (correctly matching corrupted item to correct item in item memory)
		# vs bit flips, plot along with theoretical recall based on Frady sequence paper
		num_items = self.env.pvals["num_items"]
		word_length = self.env.pvals["word_length"]
		item_memory = np.random.randint(0, high=2, size=(num_items, word_length), dtype=np.int8)
		fs = {}   # found stats
		rng = np.random.default_rng()
		for xval in self.xvals:
			fs[xval] = {"num_matches_correct": 0, "num_matches_tried":0,
				"item_hamming": [], # hamming between corrupted item and item_memory entry
				"distractor_hamming": [],  # hamming between corrupted item and distractors
				"min_distractor_hamming": [],  # minimum uncorrelated (min of "distractor_hamming" for each item)
				"margin": [], # margin between hamming for item and closest distractor ("min_distractor_hamming" - "item_hamming")
				}
			number_bits_to_flip = int(word_length * num_items * xval / 100.0)  # convert from percent to number of bits
			if self.env.pvals["debug"]:
				print("xval=%s, number_bits_to_flip=%s" % (xval, number_bits_to_flip))
			for trial in range(self.env.pvals["num_trials"]):
				corrupted_items = item_memory.flatten()
				bits_to_flip = rng.choice(word_length* num_items, size=number_bits_to_flip, replace=False, shuffle=False)
				# bits_to_flip = np.random.choice(word_length*num_items, size=number_bits_to_flip, replace=False)
				corrupted_items[bits_to_flip] = 1 - corrupted_items[bits_to_flip]
				corrupted_items.resize( (num_items, word_length) )
				for item_id in range(num_items):
					item_hamming = self.hamming(corrupted_items[item_id], item_memory[item_id])
					fs[xval]["item_hamming"].append(item_hamming)
					if self.env.pvals["debug"]:
						print("item_id=%s, corrupted_item=%s, item=%s, item_hamming=%s" % (item_id, corrupted_items[item_id],
							item_memory[item_id], item_hamming))
					min_distractor_hamming = word_length
					for did in range(num_items):  # distractor_id
						if did == item_id:
							continue
						distractor_hamming = self.hamming(corrupted_items[item_id], item_memory[did])
						fs[xval]["distractor_hamming"].append(distractor_hamming)
						if self.env.pvals["debug"]:
							print("did=%s, distractor_hamming=%s" % (did, distractor_hamming))
						if distractor_hamming < min_distractor_hamming:
							min_distractor_hamming = distractor_hamming
					fs[xval]["min_distractor_hamming"].append(min_distractor_hamming)
					fs[xval]["margin"].append(min_distractor_hamming - item_hamming)
					fs[xval]["num_matches_tried"] += 1
					if item_hamming < min_distractor_hamming:
						fs[xval]["num_matches_correct"] += 1
				if self.env.pvals["debug"]:
					print("Hamming item=%s, min_distractor_hamming=%s" % (item_hamming, min_distractor_hamming))
			# mean_hamming_to_closest_distractor = statistics.mean(hamming_to_closest_distractor)
			# stdev_hamming_to_closest_distractor = statistics.stdev(hamming_to_closest_distractor)
			# hamming_to_closest_distractor_all.append((mean_hamming_to_closest_distractor,
			#       stdev_hamming_to_closest_distractor))
			# if self.env.pvals["debug"]:
			#   print("hamming_to_closest_distractor=%s" % hamming_to_closest_distractor)
			#   print("mean=%s, stdev=%s" % (mean_hamming_to_closest_distractor, stdev_hamming_to_closest_distractor))
			#   import pdb; pdb.set_trace()
		# make plots
		# plot histograms for each xval
		if self.env.pvals["show_histograms"]:
			for xval in self.xvals:
				for ht in ("item_hamming", "distractor_hamming", "min_distractor_hamming","margin"):
					nbins = self.get_nbins(fs[xval][ht])
					fig = Figure(ptype="hist", nbins=nbins,
						title="%s (%s%% bit flips)" % (ht ,xval), xvals=fs[xval][ht], grid=True,
						xlabel="Item", ylabel="Hamming", xaxis_labels=None,
						legend_location="upper left",
						yvals=None, ebar=None, legend="item hamming")
					self.figures.append(fig)
			# hist = np.histogram(fs[xval]["item_hamming"], bins = nbins)
			# print("Item hamming (%s%% bit flips), histogram=\n%s" % (xval, hist))
		# make plots for all xvals
		yvals = [(fs[xval]["num_matches_tried"] - fs[xval]["num_matches_correct"])*100.0/fs[xval]["num_matches_tried"]
			for xval in self.xvals]
		fig = Figure(title="Recall error vs bit flips", xvals=self.xvals, grid=True,
			xlabel="bit flips (%)", ylabel="recall error (percent)", xaxis_labels=None,
			legend_location="upper left",
			yvals=yvals, ebar=None, legend="recall error", logyscale=True)
		theory_error = self.compute_theoretical_recall_error()
		fig.add_line(theory_error, legend="Theoretical error")
		self.figures.append(fig)
		margin_mean = [statistics.mean(fs[xval]["margin"]) for xval in self.xvals]
		margin_ebar = [statistics.stdev(fs[xval]["margin"]) for xval in self.xvals]
		fig = Figure(title="Hamming difference between item in closest distractor vs bit flips",
			xvals=self.xvals, grid=True,
			xlabel="bit flips (%)", ylabel="Hamming difference", xaxis_labels=None,
			legend_location="lower left",
			yvals=margin_mean, ebar=margin_ebar, legend="margin (difference in hamming)")
		# yham = [hamming_to_closest_distractor_all[i][0] * 100.0 / word_length for i in range(num_xvals)]
		# yhamebar = [hamming_to_closest_distractor_all[i][1] * 50.0 / word_length for i in range(num_xvals)]
		# fig = Figure(title="Distractor hamming % vs bit flips", xvals=self.xvals, grid=True,
		#   xlabel="bit flips (%)", ylabel="Hamming difference (percent)", xaxis_labels=None,
		#   legend_location="lower right",
		#   yvals=yham, ebar=yhamebar, legend="Distractor hamming diff (%)")

		# fig.add_line(yham, ebar=yhamebar, legend="Distractor hamming diff (%)")
		self.figures.append(fig)

	def compute_theoretical_recall_error(self):
		# compute theoretical recall error using equation in Frady paper (or what I think it should be)
		error = []
		wl = self.env.pvals["word_length"]
		num_distractors = self.env.pvals["num_items"] - 1
		for xval in self.xvals:
			print(xval)
			pflip = Fraction(xval, 100)  # probability of bit flip
			pdist = Fraction(1, 2) # probability of distractor bit matching
			match_hamming = []
			distractor_hamming = []
			for k in range(wl):
				ncomb = Fraction(math.comb(wl, k))
				match_hamming.append(ncomb * pflip ** k * (1-pflip) ** (wl-k))
				distractor_hamming.append(ncomb * pdist ** k * (1-pdist) ** (wl-k))
			# print("sum match_hamming=%s, distractor_hamming=%s" % (sum(match_hamming), sum(distractor_hamming)))
			if self.env.pvals["show_histograms"]:
				fig = Figure(title="match and distractor hamming distributions for %s%% bit flips" % xval,
					xvals=range(wl), grid=False,
					xlabel="Hamming distance", ylabel="Probability", xaxis_labels=None,
					legend_location="upper right",
					yvals=match_hamming, ebar=None, legend="match_hamming", fmt="-g")
				fig.add_line(distractor_hamming, legend="distractor_hamming", fmt="-m")
				self.figures.append(fig)
			dhg = Fraction(1) # fraction distractor hamming greater than match hamming
			pcor = Fraction(0)  # probability correct
			for k in range(wl):
				dhg -= distractor_hamming[k]
				pcor += match_hamming[k] * dhg ** num_distractors
			error.append(float((1 - pcor) * 100))  # convert to percent
		return error

	def prob_negative_after_subtract(self, distractor_hamming_mean,
		distractor_hamming_stdev, match_hamming_mean, match_hamming_stdev):
		# compute probatility of error given the match and distractor distributions
		mean_combined = distractor_hamming_mean - match_hamming_mean
		stdev_combined = math.sqrt(distractor_hamming_stdev**2 + match_hamming_stdev**2)
		prob_negative = norm.cdf(0.0, loc=mean_combined, scale=stdev_combined)
		return prob_negative

	def dplen(self, mm, per):
		# calculate vector length requred to store bundle at per accuracy
		# mm - mean of match distribution (single bit error rate, 0< mm < 0.5)
		# per - desired probability of error on recall (e.g. 0.000001)
		n = (-2*(-0.25 - mm + mm**2)*special.erfinv(-1 + 2*per)**2)/(0.5 - mm)**2
		return round(n)

	def bunlen(self, k, per):
		# calculated bundle length needed to store k items with accuracy per
		return self.dplen(0.5 - 0.4 / math.sqrt(k - 0.44), per)

	def gallen(self, s, per):
		# calculate required length using formula in Gallent paper
		return round(2*(-1 + 2*s)*special.erfcinv(2*per)**2)

	def test_dplen_function_orig(self):
		kvals = [5, 10, 20, 50, 100, 250, 500, 750, 1000, 2000, 3000]
		print("Bundle length for per and number of items (k):")
		print("per\t%s" % "\t".join(map(str,kvals)))
		for ex in range(3, 8):
			per = 10**(-ex)
			bundle_lengths = [self.bunlen(k, per) for k in kvals]
			print("%s\t%s" % (per, "\t".join(map(str,bundle_lengths))))
		sys.exit("Aborting.")

	def test_dplen_function_txt(self):
		kvals = [5, 10, 20, 50, 100, 250, 500, 750, 1000, 2000, 3000]
		call_gallen = self.env.pvals["values_to_plot"] == "g"
		method_msg = "gallen" if call_gallen else "bunlen"
		method = self.gallen if call_gallen else self.bunlen
		print("Bundle length for per and number of items (k), using method '%s':" % method_msg)
		print("per\t%s" % "\t".join(map(str,kvals)))
		for ex in range(3, 8):
			per = 10**(-ex)
			# bundle_lengths = [self.bunlen(k, per) for k in kvals]
			bundle_lengths = [method(k, per) for k in kvals]
			print("%s\t%s" % (per, "\t".join(map(str,bundle_lengths))))
		sys.exit("Aborting.")

	def test_dplen_function(self):
		# test function that returns length of vector needed for particular error rate
		kvals = [5, 11, 21, 51, 100, 250, 500, 750, 1000, 2000, 3000]
		perrs = [20, 10, 5, 2.5, 1]  # percent error to try
		# compute all bundle lengths
		bundle_lengths = {}
		for per in perrs:
			bundle_lengths[per] = [self.bunlen(k, per/100.0) for k in kvals]
		# create address, data and distractor arrays (long random integers)
		# make sure do at least 200 trials for each k. 
		num_trials = self.env.pvals["num_trials"]
		if num_trials < 200:
			num_trials = 200
		longest_length = bundle_lengths[perrs[-1]][-1]
		addr_base = xmpz(random.getrandbits(num_trials + longest_length))
		data_base = xmpz(random.getrandbits(num_trials + longest_length))
		faux_base = xmpz(random.getrandbits(num_trials + longest_length))
		# now loop through each option, storing and recalling and calculating stats
		stats={}
		debug = self.env.pvals["debug"]
		for per in perrs:
			print("per=%s" % per)
			stats[per] = []
			for ik in range(len(kvals)):
				bl = bundle_lengths[per][ik]
				fmt = "0%sb" % bl
				info = {"k":kvals[ik], "bl":bl, "ntrials":0, "nfail":0}
				match_hammings = []
				distractor_hammings = []
				ibase = 0  # index to starting item in base arrays
				ibend = num_trials
				while info["ntrials"] < num_trials:
					# create bundle
					bun = sdm.Bundle(bl)
					# store items
					for i in range(kvals[ik]):
						addr = addr_base[ibase+i:ibase+i+bl]
						# data = data_base[ibase+i:ibase+i+bl]
						data = data_base[ibend-i:ibend-i+bl]
						if i == 1 and debug:
							print("storing:")
							print("addr=%s" % format(addr, fmt))
							print("data=%s" % format(data, fmt))
						bun.bind_store(addr, data)
					# recall items
					trace = bun.binarize()
					if debug:
						print("bl=%s, trace=%s" % (bl, format(trace, fmt)))
					for i in range(kvals[ik]):
						addr = addr_base[ibase+i:ibase+i+bl]
						data = data_base[ibend-i:ibend-i+bl]
						# data = data_base[ibase+i:ibase+i+bl]
						recalled_data = bun.bind_recall(addr)
						hamming_match = self.int_hamming(data, recalled_data)
						match_hammings.append(hamming_match)
						distractor = faux_base[ibase+i:ibase+i+bl]
						hamming_distractor = self.int_hamming(distractor, recalled_data)
						distractor_hammings.append(hamming_distractor)
						if i == 1 and debug:
							print("recalling:")
							print("addr=%s" % format(addr, fmt))
							print("data=%s" % format(data, fmt))
							print("recd=%s" % format(recalled_data, fmt))
							print("faux=%s" % format(distractor, fmt))
							print("bl=%s, hamming match=%s, hamming_distractor=%s" %(bl, hamming_match, hamming_distractor))
							import pdb; pdb.set_trace()				
						info["ntrials"] += 1
						if hamming_distractor <= hamming_match:
							info["nfail"] += 0.5 if hamming_distractor == hamming_match else 1
						if info["ntrials"] >= num_trials:
							break
					ibase += kvals[ik]
					ibend -= kvals[ik]
				match_hamming_mean = statistics.mean(match_hammings) / bl
				match_hamming_stdev = statistics.stdev(match_hammings) / bl
				distractor_hamming_mean = statistics.mean(distractor_hammings) / bl
				distractor_hamming_stdev = statistics.stdev(distractor_hammings) / bl
				predicted_match_mean = 0.5 - 0.4 / math.sqrt(kvals[ik] - 0.44)
				predicted_match_stdev = math.sqrt(predicted_match_mean * (1-predicted_match_mean)/bl)
				predicted_distractor_mean = 0.5
				predicted_distractor_stdev = math.sqrt(0.5 * (1-0.5)/bl)
				predicted_fail_count = self.prob_negative_after_subtract(predicted_distractor_mean,
					predicted_distractor_stdev, predicted_match_mean, predicted_match_stdev)
				predicted_fail_count_stdev = math.sqrt(predicted_fail_count * (1-predicted_fail_count)/ info["ntrials"])
				expected_fail_count = self.prob_negative_after_subtract(distractor_hamming_mean,
					distractor_hamming_stdev, match_hamming_mean, match_hamming_stdev)
				info.update({"theoretical_fail_count": predicted_fail_count,
					"expected_fail_count": expected_fail_count,
					"predicted_fail_count_stdev": predicted_fail_count_stdev,
					# "match_hamming_mean": match_hamming_mean,
					# "predicted_match_mean":predicted_match_mean,
					# "match_hamming_stdev":match_hamming_stdev,
					# "predicted_match_stdev":predicted_match_stdev,
					# "distractor_hamming_mean": distractor_hamming_mean, "distractor_hamming_stdev":distractor_hamming_stdev
					})
				stats[per].append(info)
		print("computed stats are:")
		pp.pprint(stats)
		# make plots
		for per in perrs:
			xvals = range(len(kvals))
			xaxis_labels = ["%s/%s" % (kvals[i], bundle_lengths[per][i]) for i in range(len(kvals))]
			fail_percent = [(stats[per][i]["nfail"]*100.0/ stats[per][i]["ntrials"]) for i in range(len(kvals))]
			expected_per = [(stats[per][i]["expected_fail_count"]*100.0) for i in range(len(kvals))]
			theory_per = [(stats[per][i]["theoretical_fail_count"]*100.0) for i in range(len(kvals))]
			ebar_per = [(stats[per][i]["predicted_fail_count_stdev"]*50.0) for i in range(len(kvals))]
			title = "Found percent error when bundle length set for %s %%" % per
			fig = Figure(title=title, xvals=xvals, grid=True,
				xlabel="Num items / bundle length", ylabel="recall error (percent)", xaxis_labels=xaxis_labels,
				legend_location="upper left",
				yvals=fail_percent, ebar=None, legend="recall error", logyscale=False)
			fig.add_line(expected_per, legend="Expected error (from observed hammings)")
			fig.add_line(theory_per, ebar=ebar_per, legend="Theoretical error")
			self.figures.append(fig)
		#sys.exit("Aborting.")

	def show_petti_mean_function(self):
		# display both approximation and actual value for pentti's mean function
		kvals = [5, 11, 21, 51, 100, 250, 500, 750, 1000, 2000, 3000] 
		xvals = range(len(kvals))
		pentti_approx = []
		exact_value = []
		diff = []
		for k in kvals:
			pentti_approx.append(0.5 - 0.4 / math.sqrt(k - 0.44))
			exact_value.append(0.5 - math.comb((k-1), int((k-1)/2))/2**k)
			diff.append(pentti_approx[-1] - exact_value[-1])
		title = "Pentti approximation vs exact formula for mean"
		fig = Figure(title=title, xvals=xvals, grid=True,
			xlabel="Num items", ylabel="mean hamming distance", xaxis_labels=kvals,
			legend_location="upper left",
			yvals=pentti_approx, ebar=None, legend="pentti_approx", logyscale=False)
		fig.add_line(exact_value, legend="Exact value from combination")
		self.figures.append(fig)
		fig = Figure(title="pentti_approx - exact", xvals=xvals, grid=True,
			xlabel="Num items", ylabel="difference", xaxis_labels=kvals,
			legend_location="upper left",
			yvals=diff, ebar=None, legend="pentti_approx - exact", logyscale=False)
		self.figures.append(fig)


def main():
	env = Env()
	cal = Calculator(env)
	cal.show_plots()


main()
