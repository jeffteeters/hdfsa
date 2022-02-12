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
import sdm as sd
from scipy.stats import norm
import scipy.integrate as integrate
import scipy
import capacity_calc_routines as cc
import find_sdm_size as fs
import pprint
pp = pprint.PrettyPrinter(indent=4)
import overlap as ov

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
			"l-loop variable, e-prob error, d-dplen, g-gallant, p-pentti mean, f-frady vs gallant"
			", sdm-ber, round_trip","type":str,
			"choices":["l", "e", "d", "g", "p", "f", "s", "r"]}, "flag":"p", "required_init":"", "default":"l"},
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
			yvals=None, ebar=None, legend=None, fmt="-", logyscale=False): #  fmt="0-"
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
		elif values_to_plot == "s":
			self.show_sdm_ber()
		elif values_to_plot == "f":
			self.show_frady_vs_gallant_error()
		elif values_to_plot == "r":
			self.show_round_trip_theory()
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
		frady_error = self.compute_frady_recall_error()
		fig.add_line(frady_error, legend="Frady error")
		gallant_error = self.compute_gallant_recall_error()
		fig.add_line(gallant_error, legend="Gallant error")
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

	# Calculate analytical accuracy of the encoding according to the equation for p_corr from 2017 IEEE Tran paper
	def  p_corr (self, N, D, dp_hit):
		dp_rej=0.5
		var_hit = 0.25*N
		var_rej=var_hit
		range_var=10 # number of std to take into account
		fun = lambda u: (1/(np.sqrt(2*np.pi*var_hit)))*np.exp(-((u-N*(dp_rej-dp_hit) )**2)/(2*(var_hit)))*((norm.cdf((u/np.sqrt(var_rej))))**(D-1) ) # analytical equation to calculate the accuracy  

		acc = integrate.quad(fun, (-range_var)*np.sqrt(var_hit), N*dp_rej+range_var*np.sqrt(var_hit)) # integrate over a range of possible values
		# print("p_corr, N=%s, D=%s, dp_hit=%s, return=%s" % (N, D, dp_hit, acc[0]))
		return acc[0]

	def p_corr_fraction(self, N, D, dp_hit):
		# compute analytical accuracy using equation in Frady paper implemented using Fractions
		wl = N  # word length
		codebook_length = D
		probability_single_bit_failure = dp_hit
		delta = probability_single_bit_failure
		num_distractors = codebook_length - 1
		pflip = Fraction(int(delta*100000), 100000)  # probability of bit flip
		pdist = Fraction(1, 2) # probability of distractor bit matching
		match_hamming = []
		distractor_hamming = []
		for k in range(wl):
			ncomb = Fraction(math.comb(wl, k))
			match_hamming.append(ncomb * pflip ** k * (1-pflip) ** (wl-k))
			distractor_hamming.append(ncomb * pdist ** k * (1-pdist) ** (wl-k))
		dhg = Fraction(1) # fraction distractor hamming greater than match hamming
		pcor = Fraction(0)  # probability correct
		for k in range(wl):
			dhg -= distractor_hamming[k]
			pcor += match_hamming[k] * dhg ** num_distractors
		return float(pcor)

	# error = (float((1 - pcor) * 100))  # convert to percent
	# return error


	def compute_frady_recall_error(self):
		word_length = self.env.pvals["word_length"]
		codebook_length = self.env.pvals["num_items"]
		frady_error = []
		for xval in self.xvals:
			pflip = xval / 100
			# expected_hamming = pflip * word_length
			frady_error.append((1 - self.p_corr(word_length, codebook_length, pflip))*100.0) # convert to percent
		return frady_error

	def compute_gallant_recall_error(self):
		word_length = self.env.pvals["word_length"]
		codebook_length = self.env.pvals["num_items"]
		gallant_error = []
		for xval in self.xvals:
			pflip = xval / 100
			match_mean = pflip
			match_stdev = math.sqrt(match_mean * (1-match_mean)/word_length)
			distractor_mean = 0.5
			distractor_stdev = math.sqrt(distractor_mean *(1-distractor_mean)/word_length)
			combined_mean = distractor_mean - match_mean
			combined_stdev = math.sqrt(match_stdev**2 + distractor_stdev**2)
			perror1 = norm.cdf(0.0, loc=combined_mean, scale=combined_stdev)
			pcorrect_1 = 1.0 - perror1
			pcorrect_n = pcorrect_1 ** (codebook_length-1)
			gallant_error.append((1-pcorrect_n)*100.0)
		return gallant_error


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

	def gallan(self, s, per):
		# calculate required length using formula in Gallant paper
		return round(2*(-1 + 2*s)*special.erfcinv(2*per)**2)

	def bunlenf(self, k, perf, n):
		# bundle length from final probabability error (perf) taking
		# into account number of items in bundle (k) and number of other items (n)
		per = perf/(k*n)
		return self.bunlen(k, per)

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
		call_gallan = self.env.pvals["values_to_plot"] == "g"
		method_msg = "gallan" if call_gallan else "bunlen"
		method = self.gallan if call_gallan else self.bunlen
		print("Bundle length for per and number of items (k), using method '%s':" % method_msg)
		print("per\t%s" % "\t".join(map(str,kvals)))
		for ex in range(3, 8):
			per = 10**(-ex)
			# bundle_lengths = [self.bunlen(k, per) for k in kvals]
			bundle_lengths = [method(k, per) for k in kvals]
			print("%s\t%s" % (per, "\t".join(map(str,bundle_lengths))))
		sys.exit("Aborting.")

	def create_binary_matrix(self, num_items, word_length):
		bm = []
		for i in range(num_items):
			bm.append(xmpz(random.getrandbits(word_length)))
		return bm

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
		# addr_base = xmpz(random.getrandbits(num_trials + longest_length))
		# data_base = xmpz(random.getrandbits(num_trials + longest_length))
		# faux_base = xmpz(random.getrandbits(num_trials + longest_length))
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
				# ibend = num_trials
				while info["ntrials"] < num_trials:
					# create bundle and addresses and data to store
					bun = sd.Bundle(bl)
					addr_base = self.create_binary_matrix(kvals[ik], bl)
					data_base = self.create_binary_matrix(kvals[ik], bl)
					# store items
					for i in range(kvals[ik]):
						addr = addr_base[i]
						data = data_base[i]
						# addr = addr_base[ibase+i:ibase+i+bl]
						# data = data_base[ibase+i:ibase+i+bl]
						# data = data_base[ibend+i][0:ibend-i+bl]
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
						addr = addr_base[i]
						data = data_base[i]
						# addr = addr_base[ibase+i:ibase+i+bl]
						# data = data_base[ibend-i:ibend-i+bl]
						# data = data_base[ibase+i:ibase+i+bl]
						recalled_data = bun.bind_recall(addr)
						hamming_match = self.int_hamming(data, recalled_data)
						match_hammings.append(hamming_match)
						assert ibase+i+bl <= num_trials + longest_length
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
					# ibend -= kvals[ik]
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
			ebar_per = [(stats[per][i]["predicted_fail_count_stdev"]*100.0) for i in range(len(kvals))]
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

	def show_sdm_ber(self):
		# display bit error rate (delta) for sdm number of rows and k, compare
		# empirical to theory
		kvals = [5, 11, 21, 51, ]# 101, 250] # 500, 750, 1000, 2000, 3000]  # [20, 100] 5, 11, 21, 51, 101, 251] # 
		xvals = range(len(kvals))
		codebook_sizes = [36, 100, 200 ] # , 200, 500, 1000] # [1000]
		desired_percent_errors = [10,  1, 0.1] # , .1, .01, .001]
		for desired_percent_error in desired_percent_errors:
			for codebook_size in codebook_sizes:
				# found info
				fi = {"empirical":[],
					"theory_new":[],
					"theory_orig":[],
					"sdm_dimensions":[],
					"theory_hyp":[],
				}
				for k in kvals:
					perf = desired_percent_error / 100
					# per_all = 1 - math.exp(math.log(1-perf)/(k*(codebook_size-1))) # per_exact
					# delta = self.expectedHamming(k)  # fails when k is large
					delta = 0.5 - 0.4 / math.sqrt(k - 0.44)
					per1 = 1 - math.exp(math.log(1-perf)/(codebook_size - 1))  # error when recalling one
					sdm_size = fs.find_optimal_sdm_2d_search(k, per1, codebook_size)
					num_rows = sdm_size[0]
					nact = sdm_size[2]
					fi["empirical"].append(ov.sdm_delta_empirical(num_rows, k, nact))
					ber_new, nact_new = fs.single_bit_error_rate_new(num_rows, k)
					assert nact_new == nact
					fi["theory_new"].append(ber_new)
					ber_orig, nact_orig = fs.single_bit_error_rate_poisson(num_rows, k)
					assert nact_orig == nact
					fi["theory_orig"].append(ber_orig)
					ber_hyp, nact_hyp = fs.sber_hypergeometric(num_rows, k)
					assert nact_hyp == nact
					fi["theory_hyp"].append(ber_hyp)
					fi["sdm_dimensions"].append(sdm_size[0:3])  # ((m, n, nact, size))
					# print("Desired_percent_error=%s, codebook_size=%s, k=%s: info=%s, sdm_info=%s, "
					# 		"frady_error=%s, sim_error=%s, sdm_error=%s, AccEmp=%s" % (
					#  		desired_percent_error, codebook_size, k, info, sdm_info, fi["frady_error"][-1],
					#  		fi["sim_error"][-1], fi["sdm_error"][-1], fi["emp_error"][-1]))
				title = "SDM bit error rate, found vs predicted when desired error=%s%% codebook size=%s" % (desired_percent_error, codebook_size)
				xaxis_labels = ["%s/%s,%s,%s" % (kvals[i], fi["sdm_dimensions"][i][0],
					fi["sdm_dimensions"][i][1], fi["sdm_dimensions"][i][2]) for i in range(len(kvals))]
				# print("xaxis_labels=%s" % xaxis_labels)
				# print("full_fi:")
				# pp.pprint(fi)
				fig = Figure(title=title, xvals=xvals, grid=True,
					xlabel="Num items / sdm rows, cols, nact", ylabel="single bit error rate", xaxis_labels=xaxis_labels,
					legend_location="upper right",
					yvals=fi["empirical"], ebar=None, legend="overlap empirical", logyscale=False)
				fig.add_line(fi["theory_new"], legend="theory full t")
				fig.add_line(fi["theory_orig"], legend="theory poisson")
				fig.add_line(fi["theory_hyp"], legend="theory hypergeometric")
				self.figures.append(fig)
				# print("===============")
				# print(title)
				# print("%s == %s" % (xlabel, xaxis_labels))
				# print("found: %s" % fi["sdm_delta_found"])
				# print("predicted: %s" % fi["sdm_delta_predicted"])
				# print("full_fi:")
				# pp.pprint(fi)

	def perror_from_delta(self, delta, width, codebook_size):
		# width is number of components in vector
		mc = 0.5 - delta
		sc = math.sqrt((delta*(1-delta)/width) + (0.5*(1-0.5)/width))
		perror = norm.cdf(0, loc=mc, scale=sc)
		pcorr = (1-perror)**(codebook_size - 1)
		perror = 1 - pcorr
		return perror

	def show_round_trip_theory(self):
		# display theoretical recall error for created sdm and bundle
		# kvals = [5, 11, 21, 51, 101, 250, 500, ] # 750, 1000, 2000, 3000]  # [20, 100] 5, 11, 21, 51, 101, 251] # 
		kvals = range(5, 101) # range(75, 91) # 
		xvals = kvals # range(len(kvals))
		codebook_sizes = [27,] # 200 ] # , 200, 500, 1000] # [1000]
		desired_percent_errors = [1.0] # , .1, .01, .001]
		for desired_percent_error in desired_percent_errors:
			for codebook_size in codebook_sizes:
				# found info
				fi = {
					"sdm_dimensions":[],
					"expected_sdm_delta":[],
					"sdm_error_predicted":[],
					"sdm_error_quick_predicted":[],
					"bundle_length":[],
					"bundle_error_predicted":[],
					"bundle_error_quick_predicted":[],
					"sdm_error_predicted_using_hypd+fraction":[],
				}
				for k in kvals:
					perf = desired_percent_error / 100
					# per_all = 1 - math.exp(math.log(1-perf)/(k*(codebook_size-1))) # per_exact
					# delta = self.expectedHamming(k)  # fails when k is large
					per1 = 1 - math.exp(math.log(1-perf)/(codebook_size - 1))  # error when recalling one
					delta = 0.5 - 0.4 / math.sqrt(k - 0.44)
					bundle_length = self.dplen(delta, per1)
					fi["bundle_length"].append(bundle_length)
					frady_pcorrect1 = self.p_corr(bundle_length, codebook_size, delta)
					fi["bundle_error_predicted"].append((1- frady_pcorrect1) * 100.0)
					fi["bundle_error_quick_predicted"].append(self.perror_from_delta(delta, bundle_length, codebook_size) * 100.0)
					sdm_size = fs.find_optimal_sdm_2d_search(k, per1, codebook_size)
					num_rows = sdm_size[0]
					sdm_ncols = sdm_size[1]
					nact = sdm_size[2]
					ber_hyp, nact_hyp = fs.sber_hypergeometric(num_rows, k)
					assert nact_hyp == nact
					# fi["sdm_error_predicted"].append((1-self.p_corr(sdm_ncols, codebook_size, ber_hyp)) * 100.0)
					fi["expected_sdm_delta"].append(ber_hyp)
					fi["sdm_error_quick_predicted"].append(self.perror_from_delta(ber_hyp, sdm_ncols, codebook_size) * 100.0)
					fi["sdm_error_predicted_using_hypd+fraction"].append((1-self.p_corr_fraction(sdm_ncols, codebook_size, ber_hyp))*100.0)
					fi["sdm_dimensions"].append(sdm_size[0:3])  # ((m, n, nact, size))
					# print("Desired_percent_error=%s, codebook_size=%s, k=%s: info=%s, sdm_info=%s, "
					# 		"frady_error=%s, sim_error=%s, sdm_error=%s, AccEmp=%s" % (
					#  		desired_percent_error, codebook_size, k, info, sdm_info, fi["frady_error"][-1],
					#  		fi["sim_error"][-1], fi["sdm_error"][-1], fi["emp_error"][-1]))
				title = "Perdicted SDM recall error rate; desired error=%s%% codebook size=%s" % (desired_percent_error, codebook_size)
				xaxis_labels = ["%s/%s,%s,%s" % (kvals[i], fi["sdm_dimensions"][i][0],
					fi["sdm_dimensions"][i][1], fi["sdm_dimensions"][i][2]) for i in range(len(kvals))]
				# print("xaxis_labels=%s" % xaxis_labels)
				# print("full_fi:")
				# pp.pprint(fi)
				fig = Figure(title=title, xvals=xvals, grid=True,
					xlabel="Num items / sdm rows, cols, nact", ylabel="Predicted recall error rate",
					xaxis_labels=xaxis_labels,
					legend_location="upper right", fmt="-",
					yvals=fi["sdm_error_quick_predicted"], ebar=None, legend="sdm quick prediction", logyscale=False)
				fig.add_line(fi["sdm_error_predicted_using_hypd+fraction"], legend="sdm_error_predicted_using_hypd+fraction", fmt="-")
				# fig.add_line(fi["sdm_error_predicted"], legend="predicted sdm recall error rate", fmt="-")
				fig.add_line(fi["bundle_error_predicted"], legend="bundle Frady prediction")
				fig.add_line(fi["bundle_error_quick_predicted"], legend="bundle quick prediction")
				# fig.add_line(fi["theory_orig"], legend="theory poisson")
				# fig.add_line(fi["theory_hyp"], legend="theory hypergeometric")
				self.figures.append(fig)
				# print("===============")
				# print(title)
				# print("%s == %s" % (xlabel, xaxis_labels))
				# print("found: %s" % fi["sdm_delta_found"])
				# print("predicted: %s" % fi["sdm_delta_predicted"])
				# print("full_fi:")
				# pp.pprint(fi)


	def expectedHamming(self, K):
		if (K % 2) == 0: # If even number then break ties so add 1
			K+=1    
		deltaHam = 0.5 - (scipy.special.binom(K-1, 0.5*(K-1)))/2**K  # Pentti's formula for the expected Hamming distance
		return deltaHam

	def perform_word_recall(self, word_length, codebook_size, ber, debug=True):
		# find error rate for recalling a word with bit error rate ber
		# this used to verify that analytical pcorr using fractions is working
		num_trials = self.env.pvals["num_trials"]
		if num_trials < 1000:
			num_trials = 1000
		codebook = [xmpz(random.getrandbits(word_length)) for i in range(codebook_size) ]
		# info = {"ntrials":0, "nfail":0}
		trial_count = 0
		fail_count = 0
		while trial_count < num_trials:
			for cb_idx in range(codebook_size):
				codeword = codebook[cb_idx].copy()
				# add noise
				btof = list(np.where(np.random.random_sample((word_length,))<ber)[0])  # bits to flip to add noise
				for bidx in btof:
					codeword[bidx] = 1 - codeword[bidx]
				minimum_hamming_found = self.int_hamming(codeword, codebook[0])
				idx_min = 0
				for did in range(1, codebook_size):
					hamming = self.int_hamming(codeword, codebook[did])
					if hamming < minimum_hamming_found:
						minimum_hamming_found = hamming
						idx_min = did
				if idx_min != cb_idx:
					fail_count += 1
				trial_count += 1
				if trial_count >= num_trials:
					break
		return fail_count / trial_count


	def perform_bundle_or_sdm_recall(self, bundle_length, k, codebook_size, sdm_size=None,
			sliced_data=False, hl_selection_method="hamming", show_histogram=False):
		# recall either using bundle or sdm.  If sdm_size is not null, it is: (m, n, nact, size)
		# where m=number rows in sdm, n=number of columns in sdm, size is number bytes (can be ignored)
		# return error_count, total_trials
		# k is the number of vectors to add to the bundle
		# sliced_data == True, stores all data in single integer, individual data accessed by slice (saves memory)
		#    == False, each data separate random integer
		# sdm_address_method == 1 to use random hard locations for each address, 0 for hamming distance match (normally done)
		# This used to determin if hamming distance method is causing problems in recall, maybe because some random
		# addresses are too similar to each other
		debug = self.env.pvals["debug"]
		print("entered perform_bundle_or_sdm_recall(bundle_length=%s, k=%s, codebook_size=%s, sdm_size=%s)" %
				(bundle_length, k, codebook_size, sdm_size))
		using_sdm = sdm_size is not None
		if using_sdm:
			assert bundle_length == sdm_size[1], "if using sdm, bundle_length must equal number of columns in sdm"
		bl = bundle_length
		num_trials = self.env.pvals["num_trials"]
		if num_trials < 1000:
			num_trials = 1000
		fmt = "0%sb" % bl
		match_hammings = []
		distractor_hammings = []
		min_distractor_hammings = []
		info = {"ntrials":0, "nfail":0}	
		while info["ntrials"] < num_trials:
			# create bundle and addresses and data to store
			if using_sdm:
				# use sdm to store data
				num_rows, num_cols, nact = sdm_size[0:3]
				assert bl == num_cols
				# nact = round(fs.fraction_rows_activated(num_rows, k)*num_rows)
				if info["ntrials"] == 0:
					# display activation count
					print("k=%s, code_book=%s, sdm num_rows=%s, ncols=%s, nact=%s" % (k,
						codebook_size, num_rows, num_cols, nact))
				sdm = sd.Sdm(bl, bl, num_rows, nact=nact, hl_selection_method=hl_selection_method)
			else:
				# use bundle to store data
				bun = sd.Bundle(bl)
			addr_base_length = k + bundle_length - 1
			addr_base = xmpz(random.getrandbits(addr_base_length))
			data_base_length = codebook_size + bundle_length - 1
			data_base = xmpz(random.getrandbits(data_base_length)) if sliced_data else [xmpz(random.getrandbits(bl))
				for i in range(codebook_size)]
			# data_base = xmpz(random.getrandbits(base_length))
			exSeq= np.random.randint(low = 0, high = codebook_size, size =k) # radnom sequence to represent
			if debug==1:
				# base_length = codebook_size + bundle_length - 1
				addr_base_fmt = "0%sb" % addr_base_length
				print("bl=%s, codebook_size=%s, base_length=%s, k=%s" % (bl, codebook_size, base_length, k))
				print("addr_base=%s" % format(addr_base, addr_base_fmt))
				if sliced_data:
					data_base_fmt = "0%sb" % data_base_length
					print("data_base=%s" % format(data_base, data_base_fmt))
				else:
					print("codebook=")
					for i in range(codebook_size):
						print("%i - %s" %(i, format(data_base[i], fmt)))
				print("exSeq=%s" % exSeq)
			# data_base = self.create_binary_matrix(k, bl)
			# store items
			# check for multiple addresses the same
			addr_cache = {}
			for i in range(k):
				assert i+bl <= addr_base_length, "i=%s, bl=%s, i+bl=%s not less than addr_base_length=%s" % (
					i, bl, i+bl, addr_base_length)
				addr = addr_base[i:i+bl]
				if sliced_data:
					assert exSeq[i] >= 0 and exSeq[i] <= data_base_length and exSeq[i]+bl <= data_base_length, (
						"data slice out of range: i=%s, exSeq[i]=%s, bl=%s, exSeq[i]+bl=%s, data_base_length=%s, k=%s" % (i,
							exSeq[i], bl, exSeq[i]+bl, data_base_length, k))
				if addr not in addr_cache:
					addr_cache[addr] = [i]
				else:
					addr_cache[addr].append(i)
				data = data_base[exSeq[i]:exSeq[i]+bl] if sliced_data else data_base[exSeq[i]]
				if debug:  # i == k-1 and 
					print("storing:")
					print("exSeq=%s" % exSeq)
					print("i=%s, exSeq[i]=%s, k=%s" % (i, exSeq[i], k))
					print("addr=%s" % format(addr, fmt))
					print("data=%s" % format(data, fmt))
				if using_sdm:
					sdm.bind_store(addr, data)
				else:
					bun.bind_store(addr, data)
			# recall items
			# check for addresses used more than once
			for addr, kids in addr_cache.items():
				multiple_kids = 0
				if len(kids) > 1:
					print("%s k-> %s" % (format(addr, fmt), kids))
					multiple_kids += 1
			if multiple_kids > 0:
				sys.exit("found %s addresses with multiple k's" % multiple_kids)
			if not using_sdm:
				trace = bun.binarize()
			else:
				if False and show_histogram and info["ntrials"] < 2:
					num_addresses = len(sdm.Hls.row_cache) if hl_selection_method=="random_locations" else "unknown" 
					print("ntrials=%s, num_addresses=%s" % (info["ntrials"], num_addresses))
					# for col in range(10):
					for row in range(num_rows):
						# show histogram of sdm counters
						# sdm_counters = sdm.data_array[:,col]
						sdm_counters = sdm.data_array[row,:]
						# assert len(sdm_counters) == num_rows
						ov.plot_hist(sdm_counters, nact, k, row)
					# if info["ntrials"] > 2000:
					# 	sys.exit("done for now")
			for i in range(k):
				addr = addr_base[i:i+bl]
				data = data_base[exSeq[i]:exSeq[i]+bl] if sliced_data else data_base[exSeq[i]]
				if using_sdm:
					recalled_data = sdm.bind_recall(addr)
				else:
					recalled_data = bun.bind_recall(addr)
				match_hamming = self.int_hamming(data, recalled_data)
				match_hammings.append(match_hamming)
				min_distractor_hamming = bl
				for did in range(codebook_size):
					if did == exSeq[i]:
						continue	# this is the stored item, not a distractor
					distractor = data_base[did:did+bl] if sliced_data else data_base[did]
					distractor_hamming = self.int_hamming(distractor, recalled_data)
					distractor_hammings.append(distractor_hamming)
					if distractor_hamming < min_distractor_hamming:
						min_distractor_hamming = distractor_hamming
						min_distractor_hammind_idx = did
				min_distractor_hammings.append(min_distractor_hamming)
				if debug:
					print("recalling:")
					print("addr=%s" % format(addr, fmt))
					print("data=%s" % format(data, fmt))
					print("recd=%s" % format(recalled_data, fmt))
					# print("faux=%s" % format(distractor, fmt))
					print("bl=%s, match hamming =%s, min_distractor_hamming=%s" %(bl, match_hamming, min_distractor_hamming))
					import pdb; pdb.set_trace()				
				info["ntrials"] += 1
				if min_distractor_hamming <= match_hamming:
					if min_distractor_hamming < match_hamming or min_distractor_hammind_idx < exSeq[i]:
						# count error if distractor hamming less, or distractor with matching hamming appears first
						info["nfail"] += 1
						# 0.5 if min_distractor_hamming == match_hamming else 1
					# print("found fail: bundle_length=%s, k=%s, codebook_size=%s, info=%s" % (bundle_length,
					# 	k, codebook_size, info ))
				if info["ntrials"] >= num_trials:
					break
		info["match_hamming_mean"] = statistics.mean(match_hammings)
		if show_histogram:
			method = "sdm" if using_sdm else "bundle"
			title = "match hammings for %s, k=%s" % (method, k)
			nbins = self.get_nbins(match_hammings)
			fig = Figure(ptype="hist", nbins=nbins,
						title=title, xvals=match_hammings, grid=True,
						xlabel="hamming distance", ylabel="count", xaxis_labels=None,
						legend_location="upper left",
						yvals=None, ebar=None, legend="match hamming")
			self.figures.append(fig)
		info["match_hamming_stdev"] = statistics.stdev(match_hammings)
		info["distractor_hamming_mean"] = statistics.mean(distractor_hammings)
		info["distractor_hamming_stdev"] = statistics.stdev(distractor_hammings)
		info["normalized_hamming"] = info["match_hamming_mean"] / bl
		return info


	def show_frady_vs_gallant_error(self):
		kvals = range(5, 37) # range(255, 287) # 101) # [5, 11, 21, 51, 101] # [5, 11, 21, 51, 101, 250] #, 500, 750, 1000] # , 2000, 3000]  # [20, 100] 5, 11, 21, 51, 101, 251] # 
		xvals = range(len(kvals))
		codebook_sizes = [27 ] # , 200, 500, 1000] # [1000]
		desired_percent_errors = [1] # [10,  1, 0.1] # , .1, .01, .001]
		include_empirical = True
		show_histogram = self.env.pvals["show_histograms"]
		num_trials = self.env.pvals["num_trials"]
		for desired_percent_error in desired_percent_errors:
			for codebook_size in codebook_sizes:
				# found info
				fi = {"frady_error":[], 	# analytical frady equation, used with bundle
					"sim_error":[],	# bundle simulation
					"emp_error":[], # Denis empirical
					"bundle_error_quick_predicted":[],
					"bundle_length":[],
					"sdm_dimensions":[],
					"sdm_word_recall_error":[],
					"sdm_error":[],
					"sdm_error_predicted":[],
					"sdm_error_predicted_using_delta_found":[],
					"sdm_error_predicted_using_hypd+fraction":[],
					"sdm_error_quick_predicted":[],				
					"sdm_match_hamming_mean":[],
					"sdm_distractor_hamming_mean":[],
					"sdm_match_hamming_stdev":[],
					"sdm_distractor_hamming_stdev":[],
					"sdm_delta_found":[],
					"sdm_delta_predicted":[],
					"sdm_delta_empirical":[],  # delta from overleaf
					"sdm_error_predicted_using_delta_found_fraction":[],
					"bundle_error_predicted_using_fraction":[],
				}
				# frady_error = []
				# sim_error = []
				# emp_error = []
				# bundle_lengths = []
				# sdm_dimensions = []
				# sdm_error = []
				# sdm_delta_found = []
				# sdm_delta_predicted = []
				include_bundle = False
				include_sdm_word_recall = True
				for k in kvals:
					perf = desired_percent_error / 100
					per1 = 1 - math.exp(math.log(1-perf)/(codebook_size - 1))  # error when recalling one
					# per_all = 1 - math.exp(math.log(1-perf)/(k*(codebook_size-1))) # per_exact
					# delta = self.expectedHamming(k)  # fails when k is large
					if include_bundle:
						delta = 0.5 - 0.4 / math.sqrt(k - 0.44)
						bundle_length = self.dplen(delta, per1)
						fi["bundle_length"].append(bundle_length)
						fi["bundle_error_quick_predicted"].append(self.perror_from_delta(delta, bundle_length, codebook_size) * 100.0)
						frady_pcorrect1 = self.p_corr(bundle_length, codebook_size, delta)
						frady_pcorrect_all = frady_pcorrect1 # ** k
						fi["frady_error"].append((1-frady_pcorrect_all)*100.0)  # convert to percent error
					# bundle_lengths.append(bundle_length)
					sdm_size = fs.find_optimal_sdm_2d_search(k, per1, codebook_size)
					# sdm_size = (sdm_size[0]+1, sdm_size[1]+1, sdm_size[2], sdm_size[3])  # experiment increasing sdm size
					sdm_nrows = sdm_size[0]
					sdm_ncols = sdm_size[1]
					sdm_nact = sdm_size[2]
					fi["sdm_dimensions"].append(sdm_size)  # ((m, n, size))
					# 
					# fi["bundle_error_predicted_using_fraction"].append(1-self.p_corr_fraction(bundle_length, codebook_size, delta) * 100.0)
					# frady_error.append((1-frady_pcorrect1)*100.0)
					if include_empirical:
						# recall from SDM
						sdm_info = self.perform_bundle_or_sdm_recall(sdm_size[1], k, codebook_size, sdm_size=sdm_size,
							sliced_data=True, hl_selection_method="random_locations", show_histogram=show_histogram)
						mmean = sdm_info["match_hamming_mean"]
						mvar = sdm_info["match_hamming_stdev"]**2
						print("k=k, sdm_match_hamming_mean=%s, sdm_match_hamming_var=%s, percent diff=%s" % (
							mmean, mvar, abs(mmean- mvar)*100 / (mmean)))
						fi["sdm_error"].append(sdm_info["nfail"] * 100.0 / (sdm_info["ntrials"]))
						fi["sdm_delta_found"].append(sdm_info["normalized_hamming"])
						delta_predicted = fs.single_bit_error_rate(sdm_nrows, k)[0]
						fi["sdm_delta_predicted"].append(delta_predicted)
						# fi["sdm_error_predicted"].append((1-self.p_corr(sdm_ncols, codebook_size, delta_predicted)) * 100.0)
						fi["sdm_error_quick_predicted"].append(self.perror_from_delta(delta_predicted, sdm_ncols, codebook_size) * 100.0)
						# fi["sdm_error_predicted_using_delta_found"].append((1-self.p_corr(sdm_ncols, codebook_size, fi["sdm_delta_found"][-1])) * 100.0)
						fi["sdm_error_predicted_using_delta_found_fraction"].append((1-self.p_corr_fraction(sdm_ncols, codebook_size, fi["sdm_delta_found"][-1])) * 100.0)
						fi["sdm_error_predicted_using_hypd+fraction"].append((1-self.p_corr_fraction(sdm_ncols, codebook_size, delta_predicted))*100.0)
						fi["sdm_delta_empirical"].append(ov.sdm_delta_empirical(sdm_nrows, k, sdm_nact, num_trials=num_trials))
						fi["sdm_match_hamming_mean"].append(sdm_info["match_hamming_mean"])
						fi["sdm_distractor_hamming_mean"].append(sdm_info["distractor_hamming_mean"])
						fi["sdm_match_hamming_stdev"].append(sdm_info["match_hamming_stdev"])
						fi["sdm_distractor_hamming_stdev"].append(sdm_info["distractor_hamming_stdev"])
						if include_sdm_word_recall:
							fi["sdm_word_recall_error"].append(self.perform_word_recall(sdm_ncols, codebook_size, delta_predicted) * 100.0)
							print("k=%s, sdm_size=%s, error_found=%s, error_perdicted=%s, word_recall_error=%s" % (k, sdm_size, fi["sdm_error"][-1],
								fi["sdm_error_predicted_using_hypd+fraction"][-1], fi["sdm_word_recall_error"][-1]))
						if include_bundle:
							info = self.perform_bundle_or_sdm_recall(bundle_length, k, codebook_size, sliced_data=True) # returns: {"ntrials":0, "nfail":0}
							fi["sim_error"].append(info["nfail"] * 100.0 / (info["ntrials"]))
							fi["emp_error"].append((1-cc.AccuracyEmpirical(bundle_length,codebook_size,k)[0]) * 100.0)
						else:
							# fi["sim_error"].append(None)  # placeholder
							info = None
						# print("Desired_percent_error=%s, codebook_size=%s, k=%s: info=%s, sdm_info=%s, "
						# 	"frady_error=%s, sim_error=%s, sdm_error=%s, AccEmp=%s" % (
						# 	desired_percent_error, codebook_size, k, info, sdm_info, fi["frady_error"][-1],
					 # 		fi["sim_error"][-1], fi["sdm_error"][-1], fi["emp_error"][-1]))
				title = "Found error when desired error=%s%% codebook size=%s" % (desired_percent_error, codebook_size)
				# xaxis_labels = ["%s/%s" % (kvals[i], fi["bundle_length"][i]) for i in range(len(kvals))]
				xaxis_labels = ["%s/%s,%s,%s" % (kvals[i], fi["sdm_dimensions"][i][0],
					fi["sdm_dimensions"][i][1], fi["sdm_dimensions"][i][2]) for i in range(len(kvals))]
				# fig = Figure(title=title, xvals=xvals, grid=True,
				# 	xlabel="Num items / bundle length", ylabel="recall error (percent)", xaxis_labels=xaxis_labels,
				# 	legend_location="upper right",
				# 	yvals=fi["frady_error"], ebar=None, legend="bundle predicted (Frady)", logyscale=False)
				fig = Figure(title=title, xvals=xvals, grid=True,
					xlabel="Num items / nrows, ncols, nact", ylabel="recall error (percent)", xaxis_labels=xaxis_labels,
					legend_location="upper right",
					yvals=fi["sdm_error"], ebar=None, legend="sdm empirical", logyscale=False)
				if include_empirical:
					# print("sdm_dimensions=%s" % fi["sdm_dimensions"])
					if include_bundle:
						fig.add_line(fi["sim_error"], legend="bundle simulation")
						# fig.add_line(fi["bundle_error_predicted_using_fraction"],  legend="bundle_err_predicted_using_fraction", fmt=".-")
						fig.add_line(fi["bundle_error_quick_predicted"], legend="Bundle quick predicted")
						fig.add_line(fi["emp_error"], legend="AccuracyEmpirical")
					# fig.add_line(fi["sdm_error"], legend="sdm empirical")
					# fig.add_line(fi["sdm_error_predicted"], legend="sdm predicted (Frady)")
					# fig.add_line(fi["sdm_error_predicted_using_delta_found"], legend="sdm predicted (Frady) w/delta found")
					fig.add_line(fi["sdm_error_predicted_using_delta_found_fraction"],legend="sdm errp (Frady) w/delta found fraction")
					# print("sdm_error_predicted_using_delta_found_fraction=%s" % fi["sdm_error_predicted_using_delta_found_fraction"])
					# print("sdm_error_predicted_using_hypd+fraction=%s" % fi["sdm_error_predicted_using_hypd+fraction"])	
					fig.add_line(fi["sdm_error_predicted_using_hypd+fraction"],legend="sdm errp hypd+fraction")
					if include_sdm_word_recall:
						fig.add_line(fi["sdm_word_recall_error"], legend="word_recall_error with delta predicted")
					fig.add_line(fi["sdm_error_quick_predicted"], legend="sdm quick predicted")
				self.figures.append(fig)
				if include_empirical:
					title = "sdm delta found vs predicted for desired error %s%%" % desired_percent_error
					xaxis_labels = ["%s/(%s, %s)" % (kvals[i], fi["sdm_dimensions"][i][0], fi["sdm_dimensions"][i][1]) 
						for i in range(len(kvals))]
					xlabel = "Num items / sdm (nrows, cols)"
					fig = Figure(title=title, xvals=xvals, grid=True,
						xlabel=xlabel, ylabel="single bit error", xaxis_labels=xaxis_labels,
						legend_location="upper right",
						yvals=fi["sdm_delta_found"], ebar=None, legend="found", logyscale=False)
					fig.add_line(fi["sdm_delta_predicted"], legend="predicted")
					fig.add_line(fi["sdm_delta_empirical"], legend="overlap empirical")
					self.figures.append(fig)
					# print("===============")
					# print(title)
					# print("%s == %s" % (xlabel, xaxis_labels))
					# print("found: %s" % fi["sdm_delta_found"])
					# print("predicted: %s" % fi["sdm_delta_predicted"])
					# print("full_fi:")
					# pp.pprint(fi)


def main():
	env = Env()
	cal = Calculator(env)
	cal.show_plots()


main()
