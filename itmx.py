# item memory explorer

import numpy as np
import argparse
import matplotlib.pyplot as plt
import sys
import statistics
import math
from fractions import Fraction
from scipy import special

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
		{ "name":"values_to_plot", "kw":{"help":"Values to plot.  l-loop variable, e-prob error, e-dplen, g-gallen","type":str,
			"choices":["l", "e", "d", "g"]}, "flag":"p", "required_init":"", "default":"l"},
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
		# return hamming distance between binary vectors a and b
		return np.count_nonzero(a!=b)

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

	def dplen(self, mm, per):
		# calculate vector length requred to store bundle at per accuracy
		# mm - mean of match distribution (single bit error rate, 0< mm < 0.5)
		# per - desired probability of error on recall (e.g. 0.000001)
		n = (-2*(-0.25 - mm + mm**2)*special.erfinv(-1 + 2*per)**2)/(0.5 - mm)**2
		return round(n) + 1

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

	def test_dplen_function(self):
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

	def test_dplen_function_draft(self):
		kvals = [5, 11, 21, 51, 100, 250, 500, 750, 1000, 2000, 3000]
		print("Bundle length for per and number of items (k):")
		if self.env.pvals["loop_step"] != 1 or self.xvals[-1] > 20:
			# loop_steps is negative exponent, not specified.  Use defautt
			exps = list(range(3,8,1))
		else:
			exps = self.xvals
		# compute all bundle lengths
		bundle_lengths = {}
		for ex in exps:
			per = 10**(-ex)
			bundle_lengths[ex] = [self.bunlen(k, per) for k in kvals]
		# create item memory for largest bundle
		num_items = self.env.pvals["num_items"]
		longest_length = bundle_lengths[exps[-1]][-1]
		item_memory = np.random.randint(0, high=2, size=(num_items, longest_length), dtype=np.int8)
		item_memory[item_memory == 0] = -1
		# create counter vector (to make bundle) of longest length
		counter = np.empty(longest_length, dtype=int16)
		bundle = np.empty(longest_length, dtype=int8)
		# now loop through each option, storing and recalling and calculating stats
		stats={}
		for ex in exps:
			stats[ex] = []
			for ik in range(kvals):
				bl = bundle_lengths[ex][ik]
				info = {"bl":bl, "ncorrect":0, "nfail":0}
				# make sure do at least 1000 matches for each k
				ibase = 0  # index to starting item in item_index
				while info["ncorrect"] + info["nfail"] < 1000:
					counter[0:bl] = 0
					# store items
					for i in range(kvals[ik]):
						pass_number = int((i+ibase) / num_items)
						item_index = (i+ibase) % num_items
						counter[0:bl] += np.roll(item_memory[item_index][0:bl], pass_number)
					# threshold counter to make bundle
					bundle[counter[0:bl]>0] = 1
					bundle[counter[0:bl]<=0] = -1
					# recall items
					for i in range(kvals[ik]):
						pass_number = int((i+ibase) / num_items)
						item_index = (i+ibase) % num_items
						original_item = np.roll(item_memory[item_index][0:bl], pass_number)
						hamming_match = self.hamming(original_item, np.roll(bundle[0:bl], pass_number))
						for j in range(num_items):
							if j == item_index:
								continue
							hamming_distractor = self.hamming(item_memory[j])
						counter[0:bl] += np.roll(item_memory[item_index][0:bl], pass_number)



		sys.exit("Aborting.")


def main():
	env = Env()
	cal = Calculator(env)
	cal.show_plots()


main()
