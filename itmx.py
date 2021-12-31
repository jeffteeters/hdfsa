# item memory explorer

import numpy as np
import argparse
import matplotlib.pyplot as plt
import sys
import statistics
import math

class Env:
	# stores environment settings and data arrays

	# command line arguments
	parms = [
		{ "name":"num_items", "kw":{"help":"Number of items in item memory", "type":int},
	 	  "flag":"s", "required_init":"i", "default":100 },
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
		{ "name":"values_to_plot", "kw":{"help":"Values to plot.  l-loop variable, e-prob error","type":str,
			"choices":["l", "e"]}, "flag":"p", "required_init":"", "default":"l"},
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
		# 	nbins = 10
		# elif xv_range > 100:
		# 	nbins = 100
		# else:
		# 	nbins = int(xv_range) + 1
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
			# 		stdev_hamming_to_closest_distractor))
			# if self.env.pvals["debug"]:
			# 	print("hamming_to_closest_distractor=%s" % hamming_to_closest_distractor)
			# 	print("mean=%s, stdev=%s" % (mean_hamming_to_closest_distractor, stdev_hamming_to_closest_distractor))
			# 	import pdb; pdb.set_trace()
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
		# 	xlabel="bit flips (%)", ylabel="Hamming difference (percent)", xaxis_labels=None,
		# 	legend_location="lower right",
		# 	yvals=yham, ebar=yhamebar, legend="Distractor hamming diff (%)")

		# fig.add_line(yham, ebar=yhamebar, legend="Distractor hamming diff (%)")
		self.figures.append(fig)

	def compute_theoretical_recall_error(self):
		# compute theoretical recall error using equation in Frady paper (or what I think it should be)
		error = []
		wl = self.env.pvals["word_length"]
		num_distractors = self.env.pvals["num_items"] - 1
		for xval in self.xvals:
			pflip = xval / 100.0  # probability of bit flip
			pdist = 0.5  # probability of distractor bit matching
			match_hamming = []
			distractor_hamming = []
			for k in range(wl):
				match_hamming.append(math.comb(wl, k) * pflip ** k * (1.0-pflip) ** (wl-k))
				distractor_hamming.append(math.comb(wl, k) * pdist ** k * (1.0-pdist) ** (wl-k))
			# print("sum match_hamming=%s, distractor_hamming=%s" % (sum(match_hamming), sum(distractor_hamming)))
			if self.env.pvals["show_histograms"]:
				fig = Figure(title="match and distractor hamming distributions for %s%% bit flips" % xval,
					xvals=range(wl), grid=False,
					xlabel="Hamming distance", ylabel="Probability", xaxis_labels=None,
					legend_location="upper right",
					yvals=match_hamming, ebar=None, legend="match_hamming", fmt="-g")
				fig.add_line(distractor_hamming, legend="distractor_hamming", fmt="-m")
				self.figures.append(fig)
			dhg = 1.0 # fraction distractor hamming greater than match hamming
			pcor = 0.0  # probability correct
			for k in range(wl):
				dhg -= distractor_hamming[k]
				pcor += match_hamming[k] * dhg ** num_distractors
			error.append((1.0 - pcor) * 100.0)  # convert to percent
		return error


def main():
	env = Env()
	cal = Calculator(env)
	cal.show_plots()


main()
