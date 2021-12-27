# item memory explorer

import numpy as np
import argparse
import matplotlib.pyplot as plt
import sys

class Env:
	# stores environment settings and data arrays

	# command line arguments
	parms = [
		{ "name":"num_items", "kw":{"help":"Number of items in item memory", "type":int},
	 	  "flag":"s", "required_init":"i", "default":100 },
	 	{ "name":"word_length", "kw":{"help":"Word length for item memory, 0 to disable", "type":int},
	 	  "flag":"w", "required_init":"i", "default":512 },
	 	{ "name":"loop_start", "kw":{"help":"Start value for loop","type":int},
		  "flag":"x", "required_init":"", "default":0},
		{ "name":"loop_step", "kw":{"help":"Step value for loop","type":int},
		  "flag":"y", "required_init":"", "default":1},
		{ "name":"loop_stop", "kw":{"help":"Stop value for loop","type":int},
		  "flag":"z", "required_init":"", "default":10},
		{ "name":"values_to_plot", "kw":{"help":"Values to plot.  l-loop variable, e-prob error","type":str, "choices":["l", "e"]},
		  "flag":"p", "required_init":"", "default":"l"},
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

	def __init__(self, title=None, xvals=None, grid=False, xlabel=None, ylabel=None, xaxis_labels=None, legend_location=None,
			yvals=None, ebar=None, legend=None):
		self.title = title
		self.xvals = xvals
		self.grid = grid
		self.xlabel = xlabel
		self.ylabel = ylabel
		self.xaxis_labels = xaxis_labels
		self.legend_location = legend_location
		self.line_info = []
		if yvals is not None:
			self.add_line(yvals, ebar, legend)

	def add_line(self, yvals, ebar=None, legend=None):
		info = {"yvals": yvals, "ebar":ebar, "legend":legend}
		self.line_info.append(info)

	def make_plot(self, figure_number):
		fig = plt.figure(figure_number)
		plt.title(self.title)
		for info in self.line_info:
			plt.errorbar(self.xvals, info["yvals"], yerr=info["ebar"], label=info["legend"], fmt="-o")
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
		plt.show()


class Calculator:
	# Calculate values to plot

	def __init__(self, env):
		self.env = env
		self.figures = []
		values_to_plot = self.env.pvals["values_to_plot"]
		if values_to_plot == "l":
			# loop values
			self.calculate_loop_values()
			self.calculate_loop_values(0.5)
		else:
			sys.exit("values_to_plot '%s' not implemented" % values_to_plot)

	def calculate_loop_values(self, factor=0.75):
		start = self.env.pvals["loop_start"]
		step = self.env.pvals["loop_step"]
		stop = self.env.pvals["loop_stop"]
		xvals = list(range(start, stop, step))
		yvals = xvals
		fig = Figure(title="Loop variable %s" % factor, xvals=xvals, grid=True,
			xlabel="loop variable value", ylabel="loop var *factor", xaxis_labels=None,
			legend_location="lower right",
			yvals=yvals, ebar=None, legend="loop var")
		yvf = [y * factor for y in yvals]
		fig.add_line(yvf, ebar=None, legend="loop var*%s" % factor)
		self.figures.append(fig)

	def show_plots(self):
		for i in range(len(self.figures)):
			self.figures[i].make_plot(i+1)


def main():
	env = Env()
	cal = Calculator(env)
	cal.show_plots()


main()
