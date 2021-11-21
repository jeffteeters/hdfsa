# script to compare number of overlaps in simulation to practice
import argparse
import sys
import numpy as np
import math


class Env:
	# stores environment settings and data arrays

	# command line arguments
	parms = [
		{ "name":"num_states", "kw":{"help":"Number of states (items) stored in sdm", "type":int},
	 	  "flag":"s", "required_init":"i", "default":20 },
	 	{ "name":"num_rows", "kw":{"help":"Number rows in sdm memory","type":int},
	 	  "flag":"m", "required_init":"i", "default":2048 },
		{ "name":"activation_count", "kw":{"help":"Number memory rows to activate for each address (sdm)","type":int},
		  "flag":"a", "required_init":"m", "default":20},
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
		# print("Settings:")
		# for p in self.parms:
		# 	print("%s %s: %s" % (p["flag"], p["name"], self.pvals[p["name"]]))

	def get_settings(self):
		msg = "Arguments:\n"
		msg += " ".join(sys.argv)
		msg += "\nSettings:\n"
		for p in self.parms:
			msg += "%s %s: %s\n" % (p["flag"], p["name"], self.pvals[p["name"]])
		return msg

class SDM_tester():
	# test storing and recalling items from SDM

	def __init__(self, pvals):
		self.pvals = pvals
		self.sdm_counters = np.zeros(self.pvals["num_rows"], dtype=np.int32)
		# item values are single bit, 0 or 1
		self.item_values = np.random.randint(0, high=2, size=pvals["num_states"], dtype=np.int8)
		self.item_rows = np.empty((pvals["num_states"], pvals["activation_count"]), dtype=np.int32)
		for i in range(pvals["num_states"]):
			self.item_rows[i,:] = np.random.choice(pvals["num_rows"], size=pvals["activation_count"], replace=False)
		if self.pvals["debug"]:
			print("item_values\n%s" % self.item_values)
			print("item_rows\n%s" % self.item_rows)
		self.save_items()
		self.recall_items()

	def save_items(self):
		for i in range(self.pvals["num_states"]):
			self.sdm_counters[self.item_rows[i:]] += 1 if self.item_values[i] == 1 else -1
		if self.pvals["debug"]:
			print("sdm_counters\n%s" % self.sdm_counters)

	def recall_items(self):
		recalled_values = np.empty(self.pvals["num_states"], dtype=np.int8)
		for i in range(self.pvals["num_states"]):
			csum = np.sum(self.sdm_counters[self.item_rows[i:]])
			recalled_values[i] = 1 if csum >= 0 else 0
		num_correct = np.count_nonzero( recalled_values == self.item_values)
		print("num_correct = %s / %s (%s%%)" % (num_correct, self.pvals["num_states"], 
			round(num_correct * 100.0 / self.pvals["num_states"], 1)))
		if self.pvals["debug"]:
			print("stored values, recalled values=\n%s\n%s" % (self.item_values, recalled_values))


class Hit_Counter():
	# counts number of hits of each row in simulation of storing in sdm

	def __init__(self, pvals):

		self.pvals = pvals
		self.counters = np.zeros(self.pvals["num_rows"], dtype=np.int32)
		self.save_states()

	def save_states(self):
		for i in range(self.pvals["num_states"]):
			rows = np.random.choice(self.pvals["num_rows"], size=self.pvals["activation_count"], replace=False)
			for j in rows:
				self.counters[j] += 1

	def show_distributions(self):
		mean_overlap = np.mean(self.counters)
		stdev_overlap = np.std(self.counters)
		# compute predicted mean and stdev
		sdm_num_rows = self.pvals["num_rows"]
		sdm_activation_count = self.pvals["activation_count"]
		num_items_stored = self.pvals["num_states"]
		# pact = sdm_activation_count / sdm_num_rows
		predicted_mean = (sdm_activation_count * num_items_stored) / sdm_num_rows
		# assume poisson distribution, variance is same as mean, std should be square root of mean
		predicted_stdev = math.sqrt(predicted_mean)
		# predicted_stdev = math.sqrt(predicted_mean * (1 + pact * num_items_stored * (1.0 + pact*pact * sdm_num_rows)))
		print("mean overlap found: %s, predicted:%s" % (mean_overlap, predicted_mean))
		print("stdev overlap found: %s, predicted:%s"% (stdev_overlap, predicted_stdev))

def main():
	env = Env()
	# hc = Hit_Counter(env.pvals)
	# hc.show_distributions()
	t = SDM_tester(env.pvals)


main()