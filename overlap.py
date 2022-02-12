# script to compare number of overlaps in simulation to practice
import argparse
import sys
import numpy as np
import math
from scipy.stats import norm
import matplotlib.pyplot as plt


class Env:
	# stores environment settings and data arrays

	# command line arguments
	parms = [
		{ "name":"num_states", "kw":{"help":"Number of states (items) stored in sdm", "type":int},
	 	  "flag":"s", "required_init":"i", "default":1000 },
	 	{ "name":"num_rows", "kw":{"help":"Number rows in sdm memory","type":int},
	 	  "flag":"m", "required_init":"i", "default":1939 }, # 2048
		{ "name":"activation_count", "kw":{"help":"Number memory rows to activate for each address (sdm)","type":int},
		  "flag":"a", "required_init":"m", "default":19},
		{ "name":"noise_percent", "kw":{"help":"Percent of counters to flip in memory to test noise resiliency",
		  "type":float}, "flag":"n", "required_init":"m", "default":0.0},
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


def sdm_delta_empirical(num_rows, num_items_stored, activation_count, num_trials=10000, show_histogram=False):
	# emperically compute sdm single bit error due to overlap in rows activated
	trial_count = 0
	error_count = 0
	while trial_count < num_trials:
		# create new sdm counters
		sdm_counters = np.zeros(num_rows, dtype=np.int32)
		item_values = np.random.randint(0, high=2, size=num_items_stored, dtype=np.int8)
		item_rows = np.empty((num_items_stored, activation_count), dtype=np.int32)
		for i in range(num_items_stored):
			item_rows[i,:] = np.random.choice(num_rows, activation_count, replace=False)
		# store items in sdm counters
		for i in range(num_items_stored):
			sdm_counters[item_rows[i,:]] += 1 if item_values[i] == 1 else -1
		# recall values
		recalled_values = np.empty(num_items_stored, dtype=np.int8)
		for i in range(num_items_stored):
			csum = np.sum(sdm_counters[item_rows[i,:]])
			recalled_values[i] = 1 if csum > 0 else 0
		num_failed = np.count_nonzero( recalled_values != item_values)
		if show_histogram:
			plot_hist(sdm_counters, activation_count, num_items_stored)
		error_count += num_failed
		trial_count += num_items_stored
	return error_count / trial_count

def plot_hist(sdm_counters, nact, num_items_stored, row=None):
	# x = np.histogram(sdm.data_array, bins="auto")
	# the histogram of the data
	counts = sdm_counters
	nbins = get_nbins(counts)
	n, bins, patches = plt.hist(counts, nbins, density=False, facecolor='g', alpha=0.75)
	plt.xlabel('Counter value')
	plt.ylabel('Count')
	row_msg = ", row=%s" % row if row is not None else ""
	plt.title('Histogram of SDM counter values for nrows=%s, nact=%s, k=%s%s' % (
		len(sdm_counters), nact, num_items_stored, row_msg))
	# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
	# plt.xlim(40, 160)
	# plt.ylim(0, 0.03)
	plt.grid(True)
	plt.show()


def get_nbins(xv):
	# calculate number of bins to use in histogram for values in xv
	xv_range = max(xv) - min(xv)
	if xv_range == 0:
		nbins = 3
	else:
		nbins = int(xv_range) + 1
	return nbins

def fraction_rows_activated(m, k):
	# compute optimal fraction rows to activate in sdm
	# m is number rows, k is number items stored in sdm
	return 1.0 / ((2*m*k)**(1/3))

def single_bit_error_rate(m, k):
	# m - number of rows, k - number of items stored
	p = fraction_rows_activated(m, k)  # optimized activation count for sdm
	nact = round(p * m)
	mean = nact
	p = nact/m    # modify p to actual value
	std = math.sqrt(p*m*(1. + p*k + (1. + p*p*m)))
	delta = norm.cdf(0, loc=mean, scale=std)
	return (delta, nact)

def sdm_delta_analytical(num_rows, num_items_stored, activation_count=None):
	delta, nact = single_bit_error_rate(num_rows, num_items_stored)
	if activation_count is not None:
		assert nact == activation_count
	return (delta, nact)


# class SDM_overlap():
# 	# emperically compute sdm single bit error due to overlap in rows activated


# 	def __init__(self, num_rows, num_items_stored, activation_count, num_trials):
# 		self.num_rows = num_rows
# 		self.num_items_stored = num_items_stored
# 		self.activation_count = activation_count
# 		self.num_trials = num_trials
# 		self.trial_count = 0
# 		self.error_count = 0

# 	def get_delta_empirical(num_rows, num_items_stored, activation_count, num_trials)
# 		self.perform_trials()
# 		return
# 		self.pvals = pvals
# 		self.sdm_counters = np.zeros(self.pvals["num_rows"], dtype=np.int32)
# 		# item values are single bit, 0 or 1
# 		self.item_values = np.random.randint(0, high=2, size=pvals["num_states"], dtype=np.int8)
# 		self.item_rows = np.empty((pvals["num_states"], pvals["activation_count"]), dtype=np.int32)
# 		for i in range(pvals["num_states"]):
# 			self.item_rows[i,:] = np.random.choice(pvals["num_rows"], size=pvals["activation_count"], replace=False)
# 		if self.pvals["debug"]:
# 			print("item_values\n%s" % self.item_values)
# 			print("item_rows\n%s" % self.item_rows)
# 		self.save_items()
# 		self.add_noise()
# 		self.recall_items()


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
		self.add_noise()
		self.recall_items()

	def save_items(self):
		if self.pvals["debug"]:
			print("save_items, before counters=\n%s" % self.sdm_counters)
		for i in range(self.pvals["num_states"]):
			self.sdm_counters[self.item_rows[i,:]] += 1 if self.item_values[i] == 1 else -1
			if self.pvals["debug"]:
				print("after saving %i, counters=%s" % (i, self.sdm_counters))
		if self.pvals["debug"]:
			print("sdm_counters\n%s" % self.sdm_counters)

	def add_noise(self):
		self.number_to_flip = round(self.pvals["noise_percent"] * self.pvals["num_rows"] / 100.0)
		if self.number_to_flip == 0:
			return
		indicies = np.random.choice(self.pvals["num_rows"], size=self.number_to_flip, replace=False)
		self.sdm_counters[indicies] *= -1
		self.flipped_counters = indicies

	def recall_items(self):
		recalled_values = np.empty(self.pvals["num_states"], dtype=np.int8)
		if self.pvals["debug"]:
			print("recalling values:")
		for i in range(self.pvals["num_states"]):
			csum = np.sum(self.sdm_counters[self.item_rows[i,:]])
			recalled_values[i] = 1 if csum >= 0 else 0
			if self.pvals["debug"]:
				print("%s - csum=%s" % (i, csum))
		num_correct = np.count_nonzero( recalled_values == self.item_values)
		percent_expected_correct = self.compute_percent_expected_correct()
		hamming_if_512 = 512 * ((self.pvals["num_states"] - num_correct) / self.pvals["num_states"])
		print("num_correct = %s / %s (%s%%), expected=%s%%, hamming_if_512=%s" % (num_correct, self.pvals["num_states"], 
			round(num_correct * 100.0 / self.pvals["num_states"], 1), percent_expected_correct, hamming_if_512))
		if self.number_to_flip > 0:
			fraction_wrong = (100.0 - percent_expected_correct) / 100.0
			delta_ht = ( 1.0 - 2.0 * fraction_wrong) * self.pvals["noise_percent"] / 100.0
			total_fraction_wrong = fraction_wrong + delta_ht
			noise_percent_expected_correct = (1.0 - total_fraction_wrong) * 100.0
			noise_hamming_if_512 = 512 * (1.0 - total_fraction_wrong)
			print("with %s%% noise, expect %s%% correct, expected hamming_if_512=%s" % (
				self.pvals["noise_percent"], noise_percent_expected_correct, noise_hamming_if_512))
			percent_expected_correct_with_noise = self.compute_percent_expected_correct_with_noise()
			print("Using changed mean, expected is: %s" % percent_expected_correct_with_noise)
			percent_expected_correct_with_noise_via_single_mixture = self.compute_percent_expected_correct_with_noise_via_single_mixture()
			print("Using single_mixture, expected is: %s" % percent_expected_correct_with_noise_via_single_mixture)
			percent_expected_correct_with_noise_via_single_difference = self.compute_percent_expected_correct_with_noise_via_single_difference()
			print("Using single_differnce, expected is: %s" % percent_expected_correct_with_noise_via_single_difference)

		if self.pvals["debug"]:
			print("stored values, recalled values=\n%s\n%s" % (self.item_values, recalled_values))

	def compute_percent_expected_correct(self):
		# compute theoretical sdm bit error based on Pentti's book chapter
		nact = self.pvals["activation_count"]
		test_mean, test_variance = self.compute_distribution_for_specific_nact(nact)
		pact = self.pvals["activation_count"] / self.pvals["num_rows"]
		mean = self.pvals["activation_count"]
		num_entities_stored = self.pvals["num_states"]
		sdm_num_rows = self.pvals["num_rows"]
		standard_deviation = math.sqrt(mean * (1 + pact * num_entities_stored * (1.0 + pact*pact * sdm_num_rows)))
		assert math.isclose(test_mean, mean)
		assert math.isclose(test_variance, standard_deviation**2)
		probability_single_bit_failure = norm(0, 1).cdf(-mean/standard_deviation)
		percent_expected_correct = round((1 - probability_single_bit_failure) * 100.0, 1)
		return percent_expected_correct

	def compute_percent_expected_correct_orig(self):
		# compute theoretical sdm bit error based on Pentti's book chapter
		pact = self.pvals["activation_count"] / self.pvals["num_rows"]
		mean = self.pvals["activation_count"]
		num_entities_stored = self.pvals["num_states"]
		sdm_num_rows = self.pvals["num_rows"]
		standard_deviation = math.sqrt(mean * (1 + pact * num_entities_stored * (1.0 + pact*pact * sdm_num_rows)))
		probability_single_bit_failure = norm(0, 1).cdf(-mean/standard_deviation)
		percent_expected_correct = round((1 - probability_single_bit_failure) * 100.0, 1)
		return percent_expected_correct


	def compute_distribution_for_specific_nact(self, nact):
		# compute mean and standard distribution for a specific activation count
		pact = nact / self.pvals["num_rows"]
		mean = nact
		num_entities_stored = self.pvals["num_states"]
		sdm_num_rows = self.pvals["num_rows"]
		variance = mean * (1 + pact * num_entities_stored * (1.0 + pact*pact * sdm_num_rows))
		return (mean, variance)

	def compute_percent_expected_correct_with_noise_via_single_difference(self):
		# use single mixture distributions to compute theoretical sdm bit error
		fraction_flipped = self.pvals["noise_percent"] / 100.0
		fraction_upright = 1.0 - fraction_flipped
		nact_flipped = self.pvals["activation_count"] * fraction_flipped
		nact_upright = self.pvals["activation_count"] * fraction_upright
		print("fraction_flipped=%s, fraction_upright=%s, nact_flipped=%s, nact_upright=%s" % (
			fraction_flipped, fraction_upright, nact_flipped, nact_upright))
		mean_flipped, variance_flipped = self.compute_distribution_for_specific_nact(nact_flipped)
		mean_upright, variance_upright = self.compute_distribution_for_specific_nact(nact_upright)
		mean_combined = fraction_upright * mean_upright - fraction_flipped * mean_flipped
		# following is not sufficient
		variance_combined = (fraction_upright * variance_upright + fraction_flipped * variance_flipped)
		# from: https://stats.stackexchange.com/questions/205126/standard-deviation-for-weighted-sum-of-normal-distributions
		variance_combined += fraction_flipped * fraction_upright * (mean_flipped - mean_upright)**2
		print("mean_flipped=%s, variance_flipped=%s, mean_upright=%s, variance_upright=%s, mean_combined=%s, variance_combined=%s" % (
			mean_flipped, variance_flipped, mean_upright, variance_upright, mean_combined, variance_combined))
		standard_deviation_combined = math.sqrt(variance_combined)
		probability_single_bit_failure = norm(0, 1).cdf(-mean_combined/standard_deviation_combined)
		percent_expected_correct = round((1 - probability_single_bit_failure) * 100.0, 1)
		return percent_expected_correct

	def compute_percent_expected_correct_with_noise_via_single_mixture(self):
		# use single mixture distributions to compute theoretical sdm bit error
		fraction_flipped = self.pvals["noise_percent"] / 100.0
		fraction_upright = 1.0 - fraction_flipped
		nact_flipped = self.pvals["activation_count"] * fraction_flipped
		nact_upright = self.pvals["activation_count"] * fraction_upright
		print("fraction_flipped=%s, fraction_upright=%s, nact_flipped=%s, nact_upright=%s" % (
			fraction_flipped, fraction_upright, nact_flipped, nact_upright))
		mean_flipped, variance_flipped = self.compute_distribution_for_specific_nact(nact_flipped)
		mean_upright, variance_upright = self.compute_distribution_for_specific_nact(nact_upright)
		mean_combined = fraction_upright * mean_upright - fraction_flipped * mean_flipped
		variance_combined = (fraction_upright * (variance_upright + mean_upright**2)
			+ fraction_flipped * (variance_flipped + mean_flipped**2)) - mean_combined
		print("mean_flipped=%s, variance_flipped=%s, mean_upright=%s, variance_upright=%s, mean_combined=%s, variance_combined=%s" % (
			mean_flipped, variance_flipped, mean_upright, variance_upright, mean_combined, variance_combined))
		standard_deviation_combined = math.sqrt(variance_combined)
		probability_single_bit_failure = norm(0, 1).cdf(-mean_combined/standard_deviation_combined)
		percent_expected_correct = round((1 - probability_single_bit_failure) * 100.0, 1)
		return percent_expected_correct


	def compute_percent_expected_correct_with_noise(self):
		# compute theoretical sdm bit error based on Pentti's book chapter, but change mean because of noise
		pact = self.pvals["activation_count"] / self.pvals["num_rows"]
		mean = self.pvals["activation_count"]
		mean = mean * (1 - 2*(self.pvals["noise_percent"] / 100.0))
		num_entities_stored = self.pvals["num_states"]
		sdm_num_rows = self.pvals["num_rows"]
		standard_deviation = math.sqrt(mean * (1 + pact * num_entities_stored * (1.0 + pact*pact * sdm_num_rows)))
		probability_single_bit_failure = norm(0, 1).cdf(-mean/standard_deviation)
		percent_expected_correct_with_noise = round((1 - probability_single_bit_failure) * 100.0, 1)
		return percent_expected_correct_with_noise

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

def testde():
	tvals = [(6, 2, 2)] # [(214, 250, 5), ] # [(15, 35, 2), (100, 250, 3)]
	print("rows\titems\tnact\tnagues\td_found\td_pred")
	for tv in tvals:
		num_rows, num_items_stored, activation_count = tv
		delta_analytical, nact = sdm_delta_analytical(num_rows, num_items_stored, None)
		delta = sdm_delta_empirical(num_rows, num_items_stored, nact)

		print("%s\t%s\t%s\t%s\t%s\t%s" % (num_rows, num_items_stored, nact, activation_count, delta, delta_analytical))

if __name__ == "__main__":
	# test sdm and bundle
	testde()
	# main()
