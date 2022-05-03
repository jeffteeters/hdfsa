# script to generate plot of SDM recall performance vs bit counter widths
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from labellines import labelLine, labelLines
import sys
import random
# import statistics  # don't use statistics.mean because if argument is list of integers, will round to integer
from scipy.stats import norm
from scipy.stats import binom
import scipy.integrate as integrate
import scipy
import math
import pprint
from fractions import Fraction
import ovc as oc

pp = pprint.PrettyPrinter(indent=4)

def pop_random_element(L):
	# pop random element from a list
	# from https://stackoverflow.com/questions/10048069/what-is-the-most-pythonic-way-to-pop-a-random-element-from-a-list
	i = random.randrange(len(L)) # get random index
	L[i], L[-1] = L[-1], L[i]    # swap with the last element
	x = L.pop()                  # pop last element O(1)
	return x

def finite_state_automaton(num_actions, num_states, num_choices):
	# return array of transitions [(action, next_state), (action, next_state), ...] for each state 
	assert num_choices <= num_actions
	fsa = []
	for i in range(num_states):
		possible_actions = list(range(num_actions))
		possible_next_states = list(range(num_states))
		nas = []
		for j in range(num_choices):
			nas.append( ( pop_random_element(possible_actions), pop_random_element(possible_next_states) ) )
		fsa.append(nas)
	return fsa

class Memory:
	# superclass for SDM and bundle

	def nothing():
		return 1



class Sparse_distributed_memory(Memory):
	# implements a sparse distributed memory

	def __init__(self, word_length, nrows, nact, bc):
		# nact - number of active addresses used (top matches) for reading or writing
		# bc - number of bits to use in each counter after finalizing counter matrix
		self.word_length = word_length
		self.nrows = nrows
		self.nact = nact
		self.bc = bc

	def initialize(self):
		self.data_array = np.zeros((self.nrows, self.word_length), dtype=np.int16)
		self.addresses = np.random.randint(0,high=2,size=(self.nrows, self.word_length), dtype=np.int8)
		self.select_cache = {}

	def select(self, address):
		# return index of rows with hard locations (addresses) most closly matching address
		hl_match = np.count_nonzero( address!=self.addresses, axis=1)
		row_ids = np.argpartition(hl_match, self.nact)[0:self.nact]
		# following code to make sure same rows are selected for the same address (safety check)
		addr_str = ''.join(map('{:b}'.format, address))
		if addr_str in self.select_cache:
			assert np.array_equal(row_ids, self.select_cache[addr_str])
		else:
			self.select_cache[addr_str] = row_ids
		return row_ids

	def store(self, address, data):
		# store binary word data at top nact addresses matching address
		row_ids = self.select(address)
		dpn = data * 2 - 1   # convert to +/- 1
		for i in row_ids:
			self.data_array[i] += dpn

	def truncate_counters(self):
		# truncate counters to number of bits in bc.  if bc < 5, zero is included in the range only if
		# bc has a 0.5 fractional part.  if bc >= 5, always include zero in the range.
		include_zero = self.bc >= 5 or int(self.bc * 10) % 10 == 5
		magnitude = 2**int(self.bc) / 2
		self.data_array[self.data_array > magnitude] = magnitude
		self.data_array[self.data_array < -magnitude] = -magnitude
		if not include_zero:
			# replace zero counter values with random +1 or -1
			# random_plus_or_minus_one = np.random.randint(0,high=2,size=(self.num_rows, self.word_length), dtype=np.int8) * 2 -1
			# there might be a fast way to do this in numpy but I couldn't find a way
			for i in range(self.nrows):
				for j in range(self.word_length):
					if self.data_array[i,j] == 0:
						self.data_array[i,j] = random.choice((-1, +1))


	def recall(self, address):
		row_ids = self.select(address)
		isum = np.int32(self.data_array[row_ids[0]].copy())  # np.int32 is to convert to int32 to have range for sum
		for i in row_ids[1:]:
			isum += self.data_array[i]
		recalled_data = np.int8(isum > 0)
		return recalled_data


class Bundle_memory(Memory):
		# implements a bundle memory
	def __init__(self, word_length):
		self.word_length = word_length

	def initialize(self):
		self.data_array = np.zeros(self.word_length, dtype=np.int16)
		self.num_items_stored = 0

	def store(self, data):
		# store data
		dpn = data * 2 - 1   # convert to +/- 1
		self.data_array += dpn
		self.num_items_stored += 1

	def truncate_counters(self):
		# Threshold counters. First force the number of items stored to be odd
		if self.num_items_stored % 2 != 1:
			# add random binary value so an odd number of items are stored
			random_plus_or_minus_one = np.random.randint(0,high=2,size=self.word_length, dtype=np.int8) * 2 -1
			self.data_array += random_plus_or_minus_one
		self.data_array = (self.data_array > 0).astype(np.int8)

	def recall(self, address):
		recalled_data = np.logical_xor(address, self.data_array).astype(np.int8)
		return recalled_data

def empirical_response(mem, actions, states, choices, size, plot_margin_histogram=False, ntrials=13000):
	# find empirical response of sdm or bundle (in mem object)
	# size is number of bytes allocated to memory, used only for including in plot titles
	using_sdm = isinstance(mem, Sparse_distributed_memory)
	trial_count = 0
	fail_count = 0
	epoch_stats = []  # for generating error bars on average response.  One epoch is one storage of fsa
	hamming_margins = []  # difference between match and distractor hamming distances
	match_hammings = []
	while trial_count < ntrials:
		fsa = finite_state_automaton(actions, states, choices)
		im_actions = np.random.randint(0,high=2,size=(actions, mem.word_length), dtype=np.int8)
		im_states = np.random.randint(0,high=2,size=(states, mem.word_length), dtype=np.int8)
		mem.initialize()
		# mem = Sparse_distributed_memory(word_length, nrows, nact, bc) if using_sdm else Bundle_memory(word_length)
		# store transitions
		epoch_stats.append({"fail_count":0, "trial_count":0})
		for state in range(states):
			for transition in fsa[state]:
				action, next_state = transition
				address = np.logical_xor(im_states[state], im_actions[action])
				data = np.logical_xor(address, np.roll(im_states[next_state], 1))
				if using_sdm:
					mem.store(address, data)
				else:
					mem.store(data)
		# recall transitions
		mem.truncate_counters()
		# distractor_hammings = []
		for state in range(states):
			for transition in fsa[state]:
				action, next_state = transition
				address = np.logical_xor(im_states[state], im_actions[action])
				recalled_data = mem.recall(address)
				if using_sdm:
					recalled_next_state_vector = np.roll(np.logical_xor(address, recalled_data) , -1)
				else:
					recalled_next_state_vector = np.roll(recalled_data, -1)  # xor with address done in bundle recall
				hamming_distances = np.count_nonzero( recalled_next_state_vector!=im_states, axis=1)
				match_hamming = hamming_distances[next_state]
				match_hammings.append(match_hamming)
				found_next_state = np.argmin(hamming_distances)
				if found_next_state != next_state:
					fail_count += 1
					epoch_stats[-1]["fail_count"] += 1
					distractor_hamming = hamming_distances[found_next_state]
				else:
					hamming_distances[next_state] = mem.word_length
					closest_distractor = np.argmin(hamming_distances)
					distractor_hamming = hamming_distances[closest_distractor]
				hamming_margins.append(distractor_hamming - match_hamming)
				trial_count += 1
				epoch_stats[-1]["trial_count"] += 1
	error_rate = fail_count / trial_count
	if len(epoch_stats) > 1:
		assert epoch_stats[-1]["trial_count"] == states * choices, "must have all transitions stored in each epoch"
		fail_rates = [epoch_stats[i]["fail_count"]/(states * choices) for i in range(len(epoch_stats))]
		mean_fail_rate = np.mean(fail_rates)
		stdev_fail_rate = np.std(fail_rates)
	else:
		mean_fail_rate = None
		stdev_fail_rate = None
	mm = np.mean(match_hammings)  # match mean
	empirical_delta = mm / mem.word_length
	# mv = statistics.variance(match_hammings)
	# dm = statistics.mean(distractor_hammings)  # distractor mean
	# dv = statistics.variance(distractor_hammings)
	# cm = dm - mm  # combined mean
	# cs = math.sqrt(mv + dv)  # combined standard deviation
	# predicted_error_rate = norm.cdf(0, loc=cm, scale=cs)
	margin_mean = np.mean(hamming_margins)
	margin_std = np.std(hamming_margins)
	predicted_error_rate = norm.cdf(0, loc=margin_mean, scale=margin_std)
	if plot_margin_histogram and size in (9000, 15000):
		title = "size=%s, perr=%s" % (size, predicted_error_rate)
		plot_margin_hist(hamming_margins, title, margin_mean, margin_std, size, trial_count)
	info = {"error_rate": error_rate, "predicted_error_rate":predicted_error_rate,
		"empirical_delta":empirical_delta,
		"margin_mean": margin_mean, "margin_std": margin_std,
		"mean_fail_rate":mean_fail_rate, "stdev_fail_rate":stdev_fail_rate}
		# "mm":mm, "mv":mv, "dm":dm, "dv":dv, "cm":cm, "cs":cs}
	# plot_hist(match_hammings, distractor_hammings, "bc=%s, nact=%s, size=%s" % (bc, nact, size))
	return info


def plot_margin_hist(vals, title, margin_mean, margin_std, size, trial_count):
	nbins = get_nbins(vals)
	n, bins, patches = plt.hist(vals, nbins, density=False, facecolor='g', alpha=0.75)
	# from: https://www.geeksforgeeks.org/how-to-plot-normal-distribution-over-histogram-in-python/
	mu, std = norm.fit(vals)
	print("size=%s, mu=%s, margin_mean=%s, std=%s, margin_std=%s" %(size, mu, margin_mean, std, margin_std))
  
	# Plot the histogram.
	# plt.hist(data, bins=25, density=True, alpha=0.6, color='b')
  
	# Plot the PDF.
	xmin, xmax = plt.xlim()
	x = np.linspace(xmin, xmax, 100)
	p = norm.pdf(x, mu, std) * trial_count
  
	plt.plot(x, p, 'k', linewidth=2)
	title = ("%s, Fit Values: {:.2f} and {:.2f}".format(mu, std)) % title
	plt.title(title)

	plt.xlabel('Margin value')
	plt.ylabel('Count')
	plt.title(title)
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
		# if xv_range < 10:
		#   nbins = 10
		# elif xv_range > 100:
		#   nbins = 100
		# else:
		#   nbins = int(xv_range) + 1
		return nbins

## begin functions for bundle analytical accuracy


# Calculate analytical accuracy of the encoding according to the equation for p_corr from 2017 IEEE Tran paper
def p_corr (N, D, dp_hit):
	dp_rej=0.5
	var_hit = 0.25*N
	var_rej=var_hit
	range_var=10 # number of std to take into account
	fun = lambda u: (1/(np.sqrt(2*np.pi*var_hit)))*np.exp(-((u-N*(dp_rej-dp_hit) )**2)/(2*(var_hit)))*((norm.cdf((u/np.sqrt(var_rej))))**(D-1) ) # analytical equation to calculate the accuracy  

	acc = integrate.quad(fun, (-range_var)*np.sqrt(var_hit), N*dp_rej+range_var*np.sqrt(var_hit)) # integrate over a range of possible values
	# print("p_corr, N=%s, D=%s, dp_hit=%s, return=%s" % (N, D, dp_hit, acc[0]))
	return acc[0]

# Same as function above, but use 15 stds instead of 10 return the probability of error
def p_corr_15 (N, D, dp_hit):
	dp_rej=0.5
	var_hit = 0.25*N
	var_rej=var_hit
	range_var=15 # 10 # number of std to take into account
	fun = lambda u: (1/(np.sqrt(2*np.pi*var_hit)))*np.exp(-((u-N*(dp_rej-dp_hit) )**2)/(2*(var_hit)))*((norm.cdf((u/np.sqrt(var_rej))))**(D-1) ) # analytical equation to calculate the accuracy  

	acc = integrate.quad(fun, (-range_var)*np.sqrt(var_hit), N*dp_rej+range_var*np.sqrt(var_hit)) # integrate over a range of possible values
	# print("p_corr, N=%s, D=%s, dp_hit=%s, return=%s" % (N, D, dp_hit, acc[0]))
	return acc[0]

#Compute expected Hamming distance
def  expectedHamming (K):
	if (K % 2) == 0: # If even number then break ties so add 1
		K+=1
	deltaHam = 0.5 - (scipy.special.binom(K-1, 0.5*(K-1)))/2**K  # Pentti's formula for the expected Hamming distance
	return deltaHam

def BundleErrorAnalytical(N,D,K,ber=0.0):
	# N - width (number of components) in bundle vector
	# D - number of items in item memory that must be matched
	# K - number of vectors stored in bundle
	# ber - bit error rate (rate of bits flipped).  0 for none.
	deltaHam=  expectedHamming (K) #expected Hamming distance
	deltaBER=(1-2*deltaHam)*ber # contribution of noise to the expected Hamming distance
	dp_hit= deltaHam+deltaBER # total expected Hamming distance
	# est_acc=p_corr (N, D, dp_hit) # expected accuracy # DOES NOT WORK IF N IS LARGE, E.G. 200K OR ABOVE
	# est_error = 1 - est_acc  # expected error
	est_error = p_error_binom(N, D, dp_hit) # expected error
	return est_error

## end functions for bundle analytical accuracy


## begin alternate functions for bundle analytical accuracy

def p_error_binom (N, D, dp_hit):
	# compute error by summing values given by two binomial distributions
	# N - width of word (number of components)
	# D - number of items in item memory
	# dp_hit - normalized hamming distance of superposition vector to matching vector in item memory
	phds = np.arange(N+1)  # possible hamming distances
	match_hammings = binom.pmf(phds, N+1, dp_hit)
	distractor_hammings = binom.pmf(phds, N+1, 0.5)  # should this be N (not N+1)?
	num_distractors = D - 1
	dhg = 1.0 # fraction distractor hamming greater than match hamming
	p_err = 0.0  # probability error
	for k in phds:
		dhg -= distractor_hammings[k]
		p_err += match_hammings[k] * (1.0 - dhg ** num_distractors)
	return p_err


# def compute_theoretical_error(sl, mtype, pflip=0):
# 	# compute theoretical error based on storage length.  if mtype bind (bundle) sl is bundle length (bl)
# 	# if mtype is sdm, sl is number of rows in SDM
# 	# pflip is the probability (percent) of a bit flip (used for bundle only)
# 	k = 1000  # number of items stored in bundle
# 	assert mtype in ("sdm", "bind")
# 	num_distractors = 99
# 	if mtype == "bind":
# 		delta = 0.5 - 0.4 / math.sqrt(k - 0.44)  # from Pentti's paper
# 		if pflip > 0:
# 			delta += (1-2*delta)*pflip/100.0
# 		bl = sl
# 		print("bundle, bl=%s, delta=%s" % (bl, delta))
# 	else:
# 		# following from page 15 of Pentti's book chapter
# 		sdm_num_rows = sl
# 		sdm_activation_count = get_sdm_activation_count(sdm_num_rows, k) # round(sdm_num_rows / 100)  # used in hdfsa.py
# 		pact = sdm_activation_count / sdm_num_rows
# 		mean = sdm_activation_count
# 		num_entities_stored = k
# 		# following from Pentti's chapter
# 		standard_deviation = math.sqrt(mean * (1 + pact * num_entities_stored * (1.0 + pact*pact * sdm_num_rows)))
# 		average_overlap = ((num_entities_stored - 1) * sdm_activation_count) * (sdm_activation_count / sdm_num_rows)
# 		# standard_deviation = math.sqrt(average_overlap * 0.5 * (1 - 0.5)) # compute variance assuming binomonal distribution
# 		probability_single_bit_failure = norm(0, 1).cdf(-mean/standard_deviation)
# 		# probability_single_bit_failure = norm(mean, standard_deviation).cdf(0.0)
# 		delta = probability_single_bit_failure
# 		print("sdm num_rows=%s,act_count=%s,pact=%s,average_overlap=%s,mean=%s,std=%s,probability_single_bit_failure=%s" % (
# 			sdm_num_rows, sdm_activation_count, pact, average_overlap, mean, standard_deviation, probability_single_bit_failure))
# 		bl = 512  # 512 bit length words used for sdm

# 	# following uses delta only with explicit arrays and binom distribution
# 	# match_hamming_distribution = np.empty(bl, dtype=np.float64)
# 	# distractor_hamming_distribution = np.empty(bl, dtype=np.float64)
# 	match_hamming_distribution = np.empty(bl, dtype=np.double)
# 	distractor_hamming_distribution = np.empty(bl, dtype=np.double)
# 	if mtype == "bind":
# 		# to save time over binom
# 		mean_hamming_match = delta * bl
# 		variance_hamming_match = delta * (1 - delta) * bl
# 		sd_hamming_match = math.sqrt(variance_hamming_match)
# 		mean_hamming_distractor = 0.5 * bl
# 		variance_distractor = 0.5 * (1 - 0.5) * bl
# 		sd_hamming_distractor = math.sqrt(variance_distractor)
# 	for h in range(bl):
# 		if mtype == "bind":
# 			# use normal_dist to save time, faster than binom for long vectors
# 			match_hamming_distribution[h] = normal_dist(h, mean_hamming_match, sd_hamming_match)
# 			distractor_hamming_distribution[h] = normal_dist(h, mean_hamming_distractor, sd_hamming_distractor)
# 		else:
# 			match_hamming_distribution[h] = binom.pmf(h, bl, delta)
# 			distractor_hamming_distribution[h] = binom.pmf(h, bl, 0.5)
# 	show_distributions = False
# 	if mtype == "sdm" and show_distributions:
# 		plot_dist(match_hamming_distribution, "match_hamming_distribution, delta=%s" % delta)
# 		plot_dist(distractor_hamming_distribution, "distractor_hamming_distribution, delta=%s" % delta)
# 	# now calculate total error
# 	sum_distractors_less = 0.0
# 	prob_correct = 0.0
# 	for h in range(0, bl):
# 		try:
# 			sum_distractors_less += distractor_hamming_distribution[h]
# 			prob_correct_one = 1.0 - sum_distractors_less
# 			prob_correct += match_hamming_distribution[h] * (prob_correct_one ** num_distractors)
# 		except RuntimeWarning as err:
# 			print ('Warning at h=%s, %s' % (h, err))
# 			print("sum_distractors_less=%s" % sum_distractors_less)
# 			print("prob_correct_one=%s" % prob_correct_one)
# 			print("match_hamming_distribution[h] = %s" % match_hamming_distribution[h])
# 			import pdb; pdb.set_trace()

# 			sys.exit("aborting")
# 	prob_error = 1.0 - prob_correct
# 	# import pdb; pdb.set_trace()
# 	return prob_error



## end alternate functions for bundle analytical accuracy


## begin functions for sdm analytical accuracy

# Calculate analytical accuracy of the encoding according to the equation for p_corr from 2017 IEEE Tran paper
# but use python Fractions and the binonomal distribution to reduce roundoff error
# return error rather than p_correct for best accuracy

def p_error_Fraction (N, D, dp_hit):
	wl = N    # word length
	delta = dp_hit
	num_distractors = D - 1
	pflip = Fraction(int(delta*10000), 10000)  # probability of bit flip
	pdist = Fraction(1, 2) # probability of distractor bit matching
	match_hamming = []
	distractor_hamming = []
	for k in range(wl):
		ncomb = Fraction(math.comb(wl, k))
		match_hamming.append(ncomb * pflip ** k * (1-pflip) ** (wl-k))
		distractor_hamming.append(ncomb * pdist ** k * (1-pdist) ** (wl-k))
	# print("sum match_hamming=%s, distractor_hamming=%s" % (sum(match_hamming), sum(distractor_hamming)))
	# if self.env.pvals["show_histograms"]:
	# 	fig = Figure(title="match and distractor hamming distributions for %s%% bit flips" % xval,
	# 		xvals=range(wl), grid=False,
	# 		xlabel="Hamming distance", ylabel="Probability", xaxis_labels=None,
	# 		legend_location="upper right",
	# 		yvals=match_hamming, ebar=None, legend="match_hamming", fmt="-g")
	# 	fig.add_line(distractor_hamming, legend="distractor_hamming", fmt="-m")
	# 	self.figures.append(fig)
	dhg = Fraction(1) # fraction distractor hamming greater than match hamming
	pcor = Fraction(0)  # probability correct
	for k in range(wl):
		dhg -= distractor_hamming[k]
		pcor += match_hamming[k] * dhg ** num_distractors
	# error = (float((1 - pcor) * 100))  # convert to percent
	error = float(1 - pcor) # * 100))  # convert to percent
	return error
	# return float(pcor)


# Calculate analytical accuracy of the encoding according to the equation for p_corr from 2017 IEEE Tran paper
# but use python integers and the binonomal distribution to reduce roundoff error


# def p_corr_binom (N, D, dp_hit):
# 	bl = N    # word_length (bundle length)
# 	delta = dp_hit
# 	num_distractors = D - 1
# 	# following uses delta only with explicit arrays and binom distribution
# 	match_hamming_distribution = np.empty(bl, dtype=np.double)
# 	distractor_hamming_distribution = np.empty(bl, dtype=np.double)
# 	for h in range(bl):
# 		match_hamming_distribution[h] = binom.pmf(h, bl, delta)
# 		distractor_hamming_distribution[h] = binom.pmf(h, bl, 0.5)
# 	# show_distributions = False
# 	# if mtype == "sdm" and show_distributions:
# 	# 	plot_dist(match_hamming_distribution, "match_hamming_distribution, delta=%s" % delta)
# 	# 	plot_dist(distractor_hamming_distribution, "distractor_hamming_distribution, delta=%s" % delta)
# 	# now calculate total error
# 	sum_distractors_less = 0.0
# 	prob_correct = 0.0
# 	for h in range(0, bl):
# 		try:
# 			sum_distractors_less += distractor_hamming_distribution[h]
# 			prob_correct_one = 1.0 - sum_distractors_less
# 			prob_correct += match_hamming_distribution[h] * (prob_correct_one ** num_distractors)
# 		except RuntimeWarning as err:
# 			print ('Warning at h=%s, %s' % (h, err))
# 			print("sum_distractors_less=%s" % sum_distractors_less)
# 			print("prob_correct_one=%s" % prob_correct_one)
# 			print("match_hamming_distribution[h] = %s" % match_hamming_distribution[h])
# 			import pdb; pdb.set_trace()
# 			sys.exit("aborting")
# 	return prob_correct


def SdmErrorAnalytical(R,K,D,nact=None,word_length=512,ber=0.0):
	# R - number rows in sdm
	# K - number of vectors stored in sdm
	# D - number of items in item memory that must be matched
	# nact - activation count.  None to compute optimal activaction count based on number of rows (R)
	# word_length - width (number of components) in each word
	# ber - bit error rate (rate of counters flipped).  0 for none.
	# following from page 15 of Pentti's book chapter
	sdm_num_rows = R
	num_entities_stored = K
	sdm_activation_count = get_sdm_activation_count(sdm_num_rows, num_entities_stored) if nact is None else nact 
	pact = sdm_activation_count / sdm_num_rows
	mean = sdm_activation_count
	# following from Pentti's chapter
	standard_deviation = math.sqrt(mean * (1 + pact * num_entities_stored * (1.0 + pact*pact * sdm_num_rows)))
	# average_overlap = ((num_entities_stored - 1) * sdm_activation_count) * (sdm_activation_count / sdm_num_rows)
	# standard_deviation = math.sqrt(average_overlap * 0.5 * (1 - 0.5)) # compute variance assuming binomonal distribution
	probability_single_bit_failure = norm(0, 1).cdf(-mean/standard_deviation)
	# probability_single_bit_failure = norm(mean, standard_deviation).cdf(0.0)
	deltaHam = probability_single_bit_failure
	# print("sdm num_rows=%s,act_count=%s,pact=%s,average_overlap=%s,mean=%s,std=%s,probability_single_bit_failure=%s" % (
	# 	sdm_num_rows, sdm_activation_count, pact, average_overlap, mean, standard_deviation, probability_single_bit_failure))
	# bl = 512  # 512 bit length words used for sdm
	# deltaBER=(1-2*deltaHam)*ber # contribution of noise to the expected Hamming distance
	# dp_hit= deltaHam+deltaBER # total expected Hamming distance
	# perr = 1 - p_corr (word_length, D, dp_hit)
	# perr = 1 - p_corr_Fraction(word_length, D, deltaHam)
	perr = p_error_Fraction(word_length, D, deltaHam)
	return perr


def get_sdm_activation_count(nrows, number_items_stored):
	nact = round(fraction_rows_activated(nrows, number_items_stored)*nrows)
	if nact < 1:
		nact = 1
	return nact

## end functions for sdm analytical accuracy

def plot_hist(match_hammings, distractor_hammings, title):
	# plot histograms of match mean and distractor hammings
	bin_start = min(min(match_hammings), min(distractor_hammings))
	bin_end = max(max(match_hammings), max(distractor_hammings))
	bins = np.arange(bin_start, bin_end+1, 1)
	plt.hist(match_hammings, bins, alpha=0.5, label='match_hammings')
	plt.hist(distractor_hammings, bins, alpha=0.5, label='distractor_hammings')
	plt.legend(loc='upper right')
	plt.title(title)
	plt.show()


def fraction_rows_activated(m, k):
	# compute optimal fraction rows to activate in sdm
	# m is number rows, k is number items stored in sdm
	return 1.0 / ((2*m*k)**(1/3))

def sdm_response_info(size, bc=8, nact=None, word_length=512, actions=10, states=100, choices=10, fimp=1.0,
		bc_for_rows=None, empirical=True, empirical_delta_error=False, analytical=False, plot_margin_histogram=False):
	# compute sdm recall error for random finite state automata
	# size - number of bytes total storage allocated to sdm and item memory
	# bc - number of bits in each counter after counter finalized.  Has 0.5 added to include zero if less than 5
	# nact - sdm activaction count.  If none, calculate using optimal formula
	# fimp - fraction of item memory and hard location addresses present (1.0 all present, 0- all generated dynamically)
	# size - word_length - width (in bits) of address and counter matrix and item memory
	# actions - number of actions in finite state automaton
	# states - number of states in finite state automation
	# choices - number of choices per state
	# bc_for_rows - bc size used to calculate number of rows.  Used to compare effects of bc with same number rows
	# empirical - True if should get empirical response, False otherwise
	# empirical_delta_error - True if should include empirical_delta_error (use empirical delta to calculate theoretical error)
	item_memory_size = ((actions + states) * word_length / 8) * fimp  # size in bytes of item memory
	if bc_for_rows is None:
		bc_for_rows = bc
	bcu = int(bc_for_rows + .6)  # round up
	size_one_row = (word_length / 8) * fimp + (word_length * bcu/8)  # size one address and one counter row
	nrows = int((size - item_memory_size) / size_one_row)
	number_items_stored = actions * states
	if nact is None:
		nact = round(fraction_rows_activated(nrows, number_items_stored)*nrows)
		if nact < 1:
			nact = 1
	mem = Sparse_distributed_memory(word_length, nrows, nact, bc)
	ri = empirical_response(mem, actions, states, choices, size=size,
			plot_margin_histogram=plot_margin_histogram) if empirical else {"error_rate": None,
		"predicted_error_rate":None, "margin_mean": None, "margin_std": None,
		"mean_fail_rate": None, "stdev_fail_rate": None, "empirical_delta": None}
	num_transitions = states * choices
	ri["analytical_error"] = SdmErrorAnalytical(nrows,num_transitions,states,nact=nact,word_length=word_length) if analytical else None
	ri["sdm_ovc_error_predicted"] = oc.Ovc.compute_overall_error(nrows, word_length, nact, num_transitions, states) if analytical else None
	SdmErrorAnalytical(nrows,num_transitions,states,nact=nact,word_length=word_length) if analytical else None
	ri["empirical_delta_error"] = p_error_Fraction (word_length, states, ri["empirical_delta"]) if (
		empirical_delta_error and ri["empirical_delta"] is not None) else None
	byte_operations_required_for_recall = (word_length / 8) * states + (word_length / 8) * nrows + nact * (word_length / 8)
	parallel_operations_required_for_recall = (word_length / 8) + (word_length / 8) + nact * (word_length / 8)
	fraction_memory_used_for_data = (word_length*nrows*bcu) / (size * 8) 
	info={"err":ri["error_rate"], "predicted_error":ri["predicted_error_rate"],"analytical_error":ri["analytical_error"],
		"margin_mean":ri["margin_mean"], "margin_std":ri["margin_std"],
		"mean_fail_rate":ri["mean_fail_rate"], "stdev_fail_rate":ri["stdev_fail_rate"],
		"empirical_delta": ri["empirical_delta"],
		"empirical_delta_error": ri["empirical_delta_error"],
		"sdm_ovc_error_predicted": ri["sdm_ovc_error_predicted"],
		"nrows":nrows, "nact":nact,
		"word_length":word_length,
		"mem_eff":fraction_memory_used_for_data,
		"recall_ops": byte_operations_required_for_recall,
		"recall_pops": parallel_operations_required_for_recall,
		# "mm":ri["mm"], "ms":math.sqrt(ri["mv"]), "dm":ri["dm"], "ds":math.sqrt(ri["dv"]),
		# "cm":ri["cm"], "cs":ri["cs"]
		}
	return info

def bundle_response_info(size, actions=10, states=100, choices=10, fimp=1.0, empirical=True, analytical=False):
	# compute bundle recall error for random finite state automata
	# size - number of bytes total storage allocated to bundle and item memory
	# fimp - fraction of item memory and hard location addresses present (1.0 all present, 0- all generated dynamically)
	# actions - number of actions in finite state automaton
	# states - number of states in finite state automation
	# choices - number of choices per state
	# empirical - True if should get empirical response, False otherwise
	item_memory_length = actions + states
	word_length = int((size * 8) / (1 + item_memory_length * fimp))
	mem = Bundle_memory(word_length)
	ri = empirical_response(mem, actions, states, choices, size=size) if empirical else {"error_rate": None,
		"predicted_error_rate":None, "mean_fail_rate": None, "stdev_fail_rate": None, "empirical_delta": None}
	num_transitions = states * choices
	ri["analytical_error"] = BundleErrorAnalytical(word_length,states,num_transitions) if analytical else None
	fraction_memory_used_for_data = word_length / (size * 8)
	byte_operations_required_for_recall = (word_length / 8) * states
	parallel_operations_required_for_recall = word_length / 8
	info={"err":ri["error_rate"], "predicted_error":ri["predicted_error_rate"],"analytical_error":ri["analytical_error"],
		"mean_fail_rate": ri["mean_fail_rate"], "stdev_fail_rate": ri["stdev_fail_rate"],
		"empirical_delta":ri["empirical_delta"],
		"mem_eff":fraction_memory_used_for_data,
		"bundle_length":word_length,
		"recall_ops": byte_operations_required_for_recall,
		"recall_pops": parallel_operations_required_for_recall,
		# "mm":ri["mm"], "ms":math.sqrt(ri["mv"]), "dm":ri["dm"], "ds":math.sqrt(ri["dv"]),
		# "cm":ri["cm"], "cs":ri["cs"]
		}
	return info

def plot_info(sizes, bc_vals, resp_info, line_label):
	plots_info = [
		{"subplot": 221, "key":"err","title":"SDM error with different counter bits", "ylabel":"Recall error"},
		{"subplot": None, "key":"predicted_error","title":"SDM predicted error with different counter bits",
			"ylabel":"Recall error", "label": "predicted", "legend_location":"upper right" },
		{"subplot": 222, "key":"err","title":"SDM error with different counter bits", "ylabel":"Recall error", "scale":"log"},
		{"subplot": None, "key":"predicted_error","title":"SDM predicted error with different counter bits",
			"ylabel":"Recall error", "scale":"log", "label": "predicted","legend_location":"lower left"  },
		{"subplot": 223, "key":"nrows","title":"Number rows in SDM vs. size and counter bits","ylabel":"Number rows"},
		{"subplot": 224, "key":"nact","title":"SDM activation count vs counter bits and size","ylabel":"Activation Count"},
		# {"subplot": 221, "key":"mm","title":"Match mean","ylabel":"Hamming distance"},
		# {"subplot": 222, "key":"ms","title":"Match std","ylabel":"Hamming distance"},
		# {"subplot": 223, "key":"dm","title":"Distractor mean","ylabel":"Hamming distance"},
		# {"subplot": 224, "key":"ds","title":"Distractor std","ylabel":"Hamming distance"},
		 ]
	for pidx in range(len(plots_info)):
		pi = plots_info[pidx]
		new_plot = pi["subplot"] is not None
		finishing_plot = pidx == len(plots_info)-1 or plots_info[pidx+1]["subplot"] is not None
		if new_plot:
			plt.subplot(pi["subplot"])
		label = (pi["label"] if "label" in pi else pi["key"]) + " " + line_label
		log_scale = "scale" in pi and pi["scale"] == "log"
		yvals = [resp_info[bc_vals[0]][i][pi["key"]] for i in range(len(sizes))]
		plt.errorbar(sizes, yvals, yerr=None, label="%s %s" % (bc_vals[0], label)) # fmt="-k"
		for i in range(1, len(bc_vals)):
			yvals = [resp_info[bc_vals[i]][j][pi["key"]] for j in range(len(sizes))]
			plt.errorbar(sizes, yvals, yerr=None, label="%s %s" % (bc_vals[i], label), linewidth=1,)# fmt="-k",) # linestyle='dashed',
		labelLines(plt.gca().get_lines(), zorder=2.5)
		if log_scale:
			plt.yscale('log')
			log_msg = " (log scale)"
			# plt.xticks([2, 3, 4, 5, 10, 20, 40, 100, 200])
			# ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
		else:
			log_msg = ""
			# plt.xticks([2, 25, 50, 75, 100, 125, 150, 175, 200])
		if finishing_plot:
			xaxis_labels = ["%s" % int(size/1000) for size in sizes]
			plt.xticks(sizes,xaxis_labels)		
			# xaxis_labels = ["100k", "200k", "300k", "400k", "500k", "600k", "700k", "800k", "900k", "10^6" ]
			# plt.xticks(xvals[mtype],xaxis_labels)
			plt.title(pi["title"]+log_msg)
			plt.xlabel("Size (kB)")
			plt.ylabel(pi["ylabel"])
			plt.grid()
			legend_location = pi["legend_location"] if "legend_location" in pi else "lower right"
			plt.legend(loc=legend_location)
			if pi["subplot"] == 224:
				plt.show()
	return


def vary_sdm_bc(start_size=10000, step_size=2000, stop_size=30001, bc_vals=[1,1.5, 2, 2.5,3.5,4.5,5.5,8]):
	# bc_vals = [1,1.5, 2, 2.5, 3, 3.5 , 8]
	# start_size=20000; step_size=1000; stop_size=33001
	bc_vals = [1, 3.5, 8]
	# start_size=100000; step_size=100000; stop_size=1000001
	start_size=15000; step_size=5000; stop_size=100001
	resp_info = {}
	sizes = range(start_size, stop_size, step_size)
	bc_for_rows=None
	nact=None
	for bc in bc_vals:
		resp_info[bc] = [sdm_response_info(size, bc, nact=nact, bc_for_rows=bc_for_rows) for size in sizes]
	# make plot
	plot_info(sizes, bc_vals, resp_info, line_label="bit")

def vary_nact(start_size=10000, step_size=2000, stop_size=30001, nact_vals=[1,2, 3, 4, 5]):
	resp_info = {}
	start_size=20000; step_size=1000; stop_size=33001
	sizes = range(start_size, stop_size, step_size)
	bc = 8  # used fixed bc
	bc_for_rows=1
	for nact in nact_vals:
		resp_info[nact] = [sdm_response_info(size, bc, nact=nact, bc_for_rows=bc_for_rows) for size in sizes]
	# make plot
	plot_info(sizes, nact_vals, resp_info, line_label="act")

def vary_fimp(start_size=10000, step_size=2000, stop_size=30001, nact_vals=[1,3]):
	resp_info = {}
	start_size=18000; step_size=500; stop_size=24001
	sizes = range(start_size, stop_size, step_size)
	bc = 1  # used fixed bc
	for nact in nact_vals:
		resp_info[nact] = [sdm_response_info(size, bc, nact=nact) for size in sizes]
	# make plot
	plot_info(sizes, nact_vals, resp_info, line_label="act")

def folds2fimp(folds):
	# convert number of folds to fraction of item memory present
	return 1.0 / folds



def sdm_vs_bundle():
	# start_size=16000; step_size=1000; stop_size=33001  # was 500, 24001  # work with  fimp=1.0/16.0
	start_size=2000; step_size=1000; stop_size=15001  # for bundle with no item memory
	# start_size=100000; step_size=100000; stop_size=1000001  # full range
	# start_size=20000; step_size=10000; stop_size=150001
	# start_size=20000; step_size=5000; stop_size=80001
	# start_size=20000; step_size=5000; stop_size=100001
	sizes = range(start_size, stop_size, step_size)
	# bc = 5.5  # used fixed bc
	# bc = 3.5
	bc = 1
	# bc = 3.5 # 3.5 # 8
	# nact = None
	nact = 1
	# fimp=1.0/16.0
	fimp = 0
	sdm_ri = [sdm_response_info(size, bc, nact=nact, fimp=fimp, analytical=True, empirical=True,
		empirical_delta_error=False, plot_margin_histogram=True) for size in sizes]  # ri - response info
	print("sdm_ri=")
	pp.pprint(sdm_ri)
	bundle_ri = [bundle_response_info(size, fimp=fimp, analytical=True, empirical=True) for size in sizes]
	include_relative_error = False
	if include_relative_error:
		for i in range(len(sizes)):
			sdm_ri[i]["relative_error"] = abs(sdm_ri[i]["analytical_error"] - sdm_ri[i]["err"])/(
				max(abs(sdm_ri[i]["analytical_error"]), abs(sdm_ri[i]["err"]))) if sdm_ri[i]["err"] > 0 else (
				sdm_ri[i-1]["relative_error"] if i > 0 else 1)
	# bundle_ri = [{} for size in sizes]
	plots_info = [
		{"subplot": 221, "key":"err","title":"SDM vs bundle error with fimp=%s" % fimp, "ylabel":"Recall error",
			"label":"found"},
		{"subplot": None, "key":"predicted_error","title":"SDM vs bundle error with fimp=%s" % fimp, "ylabel":"Recall error",
			"label":"found_cdf"},
		{"subplot": None, "key":"sdm_ovc_error_predicted","title":"SDM vs bundle error with fimp=%s" % fimp, "ylabel":"Recall error",
			"label":"sdm_ovc_predicted"},
		{"subplot": None, "key":"analytical_error","title":"SDM vs bundle error with fimp=%s" % fimp, "ylabel":"Recall error",
			"label":"analytical"},
		# {"subplot": None, "key":"empirical_delta_error","title":"SDM vs bundle error with fimp=%s" % fimp, "ylabel":"Recall error",
		# 	"label":"empirical_delta_err"},
		# {"subplot": None, "key":"mean_fail_rate","title":"SDM vs bundle error with fimp=%s" % fimp, "ylabel":"Recall error",
		# 	"label":"mean_fail_rate", "yerr_key":"stdev_fail_rate"},
		{"subplot": 222, "key":"err","title":"SDM vs bundle error with fimp=%s (log scale)" % fimp,
			"ylabel":"Recall error", "scale":"log", "label":"found"},
		{"subplot": None, "key":"predicted_error","title":"SDM vs bundle error with fimp=%s" % fimp, "ylabel":"Recall error",
			"label":"found_cdf",},
		{"subplot": None, "key":"sdm_ovc_error_predicted","title":"SDM vs bundle error with fimp=%s" % fimp, "ylabel":"Recall error",
			"label":"sdm_ovc_predicted"},
		{"subplot": None, "key":"analytical_error","title":"SDM vs bundle error with fimp=%s" % fimp, "ylabel":"Recall error",
			"label":"analytical",  "scale":"log", "legend_location":"lower left"},
		# {"subplot": None, "key":"empirical_delta_error","title":"SDM vs bundle error with fimp=%s" % fimp, "ylabel":"Recall error",
		# 	"label":"empirical_delta_err", "scale":"log"},
		# {"subplot": None, "key":"mean_fail_rate","title":"SDM vs bundle error with fimp=%s" % fimp, "ylabel":"mean_error",
		# 	"label":"mean_fail_rate", "yerr_key":"stdev_fail_rate", "scale":"log"},
		# {"subplot": 222, "key":"mem_eff","title":"SDM vs bundle mem_eff with fimp=%s" % fimp, "ylabel":"Fraction mem used"},
		# {"subplot": 223, "key":"nrows","title":"SDM num rows", "ylabel":"Number rows"},
		# {"subplot": 224, "key":"nact","title":"SDM num rows activated", "ylabel":"Number rows"},
		{"subplot": 223, "key":"margin_mean","title":"SDM margin mean", "ylabel":"hamming distance", "yerr_key":"margin_std"},
		# {"subplot": 223, "key":"relative_error","title":"Relative err vs analytical", "ylabel":"Relative error",},
		{"subplot": 224, "key":"nact","title":"SDM num rows activated", "ylabel":"Number rows"},
		# {"subplot": 224, "key":"bundle_length","title":"bundle_length with fimp=%s" % fimp, "ylabel":"Bundle length"},

		# {"subplot": 224, "key":"nact","title":"sdm activation with fimp=%s" % fimp, "ylabel":"activaction count"},
		]
	for pidx in range(len(plots_info)):
		pi = plots_info[pidx]
		new_plot = pi["subplot"] is not None
		finishing_plot = pidx == len(plots_info)-1 or plots_info[pidx+1]["subplot"] is not None
		if new_plot:
			plt.subplot(pi["subplot"])
		label = pi["label"] if "label" in pi else ""
		log_scale = "scale" in pi and pi["scale"] == "log"
		if pi["key"] in sdm_ri[0] and sdm_ri[0][pi["key"]] is not None:
			yerr = [sdm_ri[i][pi["yerr_key"]]/2.0 for i in range(len(sizes))  # / 2 for symmetric stdev
				] if "yerr_key" in pi and sdm_ri[0][pi["yerr_key"]] is not None else None
			yvals = [sdm_ri[i][pi["key"]] for i in range(len(sizes))]
			plt.errorbar(sizes, yvals, yerr=yerr, label="%ssdm" % label)
		if pi["key"] in bundle_ri[0]  and bundle_ri[0][pi["key"]] is not None:
			yvals = [bundle_ri[i][pi["key"]] for i in range(len(sizes))]
			plt.errorbar(sizes, yvals, yerr=None, label="%sbundle" % label)
		if finishing_plot:
			xaxis_labels = ["%s" % int(size/1000) for size in sizes]
			plt.xticks(sizes,xaxis_labels)
			if log_scale:
				plt.yscale('log')
				plt.ylim([10e-16, None])
			plt.title(pi["title"])
			plt.xlabel("Size (kB)")
			plt.ylabel(pi["ylabel"])
			legend_location = pi["legend_location"] if "legend_location" in pi else "upper right"
			plt.legend(loc=legend_location)
			plt.grid()
	plt.show()


def widths_vs_folds():
	# display plot of SDM length (number of rows) and bundle word length for different number of folds and size
	# memory
	start_size=100000; step_size=100000; stop_size=1000001
	# start_size=100000; step_size=100000; stop_size=600001
	sizes = range(start_size, stop_size, step_size)
	# folds = [1, 2, 4, 8, 16, 32, 64, 128, "inf"]
	folds = [1, 2, 4, 8, 16, 32, 64, 128, 256, "inf"]
	# folds = [1, 2, 4]
	fimps = [ 1 / f for f in folds[0:-1]] + [0.0]
	# fimps = [ 1 / f for f in folds]
	sdm_ri = []
	bun_ri = []
	for size in sizes:
		bc = 4
		sdm_ri.append( [sdm_response_info(size, bc, fimp=fimp, empirical=False, analytical=True) for fimp in fimps] ) # ri - response info
		bun_ri.append( [bundle_response_info(size, fimp=fimp, empirical=False, analytical=True) for fimp in fimps] )
		# add in ratio to fimp=1
		for i in range(len(fimps)):
			sdm_ri[-1][i]["sdm_ratio"] = sdm_ri[-1][i]["nrows"] / sdm_ri[-1][0]["nrows"]
			bun_ri[-1][i]["bun_ratio"] = bun_ri[-1][i]["bundle_length"] / bun_ri[-1][0]["bundle_length"]
	# make plots
	plots_info = [
		# {"subplot": 221, "key":"nrows","title":"SDM rows vs. folds", "ylabel":"Number rows"},
		# {"subplot": 222, "key":"bundle_length","title":"superposition width vs. folds", "ylabel":"Bundle length"},
		# {"subplot": 223, "key":"sdm_ratio","title":"SDM rows vs. folds ratio", "ylabel":"Ratio"},
		# {"subplot": 224, "key":"bun_ratio","title":"superposition width vs. folds ratio", "ylabel":"Ratio"},
		{"subplot": 121, "key":"analytical_error","title":"Analytical error vs. folds", "ylabel":"Fraction error", "label":"theory error"},
		{"subplot": 122, "key":"analytical_error","title":"Analytical error vs. folds (log scale)", "ylabel":"Fraction error",
			"label": "theory error", "scale":"log"},
		# {"subplot": 223, "key":"sdm_ratio","title":"SDM rows vs. folds ratio", "ylabel":"Ratio"},
		# {"subplot": 224, "key":"bun_ratio","title":"superposition width vs. folds ratio", "ylabel":"Ratio"},

		# {"subplot": 223, "key":"recall_ops","title":"Recall byte operations vs num folds", "ylabel":"Byte operations",
		# 	"scale": "log"},
		# {"subplot": 224, "key":"recall_pops","title":"Parallel recall byte operations vs num folds", "ylabel":"Parallel operations",
		# 	"scale": "log"},
		]
	for pi in plots_info:
		plt.subplot(pi["subplot"])
		log_scale = "scale" in pi and pi["scale"] == "log"
		have_sdm_key = pi["key"] in sdm_ri[0][0] and sdm_ri[0][0][pi["key"]] is not None
		have_bun_key = pi["key"] in bun_ri[0][0] and bun_ri[0][0][pi["key"]] is not None
		need_mem_label = have_sdm_key and have_bun_key
		# import pdb; pdb.set_trace()
		if have_sdm_key:
			yvals = [sdm_ri[i][0][pi["key"]] for i in range(len(sizes))]
			mem_label = "sdm " if need_mem_label else ""
			# print("plotting sdm, mem_label=%s" % mem_label)
			# print("fold=%s, yvals=%s" % (folds[0], yvals))
			plt.errorbar(sizes, yvals, yerr=None, label="%s%s" % (mem_label, folds[0])) # fmt="-k"
			for j in range(1, len(folds)):
				yvals = [sdm_ri[i][j][pi["key"]] for i in range(len(sizes))]
				# print("fold=%s, yvals=%s" % (folds[j], yvals))
				plt.errorbar(sizes, yvals, yerr=None, label="%s%s" % (mem_label, folds[j]), linewidth=1,)# fmt="-k",) # linestyle='dashed',
			# labelLines(plt.gca().get_lines(), zorder=2.5)
		if have_bun_key:
			mem_label = "bun " if need_mem_label else ""
			yvals = [bun_ri[i][0][pi["key"]] for i in range(len(sizes))]
			print("plotting bun, mem_label=%s" % mem_label)
			print("fold=%s, yvals=%s" % (folds[0], yvals))
			plt.errorbar(sizes, yvals, yerr=None, label="%s%s" % (mem_label, folds[0])) # fmt="-k"
			for j in range(1, len(folds)):
				yvals = [bun_ri[i][j][pi["key"]] for i in range(len(sizes))]
				print("fold=%s, yvals=%s" % (folds[j], yvals))
				plt.errorbar(sizes, yvals, yerr=None, label="%s%s" % (mem_label, folds[j]), linewidth=1,)# fmt="-k",) # linestyle='dashed',
		if have_sdm_key or have_bun_key:
			labelLines(plt.gca().get_lines(), zorder=2.5)
		else:
			sys.exit("trying to plot nothing")
		xaxis_labels = ["%s" % int(size/1000) for size in sizes]
		plt.xticks(sizes,xaxis_labels)
		if log_scale:
			plt.yscale('log')
		plt.title(pi["title"])
		plt.xlabel("Size (kB)")
		plt.ylabel(pi["ylabel"])
		# plt.legend(loc='upper left')
		plt.grid()
	plt.savefig('widths_vs_folds.pdf')
	plt.show()

def test_analytical_methods():
	# compare analytical methods for different size bundle
	# deltas =  np.linspace(0.1,0.48,10)
	item_memory_size = 100
	err_frady = []
	err_fraction= []
	err_binom= []
	test_type = "vary_delta"
	if test_type == "vary_delta":
		word_length = 512  # use short word length
		# deltas = np.linspace(0.05,0.45,15)
		deltas = np.arange(0.05, 0.46, .05)
		for delta in deltas:
			err_frady.append(1.0 - p_corr (word_length, item_memory_size, delta))
			err_fraction.append(p_error_Fraction (word_length, item_memory_size, delta))
			err_binom.append(p_error_binom(word_length, item_memory_size, delta))
		plt.subplot(111)
		plt.errorbar(deltas, err_frady, yerr=None, label="Frady") # fmt="-k"
		plt.errorbar(deltas, err_binom, yerr=None, label="binom") # fmt="-k"
		plt.errorbar(deltas, err_fraction, yerr=None, label="fraction") # fmt="-k"
		title = "Recall error vs. normalized hamming distance (delta)"
		xlabel="normalized hamming distance"
	elif test_type == "vary_word_length":
		delta_hamming = 0.48738749091081174  # hamming for storing 1001 transitions in a bundle
		start_size=100000; step_size=100000; stop_size=600001
		word_lengths = list(range(start_size, step_size, stop_size))
		for word_length in word_lengths:
			err_frady.append(1.0 - p_corr (word_length, item_memory_size, delta_hamming))
			err_binom.append(p_error_binom(word_length, item_memory_size, delta_hamming))
			# make plots
			plt.errorbar(word_lengths, err_frady, yerr=None, label="Frady") # fmt="-k"
			plt.errorbar(word_lengths, err_binom, yerr=None, label="binom") # fmt="-k"
			title = "Recall error vs word_length"
			xlabel="word_length"
	else:
		sys.exit("Invalid test_type: %s" % test_type)
	log_scale = True
	if log_scale:
		plt.yscale('log')
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel("Recall error fraction")
	plt.grid()
	plt.legend(loc='upper left')
	plt.show()


def widths_vs_folds_single_size(size=100000, empirical=True):
	# display plot of SDM length (number of rows) and bundle word length for different number of folds at a fixed size
	# memory
	# empirical == True to include empirical recall error
	folds = [1, 2, 4, 8, 16, 32, 64, 128, "inf"]
	fimps = [ 1 / f for f in folds[0:-1]] + [0.0]
	# folds = [1, 2, 4, 8, 16, 32, 64] # 8, 16, 32, 64]
	# fimps = [ 1 / f for f in folds]
	bc = 1
	sdm_ri = [sdm_response_info(size, bc, fimp=fimp, empirical=empirical) for fimp in fimps]  # ri - response info
	bun_ri = [bundle_response_info(size, fimp=fimp, empirical=empirical) for fimp in fimps]
	# compute ratio, add to sdm_ri
	for i in range(len(fimps)):
		sdm_ri[i]["bun/sdm_pops"] = bun_ri[i]["recall_pops"] / sdm_ri[i]["recall_pops"] 
	# print("sdm_ri=")
	# pp.pprint(sdm_ri)
	# print("bun_ri=")
	# pp.pprint(bun_ri)
	# make plots
	plots_info = [
		{"subplot": 221, "key":"nrows","title":"SDM num rows vs num folds for size=%s" % size, "ylabel":"Number rows"},
		{"subplot": 222, "key":"bundle_length","title":"bundle_length vs num folds for size=%s" % size, "ylabel":"Bundle length"},
		{"subplot": 223, "key":"recall_ops","title":"Recall byte operations vs num folds", "ylabel":"Byte operations",
			"scale": "log"},
		{"subplot": 224, "key":"recall_pops","title":"Parallel recall byte operations vs num folds", "ylabel":"Parallel operations",
			"scale": "log"},
		{"subplot": 121, "key":"bun/sdm_pops", "title":"Ratio of bundle/sdm recall byte operations vs num folds", "ylabel":"Parallel operations",},
		{"subplot": 122, "key":"bun/sdm_pops", "title":"Ratio of bundle/sdm recall byte operations vs num folds (log scale)", "ylabel":"Parallel operations",
			"scale": "log"},
		{"subplot": 221, "key":"err","title":"SDM vs bundle error vs num folds for size=%s" % size, "ylabel":"Recall error"}, # "scale":"log"},
		{"subplot": 222, "key":"err","title":"SDM vs bundle error vs num folds for size=%s (log scale)" % size, "ylabel":"Recall error", "scale":"log"},
		{"subplot": 223, "key":"predicted_error","title":"SDM vs bundle predicted error vs num folds for size=%s" % size, "ylabel":"Recall error"}, # "scale":"log"},
		{"subplot": 224, "key":"predicted_error","title":"SDM vs bundle predicted error vs num folds for size=%s (log scale)" % size, "ylabel":"Recall error", "scale":"log"},

		]
	for pi in plots_info:
		plt.subplot(pi["subplot"])
		log_scale = "scale" in pi and pi["scale"] == "log"
		need_mem_label = pi["key"] in sdm_ri[0] and pi["key"] in bun_ri[0]
		xpos = list(range(len(folds)))
		if pi["key"] in sdm_ri[0]:
			yvals = [sdm_ri[i][pi["key"]] for i in range(len(folds))]
			mem_label = "sdm " if need_mem_label else ""
			plt.errorbar(xpos, yvals, yerr=None, label="%s%s" % (mem_label, pi["key"])) # fmt="-k"
			# yvals = [sdm_ri[i][pi["key"]] for i in range(len(folds))]
			# plt.errorbar(sizes, yvals, yerr=None, label="%s%s" % (mem_label, folds[j]), linewidth=1,)# fmt="-k",) # linestyle='dashed',
			# labelLines(plt.gca().get_lines(), zorder=2.5)
		if pi["key"] in bun_ri[0]:
			mem_label = "bun " if need_mem_label else ""
			yvals = [bun_ri[i][pi["key"]] for i in range(len(folds))]
			plt.errorbar(xpos, yvals, yerr=None, label="%s%s" % (mem_label, pi["key"])) 
		xaxis_labels = ["%s" % f for f in folds]
		plt.xticks(xpos,xaxis_labels)
		if log_scale:
			plt.yscale('log')
		plt.title(pi["title"])
		plt.xlabel("Folds")
		plt.ylabel(pi["ylabel"])
		plt.legend(loc='upper left')
		plt.grid()
		if pi["subplot"] in (224, 122):
			plt.show()
	# plt.show()



if __name__ == "__main__":
	# vary_sdm_bc()
	# vary_nact()
	sdm_vs_bundle()
	# widths_vs_folds()
	# test_analytical_methods()
	# widths_vs_folds_single_size()
