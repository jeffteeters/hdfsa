# script to count overlaps in SDM
import math
from scipy.stats import hypergeom
from scipy.stats import binom
# from scipy.special import binom as binom_coef
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import copy

import pprint
pp = pprint.PrettyPrinter(indent=4)

class Cop_orig:
	# "chunk overlay probability" - keeps track of multiple overlaps (chunks) onto target rows from the same item.
	# This is needed because multiple overlaps from the same item change the probability of error when computing
	# the sum.  For example, a chunk of size 2 would contribute -2 or +2 to the sum.  But two independent items
	# would contribute either (-2, 0, +2).  Another example, chunk of size 3 contributes -3 or +3.  But three
	# independent items could contribute: -3, -1, 1, or 3.

	
	# chunk_ints = {}   # magnitudes stored as numpy array of integers to make combining easier

	def __init__(self, nact, initial_chunk=None, initial_probability=None):
		# initial_chunk is magnitude of initial overlap.  If specified, must be in range: 0, 1, 2, 3 ... nact
		self.nact = nact
		self.chunk_probabilities = {}  # maps chunk key, e.g. "0,1,0", to probability of that combination
		if initial_chunk is not None:
			assert isinstance(initial_chunk, int) and initial_chunk >= 0 and initial_chunk <= self.nact,(
				"initial_chunk=%s" % initial_chunk)
			if not (isinstance(initial_probability, float) and initial_probability <= 1.0):
				print("initial_probability=%s, type=%s" % (initial_probability, type(initial_probability)))
			assert isinstance(initial_probability, float) and initial_probability <= 1.0
			iar = np.zeros(self.nact, dtype=np.uint16)
			if initial_chunk is not None:
				if initial_chunk > 0:
					iar[initial_chunk - 1] = 1  # to indicate have one chuck of specified magnitude
				key = ','.join(["%s" % x for x in iar])
				self.chunk_probabilities[key] = [initial_probability, iar]

	def add_overlap(self, prev_cop, num_to_add, prob):
		# add overlap to prev_cop to create new entries in this item.
		# prev_cop - previous cop which is having overlaps added
		# num_to_add - number of overlaps to add.  Range must be: 0 <= num_to_add <= nact
		# prob - probability associate with the overlap, is multiplied by existing probabilities in prev_cop
		for prev_key, prev_info in prev_cop.chunk_probabilities.items():
			new_prob = prev_info[0] * prob
			iar = copy.copy(prev_info[1]) # iar = list(map(int, prev_key.split(",")))
			if num_to_add == 0:
				new_key = prev_key  # no need to modify previous key
			else:
				# make updatted key
				iar[num_to_add - 1] += 1   # increment chunk indicator of specified magnitude
				new_key = ','.join(["%s" % x for x in iar])  # form new key
			if new_key in self.chunk_probabilities:
				# key exists, add to current probability
				self.chunk_probabilities[new_key][0] += new_prob
			else:
				self.chunk_probabilities[new_key] = [new_prob, iar]
		
class Cop:
	# "chunk overlay probability" - keeps track of multiple overlaps (chunks) onto target rows from the same item.
	# This is needed because multiple overlaps from the same item change the probability of error when computing
	# the sum.  For example, a chunk of size 2 would contribute -2 or +2 to the sum.  But two independent items
	# would contribute either (-2, 0, +2).  Another example, chunk of size 3 contributes -3 or +3.  But three
	# independent items could contribute: -3, -1, 1, or 3.

	def __init__(self, nrows, nact, k, threshold=10000, show_pruning=False, show_items=True):
		# nrows - number of rows in sdm
		# nact - activaction count
		# k - number of items to store in sdm
		# threshold - maximum ratio of largest to smallest probability.  Drop patterns that have smaller probability
		#  This done to limit number of patterns to only those that are contributing the most to the result
		self.nrows = nrows
		self.nact = nact
		self.k = k
		self.threshold = threshold
		self.show_pruning = show_pruning
		self.show_items = show_items
		self.ov1_pmf = self.compute_one_item_overlap_pmf()
		# print("self.ov1_pmf=%s" % self.ov1_pmf)
		self.key_increments = self.compute_key_increments()
		self.chunk_probabilities = self.set_initial_chunk_probabilities()
		# print("before add_overlap, self.chunk_probabilities=%s" % self.chunk_probabilities)
		for i in range(2, k):
			self.add_overlap(i)
			end = "\n" if i % 10 == 0 else ""
			print("%s-%s "%(i,len(self.chunk_probabilities)), end=end)
			# print("after add_overlap %s self.chunk_probabilities=\n%s" % (i, self.chunk_probabilities))
		self.add_error_rate()
		self.display_result()

	def compute_one_item_overlap_pmf(self):
		# based on: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html#scipy.stats.hypergeom
		[M, n, N] = [self.nrows, self.nact, self.nact]
		rv = hypergeom(M, n, N)
		x = np.arange(0, n+1)
		pmf = rv.pmf(x)
		return pmf

	def compute_key_increments(self):
		# keys in the self.chunk_probabilities dictionary are integers with each three digits representing the
		# overlap count for 0 through nact overlaps.  Least significant digits are used for count of 0 overlaps.
		# In other words, keys look like: 444333222111000, where 000 is the count of 0 overlaps, and 111 is the
		# count of 1 overlaps.  The key increment array has the numbers to add to the previous key to convert it
		# to the new key.
		ki = 1
		key_increments = []
		for i in range(self.nact+1):
			key_increments.append(ki)
			ki = ki * 1000
		# print("key_increments=%s" % key_increments)
		return key_increments

		# create array of numbers that are added to key

	def set_initial_chunk_probabilities(self):
		# create dictionary mapping key for pattern to probability of that pattern
		chunk_probabilities = {}
		for i in range(self.nact+1):
			chunk_probabilities[self.key_increments[i]] = self.ov1_pmf[i]
			# iar = np.zeros(self.nact, dtype=np.uint16)
			# if i > 0:
			# 	iar[i-1] = 1
			# key = ','.join(["%s" % x for x in iar])
			# chunk_probabilities[key] = self.ov1_pmf[i]
		return chunk_probabilities

	def add_overlap(self, iteration):
		# givin current chunk_probabilities, update it by multiplying each probability by every probability in
		# self.ov1_pmf and then combine terms.  Remove any terms with probability less than threshold.
		chunk_probabilities = self.chunk_probabilities
		items_before = len(chunk_probabilities)
		max_probability = 0.
		for prev_key in list(chunk_probabilities):
			prev_prob = chunk_probabilities[prev_key]
			for i in range(self.nact + 1):
				new_prob = prev_prob * self.ov1_pmf[i]
				if new_prob > max_probability:
					max_probability = new_prob
				new_key = prev_key + self.key_increments[i]
				if new_key in chunk_probabilities:
					chunk_probabilities[new_key] += new_prob
				else:
					chunk_probabilities[new_key] = new_prob
				# print("i=%i, prev=(%s, %s) new=(%s,%s), cp[nk]=%s" % (i, prev_key, prev_prob, new_key, new_prob,
				# 	chunk_probabilities[new_key]))
			del chunk_probabilities[prev_key]
		# now check for terms to remove
		min_allowd_probability = max_probability / self.threshold
		prune_count = 0
		prune_prob = 0.0
		total_prob = 0
		items_before_prune = len(chunk_probabilities)
		pruned_items = {}
		for key in list(chunk_probabilities):
			prob = chunk_probabilities[key]
			total_prob += prob
			if prob < min_allowd_probability:
				pruned_items[key] = prob
				del chunk_probabilities[key]
				prune_count += 1
				prune_prob += prob
		items_after_prune = len(chunk_probabilities)
		if self.show_pruning:
			print("cop %s overlap, items_before=%s, items_before_prune=%s, items_after_prune=%s, prune_count=%s,"
				" prune_prob=%s, total_prob=%s" % (iteration, items_before, items_before_prune, items_after_prune,
					prune_count, prune_prob, total_prob))
			print("pruned_items are: %s" % pruned_items)

	def add_error_rate(self):
		# add error rate to chunk_probabilities
		chunk_probabilities = self.chunk_probabilities
		for key in list(chunk_probabilities):
			prob = chunk_probabilities[key]
			err = self.cop_err(self.nact, key)
			chunk_probabilities[key] = (prob, err)

	def display_result(self):
		num_items = len(self.chunk_probabilities)
		total_prob = 0.0
		for key, info in self.chunk_probabilities.items():
			prob, err = info
			total_prob += prob
		print("After %s items stored with nact=%s, threshold=%s, size cop = %s, total_probability = %s" % (self.k,
			self.nact, self.threshold, num_items, total_prob))
		if self.show_items:
			max_num_to_show = 50
			print("items are:")
			for key, info in sorted(self.chunk_probabilities.items())[0:max_num_to_show]:
				print("%s -> %s" % (key, info))
			# self.test_cop_err()

	def cop_err(self, nact, key):
		# determine probability of error given chunk overlap pattern specified in key
		cfreq=[]
		# print("entered cop_err, key=%s" % key)
		dkey = int(key / 1000)  # strip off no overlap count
		if dkey == 0:
			return 0.0  # no overlap, so error rate is zero
		while dkey > 0:
			cfreq.append(dkey % 1000)
			dkey = int(dkey / 1000)
		assert len(cfreq) <= nact
		cw = []  # chunk weights, will store all possible weights for each chunk size, weights are multiples of nact
		cp = []  # chunk probabilities, stores probability of each weight, from binomonial distribution
		for i in range(len(cfreq)):
			weight = i+1
			nchunks = cfreq[i]   # number of chunks of size i+1
			x = np.arange(nchunks+1)
			chunk_weights = x * weight - (nchunks - x) * weight  # positive weights - negative weights
			chunk_probabilities = binom.pmf(x, nchunks, 0.5)  # binomonial coefficients, give number of possible ways to select x items
			cw.append(chunk_weights)
			cp.append(chunk_probabilities)
		chunk_weights = cw[0]
		chunk_probabilities = cp[0]
		for i in range(1, len(cfreq)):
			chunk_weights = np.add.outer(cw[i],chunk_weights).flatten()
			chunk_probabilities = np.outer(cp[i],chunk_probabilities).flatten()
		# if target positive, to cause error, want sum of overlaps plus target <= 0
		perror_sum = np.dot(chunk_weights + nact <= 0, chunk_probabilities)
		# if target negative, to cause error, want sum of overlaps plus target > 0
		nerror_sum = np.dot(chunk_weights - nact > 0, chunk_probabilities)
		prob_sum = np.sum(chunk_probabilities)
		error_rate = (perror_sum + nerror_sum) / (2* prob_sum)  # divide by 2 because positive and negative errors summed
		return error_rate

	def cop_err_empirical(self, nact, key, trials=100000):
		# perform multiple trials to calculate empirical error, used to compare with perdicted error
		cfreq=[]
		dkey = int(key / 1000)  # strip off no overlap count
		if dkey == 0:
			return 0.0  # no overlap, so error rate is zero
		while dkey > 0:
			cfreq.append(dkey % 1000)
			dkey = int(dkey / 1000)
		assert len(cfreq) <= nact
		weights = np.arange(1,len(cfreq)+1)  # will be like: 1,2,3,4,5
		sw = np.repeat(weights, cfreq) # like: [1,1,1,1,1,2,2,3,3,3,4,4,5] if cfreq=[5,2,3,2,1]
		mul = np.random.choice([-1, 1], size=(trials, len(sw)))
		sums = np.matmul(mul, sw)
		pfail = np.count_nonzero(sums + nact <= 0)
		nfail = np.count_nonzero(sums - nact > 0)
		error_count = pfail + nfail
		trial_count = 2 * trials
		error_rate = error_count / trial_count
		return error_rate


	def test_cop_err(self, ntrials=10):
		max_overlaps = np.array([20, 4, 6, 5, 3], dtype=np.int8)
		error_rates = np.zeros(len(max_overlaps), dtype=float)
		for nact in range(1, len(max_overlaps) + 1):
			trial_count = 0
			error_count = 0
			while trial_count < ntrials:
				cfreq = np.random.randint(max_overlaps[0:nact]+1, high=None)
				if(sum(cfreq)==0):
					continue
				key = 0
				for i in range(nact):
					key = key * 1000 + cfreq[nact - i - 1]
				key = key * 1000   # shift so no overlaps takes first three digits in key
				predicted_error_rate = self.cop_err(nact, key)
				found_error_rate = self.cop_err_empirical(nact, key)
				if predicted_error_rate == found_error_rate:
					percent_difference = "None"
				else:
					percent_difference = round(abs(predicted_error_rate - found_error_rate)* 100 /
					   ((predicted_error_rate + found_error_rate)/2), 2)
				print("nact=%s, key=%s, error predicted=%s, found=%s, difference=%s%%" % (
					nact, key, predicted_error_rate, found_error_rate, percent_difference))
				trial_count += 1

	def compute_hamming_dist(self, ncols):
		# compute distribution of probability of each hamming distance
		n = ncols
		chunk_probabilities = self.chunk_probabilities
		pov = np.empty(len(chunk_probabilities), dtype=np.float64)  # probability of overlaps
		perr = np.empty(len(chunk_probabilities), dtype=np.float64) # probability of error for each pattern of overlap
		i = 0
		for key, info in chunk_probabilities.items():
			prob, err = info
			pov[i] = prob
			perr[i] = err
			i += 1
		assert len(pov) == len(perr)
		pmf = np.empty(n + 1)  # probability mass function
		for h in range(len(pmf)):  # hamming distance
			phk = binom.pmf(h, n, perr)
			pmf[h] = np.dot(phk, pov)
		print("hdist (pmf) sum is %s (should be close to 1)" % np.sum(pmf))
		# assert math.isclose(np.sum(pmf), 1.0), "hdist sum is not equal to 1, is: %s" % np.sum(pmf)
		self.ncols = ncols
		self.hdist = pmf

	def compute_overall_perr(self, d):
		# compute overall error, by integrating over all hamming distances
		# ncols is the number of columns in each row of the sdm
		# d is number of item in item memory
		n = self.ncols
		hdist = self.hdist
		# import pdb; pdb.set_trace()
		h = np.arange(len(hdist))
		# self.plot(binom.pmf(h, n, 0.5), "distractor pmf", "hamming distance", "probability")
		# distractor_pmf = binom.pmf(h, n, 0.5)  # this tried to account for matching hammings, but does not work
		# match_hammings_area = (hdist / (hdist + distractor_pmf)) / n
		ph_corr = binom.sf(h, n, 0.5) ** (d-1)
		# ph_corr = (binom.sf(h, n, 0.5) + match_hammings_area) ** (d-1)
		# self.plot(ph_corr, "probability correct", "hamming distance", "fraction correct")
		# self.plot(ph_corr * hdist, "p_corr weighted by hdist", "hamming distance", "weighted p_corr")
		p_corr = np.dot(ph_corr, hdist)
		self.overall_perr = 1 - p_corr
		return self.overall_perr

	def p_error_binom(self, d):
		# compute error by integrating predicted distribution with distractors
		# should do same function as above
		match_hammings = self.hdist
		phds = np.arange(self.ncols+1)  # possible hamming distances
		distractor_hammings = binom.pmf(phds, self.ncols, 0.5)
		# if self.include_empirical:
		# 	self.plot(match_hammings, "match_hammings vs distractor_hammings", "hamming distance",
		# 		"relative frequency", label="match_hammings", data2=distractor_hammings, label2="distractor_hammings")
		num_distractors = d - 1
		dhg = 1.0 # fraction distractor hamming greater than match hamming
		p_err = 0.0  # probability error
		for k in phds:
			dhg -= distractor_hammings[k]
			p_err += match_hammings[k] * (1.0 - dhg ** num_distractors)
		self.overall_perr_binom = p_err
		return p_err


class Ovc:

	def __init__(self, nrows, ncols, nact, k, d=2, include_empirical=True):
		# compute overlaps of target (nact rows) in SDM when k items are stored (k-1 which cause overlaps)
		# nrows is number of rows (hard locations) in the SDM
		# ncols is the number of columns
		# nact is activaction count
		# k is number of items stored in sdm
		# d is the number of items in item memory.  Used to compute probability of error in recall
		# include_empirical is True to display plots and compute empirical distribution
		self.nrows = nrows
		self.ncols = ncols
		self.nact = nact
		self.k = k
		self.d = d
		self.include_empirical = include_empirical
		self.max_num_overlaps = (k-1) * nact
		self.ov = {1 : self.one_item_overlaps()}  # 1 item overlaps, means 2 items are stored
		for i in range(2, k):
			self.ov[i] = self.n_item_overlaps(i)  # i items overlap, means i+1 items are stored
		self.perr = self.compute_perr()
		self.hdist = self.compute_hamming_dist()
		# compute error via new Cop class
		cop = Cop(nrows, nact, k)
		cop.compute_hamming_dist(ncols)
		cop_error = cop.compute_overall_perr(d)
		cop_error_binonimal = cop.p_error_binom(d)
		self.cop = cop
		if include_empirical:
			self.emp_overlap_err = self.empiricalOverlapError()
			self.empiricalError()
			# self.plot(self.perr, "Error vs overlaps", "number of overlaps",
			# 	"probability of error", label="predicted", data2=self.emp_overlap_err, label2="found")
			# self.plot(self.ov[k - 1]["pmf"], "Perdicted vs empirical overlap distribution", "number of overlaps",
			# 	"relative frequency", label="predicted", data2=self.emp_overlaps, label2="found")
			# print("emp_overlaps=%s" % self.emp_overlaps)
			# print("predicted_overlaps=%s" % self.ov[k - 1]["pmf"])
			# print("predicted_hammings=%s" % self.hdist)
			# print("emp_hammings=%s" % self.ehdist)
			# print("cops=")
			# total_patterns = 0
			# for i in range(len(self.ov[k - 1]["cop"])):
			# 	num_pats = len(self.ov[k - 1]["cop"][i].chunk_probabilities)
			# 	total_patterns += num_pats
			# 	print("%s %s item overlaps:" % (num_pats, i))
			# 	samples = {}
			# 	for key, value in self.ov[k - 1]["cop"][i].chunk_probabilities.items():
			# 		samples[key] = value[0]
			# 		if len(samples) == 10:
			# 			break
			# 	pp.pprint(samples)
			# print("total_patterns=%s" % total_patterns)
			self.plot(self.hdist, "Perdicted vs empirical match hamming distribution", "hamming distance",
				"relative frequency", label="predicted", data2=self.ehdist, label2="found")
			self.plot(cop.hdist, "Perdicted vs empirical match hamming distribution (via cop)", "hamming distance",
				"relative frequency", label="predicted", data2=self.ehdist, label2="found")

	def compute_overall_error(nrows, ncols, nact, k, d):
		ov = Ovc(nrows, ncols, nact, k, d, include_empirical=False)
		overall_perr_orig = ov.compute_overall_perr()
		overall_perr = ov.p_error_binom()
		if not math.isclose(overall_perr_orig, overall_perr):
			print("ovc error computations differ:\noverall_perr_orig=%s, p_error_binom=%s" % (
				overall_perr_orig, overall_perr))
		return overall_perr_orig


	def one_item_overlaps(self):
		# compute probability of: (0, 1, 2, ... nact) overlaps if there is just one additional
		# item stored beyond the first item (which is the target, has nact rows that are targets)
		# nt - number of target rows
		# nd - number of non-target rows (distractors)
		nt = self.nact
		nd = self.nrows - nt
		# based on: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html#scipy.stats.hypergeom
		[M, n, N] = [self.nrows, self.nact, self.nact]
		rv = hypergeom(M, n, N)
		x = np.arange(0, n+1)
		pmf = rv.pmf(x)
		cop = [Cop_orig(self.nact, int(i), pmf[i]) for i in x]
		ovi = {"pmf":pmf, "cop":cop}
		return ovi

	def n_item_overlaps(self, n_items):
		# n_items is number of items beyound the first being stored in the sdm.  In other words, n_items is
		# the number of items which could contribute to overlaps onto target rows.  Maximum number of possible overlaps
		# is n_items * nact.  This would occur if every item overlapped the target rows completely.
		# compute probability of: (0, 1, 2, ... (n_items * nact) overlaps if there are n_items additional
		# item stored beyond the first item (which is the target, has nact rows that are targets).
		nact = self.nact
		max_num_overlaps = nact * n_items
		max_num_previous_overlaps = max_num_overlaps - nact
		pmf = np.empty(max_num_overlaps + 1)  # probability mass function
		copl = []
		for no in range(max_num_overlaps + 1):  # number overlaps
			prob = 0.0
			cop = Cop_orig(self.nact)
			if no > max_num_previous_overlaps:
				cio_start = no - max_num_previous_overlaps
			else:
				cio_start = 0
			if no < nact:
				cio_stop = no
			else:
				cio_stop = nact
			# print("n_items=%s, no=%s, cio_start=%s, stop=%s, max_num_overlaps=%s, max_num_previous_overlaps=%s" % (n_items,
			# 	no, cio_start, cio_stop, max_num_overlaps, max_num_previous_overlaps))
			for cio in range(cio_start, cio_stop+1):  # contributions from current item to overlaps
				# print("\tAdding ov[%s][%s] * ov[1][%s]" % (n_items-1, no-cio, cio))
				prob += self.ov[n_items - 1]["pmf"][no - cio] * self.ov[1]["pmf"][cio]
				cop.add_overlap(self.ov[n_items - 1]["cop"][no - cio], cio, self.ov[1]["pmf"][cio])
			pmf[no] = prob
			copl.append(cop)
		assert math.isclose(np.sum(pmf), 1.0), "sum of pmf should be 1 is: %s" % np.sum(pmf)
		ovi = {"pmf":pmf, "cop":copl}
		return ovi

	def compute_perr(self):
		# compute probability of error given number of overlaps
		nact = self.nact
		n_items = self.k - 1  # number of items beyond first (that is overlapping)
		max_num_overlaps = nact * n_items
		no = np.arange(max_num_overlaps + 1)  # number overlaps
		thres = (no - nact)/2
		pe_plus = binom.cdf(thres, no, 0.5)
		# self.plot(pe_plus, "pe_plus", "# overlaps", "binom.cdf(thres, no, 0.5)")
		pe_minus = binom.cdf(thres - .25, no, 0.5)
		# self.plot(pe_minus, "pe_minus", "# overlaps", "binom.cdf(thres - 1, no, 0.5)")
		perr = (pe_plus + pe_minus) / 2.0
		# for i in range(len(no)):
		#	thres = (no - nact)/2
			# if bit stored is positive, then error occurs if counter sum is zero or negative
			# that is, if count of #1's in overlaps is <= thresh

			# if bit stored is negative, then error occurs if counter sum is > nact
			# that is, if count of #-1's is < thresh.  < thresh means <= thresh -1
		return perr

	def compute_hamming_dist(self):
		# compute distribution of probability of each hamming distance
		n = self.ncols
		pov = self.ov[self.k - 1]["pmf"]  # probability of overlaps
		perr = self.perr    # probability of error for each number of overlap
		assert len(pov) == len(perr)
		pmf = np.empty(n + 1)  # probability mass function
		# import pdb; pdb.set_trace()
		for h in range(len(pmf)):  # hamming distance
			phk = binom.pmf(h, n, self.perr)
			pmf[h] = np.dot(phk, pov)
		assert math.isclose(np.sum(pmf), 1.0), "hdist sum is not equal to 1, is: %s" % np.sum(pmf)
		return pmf

	# def compute_perr_orig(self):
	# 	# compute probability of error given number of overlaps
	# 	nact = self.nact
	# 	n_items = self.k - 1  # number of items beyond first (that is overlapping)
	# 	max_num_overlaps = nact * n_items
	# 	no = np.arange(max_num_overlaps + 1)  # number overlaps
	# 	# for i in range(len(no)):
	# 	# 	ner = (no - nact)
	# 	# 	# if bit stored is positive, then error occurs if counter sum is zero or negative
	# 	# 	pe_plus = 
	# 	mer = (no - nact)/2.0
	# 	# self.plot_mer(mer)
	# 	perr = binom.cdf(mer, no, 0.5)
	# 	return perr

	def compute_overall_perr(self):
		# compute overall error, by integrating over all hamming distances
		n = self.ncols
		hdist = self.hdist
		h = np.arange(len(hdist))
		# self.plot(binom.pmf(h, n, 0.5), "distractor pmf", "hamming distance", "probability")
		ph_corr = binom.sf(h, n, 0.5) ** (self.d-1)
		# self.plot(ph_corr, "probability correct", "hamming distance", "fraction correct")
		# self.plot(ph_corr * hdist, "p_corr weighted by hdist", "hamming distance", "weighted p_corr")
		p_corr = np.dot(ph_corr, hdist)
		return 1 - p_corr

	def p_error_binom(self, use_empirical=False):
		# compute error by integrating predicted distribution with distractors
		# should do same function as above
		match_hammings = self.hdist if not use_empirical else self.ehdist
		phds = np.arange(self.ncols+1)  # possible hamming distances
		distractor_hammings = binom.pmf(phds, self.ncols, 0.5)
		if self.include_empirical:
			self.plot(match_hammings, "match_hammings vs distractor_hammings", "hamming distance",
				"relative frequency", label="match_hammings", data2=distractor_hammings, label2="distractor_hammings")
		num_distractors = self.d - 1
		dhg = 1.0 # fraction distractor hamming greater than match hamming
		p_err = 0.0  # probability error
		for k in phds:
			dhg -= distractor_hammings[k]
			p_err += match_hammings[k] * (1.0 - dhg ** num_distractors)
		return p_err


	# def compute_overall_perr_orig(self):
	# 	# compute overall error, assuming each overlap is a separate binomonal distribution and compute
	# 	# the probability of error for that.  Then combine them all by weighting them according to the
	# 	# probability of the overlaps
	# 	n = self.ncols
	# 	mm = self.perr   # match mean (single bit error rate for each overlap number)
	# 	mv = mm*(1.-mm)/n  # match variance
	# 	dm = 0.5
	# 	dv = dm*(1.-dm)/n  # distractor variance
	# 	cm = dm - mm       # combined mean
	# 	cs = np.sqrt(mv + dv)       # combined standard deviation
	# 	ov_per = norm.cdf(0, cm, cs)  # error in each overlap
	# 	# self.plot_ov_per(ov_per)
	# 	self.plot(ov_per, "Error in each overlap computed by cdf of difference between distributions", "# overlaps",
	# 		"fraction error")
	# 	weighted_error = ov_per * self.ov[self.k - 1]
	# 	self.plot(weighted_error, "weighted error", "# overlaps", "fraction error")
	# 	overall_perr = np.dot(ov_per, self.ov[self.k - 1])
	# 	return overall_perr

	# def plot_mer(self, mer):
	# 	nact = self.nact
	# 	n_items = self.k - 1  # number of items beyond first (that are overlapping)
	# 	max_num_overlaps = nact * n_items
	# 	no = np.arange(max_num_overlaps + 1)  # number overlaps
	# 	# show frequency of overlaps
	# 	plt.plot(no, mer, "o-")
	# 	plt.xlabel('Number of overlaps')
	# 	plt.ylabel('int((#overlaps - nact)/2)')
	# 	plt.title('Input to binom.cdf function when  k=%s' % self.k)
	# 	plt.grid(True)
	# 	plt.show()

	def plot(self, data, title, xlabel, ylabel, label=None, data2=None, label2=None):
		xvals = range(len(data))
		plt.plot(xvals, data, "o-", label=label)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.title(title)
		if data2 is not None:
			assert len(data) == len(data2), "len data=%s, data2=%s" % (len(data), len(data2))
			plt.plot(xvals, data2, "o-", label=label2)
			plt.legend(loc="upper right")
		plt.grid(True)
		plt.show()


	# def plot_ov_per(self, ov_per):
	# 	nact = self.nact
	# 	n_items = self.k - 1  # number of items beyond first (that are overlapping)
	# 	max_num_overlaps = nact * n_items
	# 	no = np.arange(max_num_overlaps + 1)  # number overlaps
	# 	# show frequency of overlaps
	# 	plt.plot(no, ov_per, "o-")
	# 	plt.xlabel('Number of overlaps')
	# 	plt.ylabel('norm.cdf(0, cm, cs)')
	# 	plt.title('Result of norm.cdf(0, cm, cs) function when  k=%s' % self.k)
	# 	plt.grid(True)
	# 	plt.show()

	def show_found_values(self):
		nact = self.nact
		n_items = self.k - 1  # number of items beyond first (that are overlapping)
		max_num_overlaps = nact * n_items
		no = np.arange(max_num_overlaps + 1)  # number overlaps
		# show frequency of overlaps
		plt.plot(no, self.ov[n_items]*(n_items*nact), "o-")
		plt.xlabel('Number of overlaps (x)')
		# plt.ylabel('Relative frequency')
		plt.ylabel('Count of overlaps')
		# plt.title('Relative probablilty of x overlaps when  k=%s' % self.k)
		plt.title('Expected count of overlaps when  k=%s' % self.k)
		plt.grid(True)
		plt.show()
		# show probability of error vs. number of overlaps
		plt.plot(no, self.perr, "o-")
		plt.xlabel('Number of overlaps (x)')
		plt.ylabel('Probability of error')
		plt.title('Probability of error vs. overlaps when  k=%s' % self.k)
		plt.grid(True)
		plt.show()
		# show hamming distribution
		self.plot(self.hdist, "hamming distribution", "hamming distance", "probability")


	def empiricalError(self, ntrials=100000):
		# compute empirical error by storing then recalling items from SDM
		trial_count = 0
		fail_count = 0
		bit_errors_found = 0
		bits_compared = 0
		mhcounts = np.zeros(self.ncols+1, dtype=np.int32)  # match hamming counts
		ovcounts = np.zeros(self.max_num_overlaps + 1, dtype=np.int32)
		while trial_count < ntrials:
			# setup sdm structures
			hl_cache = {}  # cache mapping address to random hard locations
			contents = np.zeros((self.nrows, self.ncols+1,), dtype=np.int8)  # contents matrix
			im = np.random.randint(0,high=2,size=(self.d, self.ncols), dtype=np.int8)  # item memory
			addr_base_length = self.k + self.ncols - 1
			address_base = np.random.randint(0,high=2,size=addr_base_length, dtype=np.int8)
			exSeq= np.random.randint(low = 0, high = self.d, size=self.k) # random sequence to represent
			# store sequence
			# import pdb; pdb.set_trace()
			for i in range(self.k):
				address = address_base[i:i+self.ncols]
				data = im[exSeq[i]]
				vector_to_store = np.append(np.logical_xor(address, data), [1])
				hv = hash(address.tobytes()) # hash of address
				if hv not in hl_cache:
					hl_cache[hv] =  np.random.choice(self.nrows, size=self.nact, replace=False)
				hl = hl_cache[hv]  # random hard locations selected for this address
				contents[hl] += vector_to_store*2-1  # convert vector to +1 / -1 then store
			# recall sequence
			# import pdb; pdb.set_trace()
			for i in range(self.k):
				address = address_base[i:i+self.ncols]
				data = im[exSeq[i]]
				hv = hash(address.tobytes()) # hash of address
				hl = hl_cache[hv]  # random hard locations selected for this address
				csum = np.sum(contents[hl], axis=0)  # counter sum
				nadds = csum[-1]     # number of items added to form this sum
				ovcounts[nadds - self.nact] += 1  # for making distribution of overlaps
				recalled_vector = csum[0:-1] > 0   # will be binary vector, also works as int8; slice to remove nadds
				recalled_data = np.logical_xor(address, recalled_vector)
				hamming_distances = np.count_nonzero(im[:,] != recalled_data, axis=1)
				selected_item = np.argmin(hamming_distances)
				if selected_item != exSeq[i]:
					fail_count += 1
				mhcounts[hamming_distances[exSeq[i]]] += 1
				bit_errors_found += hamming_distances[exSeq[i]]
				bits_compared += self.ncols
				trial_count += 1
				if trial_count >= ntrials:
					break
		perr = fail_count / trial_count
		self.plot(mhcounts, "hamming distances found", "hamming distance", "count")
		print("Empirical bit error rate = %s" % (bit_errors_found / bits_compared))
		self.ehdist = mhcounts / trial_count  # form distribution of match hammings
		self.emp_overlaps = ovcounts / np.sum(ovcounts)
		self.emp_overall_perr = perr


	def empiricalOverlapError(self):
		# find probability of error vs number of overlaps empirically
		error_counts = np.zeros(self.max_num_overlaps + 1, dtype=float)
		ntrials = 50000
		vlen = 1000  # number of trials done simultaneously
		trial_count = 0
		while trial_count < ntrials:
			pcounter = np.full(vlen, self.nact, dtype=np.int32)   # positive bit stored
			ncounter = np.full(vlen, -self.nact, dtype=np.int32)  # negative bit stored
			for no in range(1, self.max_num_overlaps+1):
				vector_to_store = np.random.choice([-1, 1], size=vlen)
				pcounter += vector_to_store
				ncounter += vector_to_store
				pfail = np.count_nonzero(pcounter < 1)
				nfail = np.count_nonzero(ncounter > 0)
				error_counts[no] += pfail + nfail
			trial_count += 2 * vlen
			if trial_count >= ntrials:
				break
		emp_overlap_err = error_counts / trial_count
		return emp_overlap_err






	# def make_hamming_hist(self):
	# 	# compute histogram of hamming distance frequencies and make a histogram
	# 	hx = self.perr * self.ov[self.k-1]
	# 	print("hx - input to histogram is:\n%s" % hx)

	# 	n, bins, patches = plt.hist(hx, 50, density=True, facecolor='g', alpha=0.75)
	# 	plt.xlabel('Normalized hamming distance')
	# 	plt.ylabel('Probability')
	# 	plt.title('Histogram of Hamming distance for k=%s' % self.k)
	# 	# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
	# 	# plt.xlim(40, 160)
	# 	# plt.ylim(0, 0.03)
	# 	plt.grid(True)
	# 	plt.show()


def main():
	nrows = 6; nact = 2; k = 5; d = 27; ncols = 33  # original test case
	# nrows = 80; nact = 6; k = 1000; d = 27; ncols = 51  # near full size 
	# test new cop class
	# cop = Cop(nrows, nact, k)
	# return

	# nrows = 2; nact = 2; k = 2; d = 27; ncols = 33 	# test smaller with overlap all the time
	# nrows = 80; nact = 3; k = 300; d = 27; ncols = 51  # near full size 
	ov = Ovc(nrows, ncols, nact, k, d)
	predicted_using_theory_dist = ov.p_error_binom() # ov.compute_overall_perr()
	predicted_using_empirical_dist = ov.p_error_binom(use_empirical=True)
	predicted_using_cop = ov.cop.overall_perr
	predicted_using_cop_binom = ov.cop.overall_perr_binom
	# overall_perr = Ovc.compute_overall_error(nrows, ncols, nact, k, d)
	empirical_err = ov.emp_overall_perr
	print("for k=%s, d=%s, sdm size=(%s, %s, %s), predicted_using_theory_dist=%s, predicted_using_empirical_dist=%s,"
		" predicted_using_cop=%s, predicted_using_cop_binom=%s, empirical_err=%s" % (k, d, nrows, ncols, nact, predicted_using_theory_dist,
			predicted_using_empirical_dist, predicted_using_cop, predicted_using_cop_binom, empirical_err))
	# ncols = 52
	# overall_perr = Ovc.compute_overall_error(nrows, ncols, nact, k)
	# print("for k=%s, sdm size=(%s, %s, %s), overall_perr=%s" % (k, nrows, ncols, nact, overall_perr))


if __name__ == "__main__":
	main()
