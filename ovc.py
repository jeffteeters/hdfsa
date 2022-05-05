# script to count overlaps in SDM
import math
from scipy.stats import hypergeom
from scipy.stats import binom
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

import pprint
pp = pprint.PrettyPrinter(indent=4)

class Cop:
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
			assert isinstance(initial_probability, float) and initial_probability < 1.0
			iar = np.zeros(self.nact, dtype=np.uint16)
			if initial_chunk > 0:
				iar[initial_chunk - 1] = 1  # to indicate have one chuck of specified magnitude
				key = ','.join(["%s" % x for x in iar])
				self.chunk_probabilities[key] = initial_probability

	def add_overlap(self, prev_cop, num_to_add, prob):
		# add overlap to prev_cop to create new entries in this item.
		# prev_cop - previous cop which is having overlaps added
		# num_to_add - number of overlaps to add.  Range must be: 0 <= num_to_add <= nact
		# prob - probability associate with the overlap, is multiplied by existing probabilities in prev_cop
		for prev_key, prev_prob in prev_cop.chunk_probabilities.items():
			if num_to_add == 0:
				new_key = prev_key    # no need to modify previous key
			else:
				# make updatted key
				iar = list(map(int, prev_key.split(",")))
				iar[num_to_add - 1] += 1   # increment chunk indicator of specified magnitude
				new_key = ','.join(["%s" % x for x in iar])  # form new key
			new_prob = prev_prob * prob
			if new_key in self.chunk_probabilities:
				# key exists, add to current probability
				self.chunk_probabilities[new_key] = self.chunk_probabilities[new_key] + new_prob
			else:
				self.chunk_probabilities[new_key] = new_prob


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
		if include_empirical:
			self.emp_overlap_err = self.empiricalOverlapError()
			self.empiricalError()
			self.plot(self.perr, "Error vs overlaps", "number of overlaps",
				"probability of error", label="predicted", data2=self.emp_overlap_err, label2="found")
			self.plot(self.ov[k - 1]["pmf"], "Perdicted vs empirical overlap distribution", "number of overlaps",
				"relative frequency", label="predicted", data2=self.emp_overlaps, label2="found")
			print("emp_overlaps=%s" % self.emp_overlaps)
			print("predicted_overlaps=%s" % self.ov[k - 1]["pmf"])
			print("predicted_hammings=%s" % self.hdist)
			print("emp_hammings=%s" % self.ehdist)
			print("cops=")
			for i in range(1,k):
				print("%s item overlaps:" % i)
				for j in range(len(self.ov[i]["cop"])):
					print("element %s" % j)
					pp.pprint(self.ov[i]["cop"][j].chunk_probabilities)
			self.plot(self.hdist, "Perdicted vs empirical match hamming distribution", "hamming distance",
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
		cop = [Cop(self.nact, int(i), pmf[i]) for i in x]
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
			cop = Cop(self.nact)
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
	# nrows = 2; nact = 2; k = 2; d = 27; ncols = 33 	# test smaller with overlap all the time
	# nrows = 80; nact = 2; k = 1000; d = 100; ncols = 512  # near full size 
	ov = Ovc(nrows, ncols, nact, k, d)
	predicted_using_theory_dist = ov.p_error_binom() # ov.compute_overall_perr()
	predicted_using_empirical_dist = ov.p_error_binom(use_empirical=True)
	# overall_perr = Ovc.compute_overall_error(nrows, ncols, nact, k, d)
	empirical_err = ov.emp_overall_perr
	print("for k=%s, d=%s, sdm size=(%s, %s, %s), predicted_using_theory_dist=%s, predicted_using_empirical_dist=%s,"
		" empirical_err=%s" % (k, d, nrows, ncols, nact, predicted_using_theory_dist, predicted_using_empirical_dist,
			empirical_err))
	# ncols = 52
	# overall_perr = Ovc.compute_overall_error(nrows, ncols, nact, k)
	# print("for k=%s, sdm size=(%s, %s, %s), overall_perr=%s" % (k, nrows, ncols, nact, overall_perr))


if __name__ == "__main__":
	main()
