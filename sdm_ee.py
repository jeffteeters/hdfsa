import numpy as np
import matplotlib.pyplot as plt


class Sdm_ee():

	# calculate SDM empirical error

	def __init__(self, nrows, ncols, nact, k, d, ntrials=100000, count_multiple_matches_as_error=True, debug=False,
			save_error_rate_vs_hamming_distance=False):
		# nrows is number of rows (hard locations) in the SDM
		# ncols is the number of columns
		# nact is activaction count
		# k is number of items stored in sdm
		# d is the number of items in item memory.  Used to compute probability of error in recall with d-1 distractors
		# count_multiple_matches_as_error = True to count multiple distractor hammings == match as error
		# save_error_rate_vs_hamming_distance true to save error_count_vs_hamming, used to figure out what
		# happens to error rate when distractor(s) having same hamming distance as match not always counted as error
		self.nrows = nrows
		self.ncols = ncols
		self.nact = nact
		self.k = k
		self.d = d
		self.ntrials = ntrials
		self.count_multiple_matches_as_error = count_multiple_matches_as_error
		self.debug = debug
		self.save_error_rate_vs_hamming_distance = save_error_rate_vs_hamming_distance
		self.empiricalError()


	def empiricalError(self):
		# compute empirical error by storing then recalling items from SDM
		debug = self.debug
		trial_count = 0
		fail_count = 0
		bit_errors_found = 0
		bits_compared = 0
		mhcounts = np.zeros(self.ncols+1, dtype=np.int32)  # match hamming counts
		max_num_overlaps = (self.k-1) * self.nact
		ovcounts = np.zeros(max_num_overlaps + 1, dtype=np.int32)
		rng = np.random.default_rng()
		if self.save_error_rate_vs_hamming_distance:
			error_count_vs_hamming = np.zeros(self.ncols+1, dtype=np.int32)
		while trial_count < self.ntrials:
			# setup sdm structures
			hl_cache = {}  # cache mapping address to random hard locations
			contents = np.zeros((self.nrows, self.ncols+1,), dtype=np.int8)  # contents matrix
			im = rng.integers(0, high=2, size=(self.d, self.ncols), dtype=np.int8)
			# im = np.random.randint(0,high=2,size=(self.d, self.ncols), dtype=np.int8)  # item memory
			addr_base_length = self.k + self.ncols - 1
			# address_base = np.random.randint(0,high=2,size=addr_base_length, dtype=np.int8)
			## address_base = rng.integers(0, high=2, size=addr_base_length, dtype=np.int8)
			address_base2 = rng.integers(0, high=2, size=(self.k, self.ncols), dtype=np.int8)
			# exSeq= np.random.randint(low = 0, high = self.d, size=self.k) # random sequence to represent
			exSeq = rng.integers(0, high=self.d, size=self.k, dtype=np.int16)
			if debug:
				print("EmpiricalError, trial %s" % (trial_count+1))
				print("im=%s" % im)
				print("address_base2=%s" % address_base2)
				print("exSeq=%s" % exSeq)
				print("contents=%s" % contents)
			# store sequence
			# import pdb; pdb.set_trace()
			for i in range(self.k):
				## address = address_base[i:i+self.ncols]
				address = address_base2[i]
				data = im[exSeq[i]]
				vector_to_store = np.append(np.logical_xor(address, data), [1])
				hv = hash(address.tobytes()) # hash of address
				if hv not in hl_cache:
					hl_cache[hv] =  np.random.choice(self.nrows, size=self.nact, replace=False)
				hl = hl_cache[hv]  # random hard locations selected for this address
				contents[hl] += vector_to_store*2-1  # convert vector to +1 / -1 then store
				if self.debug:
					print("Storing item %s" % (i+1))
					print("address=%s, data=%s, vector_to_store=%s, contents=%s" % (address, data, vector_to_store, contents))
			# recall sequence
			if debug:
				print("Starting recall")
			for i in range(self.k):
				## address = address_base[i:i+self.ncols]
				address = address_base2[i]
				data = im[exSeq[i]]
				hv = hash(address.tobytes()) # hash of address
				hl = hl_cache[hv]  # random hard locations selected for this address
				csum = np.sum(contents[hl], axis=0)  # counter sum
				nadds = csum[-1]     # number of items added to form this sum
				ovcounts[nadds - self.nact] += 1  # for making distribution of overlaps
				recalled_vector = csum[0:-1] > 0   # will be binary vector, also works as int8; slice to remove nadds
				recalled_data = np.logical_xor(address, recalled_vector)
				hamming_distances = np.count_nonzero(im[:,] != recalled_data, axis=1)
				mhcounts[hamming_distances[exSeq[i]]] += 1
				bit_errors_found += hamming_distances[exSeq[i]]
				hamming_d_found = hamming_distances[exSeq[i]]
				selected_item = np.argmin(hamming_distances)
				if selected_item != exSeq[i]:
					fail_count += 1
					if self.save_error_rate_vs_hamming_distance:
						error_count_vs_hamming[hamming_distances[exSeq[i]]] += 1
				elif self.count_multiple_matches_as_error:
					# check for another item with the same hamming distance, if found, count as error
					hamming_distances[selected_item] = self.ncols+1
					next_closest = np.argmin(hamming_distances)
					if(hamming_distances[next_closest] == hamming_d_found):
						fail_count += 1
						if self.save_error_rate_vs_hamming_distance:
							error_count_vs_hamming[hamming_d_found] += 1
				bits_compared += self.ncols
				trial_count += 1
				if trial_count >= self.ntrials:
					break
				if debug:
					print("Recall item %s" % (i+1))
					print("address=%s,data=%s,csum=%s,recalled_vector=%s,recalled_data=%s,hamming_distances=%s,hamming_d_found=%s,fail_count=%s" % (
						address,data,csum,recalled_vector,recalled_data,hamming_distances,hamming_d_found,fail_count))
				if debug and trial_count > 10:
					debug=False
		self.overall_perr = fail_count / trial_count  # overall probability of error
		self.mhcounts = mhcounts   # count of hamming distances found to matching item
		self.error_rate_vs_hamming = [error_count_vs_hamming[x]/mhcounts[x] for x in range(len(mhcounts))
			] if self.save_error_rate_vs_hamming_distance else None
		self.bit_error_rate = bit_errors_found / bits_compared
		self.ehdist = mhcounts / trial_count  # form distribution of match hammings
		self.emp_overlaps = ovcounts / np.sum(ovcounts)  # distribution of overlaps


def plot(data, title, xlabel, ylabel, label=None, data2=None, label2=None):
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


def main():
	nrows=6; ncols=33; nact=2; k=6; d=27
	ee = Sdm_ee(nrows, ncols, nact, k, d)
	plot(ee.mhcounts, "hamming distances found", "hamming distance", "count")
	if ee.error_rate_vs_hamming is not None:
		plot(ee.error_rate_vs_hamming, "error rate vs hamming distances found", "hamming distance", "error rate")
		print("error_rate_vs_hamming=%s" % error_rate_vs_hamming)
	print("Bit error rate = %s" % ee.bit_error_rate)
	print("overall perr=%s" % ee.overall_perr)


if __name__ == "__main__":
	# test sdm and bundle
	main()
