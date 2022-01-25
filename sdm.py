# Sdm and Bundle classes

import sdm_utils as su

import numpy as np
# import sys
import random
# from random import randint
# from random import randrange
# # import pprint
# # pp = pprint.PrettyPrinter(indent=4)
import gmpy2
# from gmpy2 import xmpz


# byteorder = "big" # sys.byteorder



class Sdm:
	# implements a sparse distributed memory
	def __init__(self, address_length=128, word_length=128, num_rows=512, nact=5, noise_percent=0,
		sdm_address_method=0, debug=False):
		# nact - number of active addresses used (top matches) for reading or writing
		self.address_length = address_length
		self.word_length = word_length
		self.num_rows = num_rows
		self.nact = nact
		self.noise_percent = noise_percent
		self.data_array = np.zeros((num_rows, word_length), dtype=np.int16)
		self.addresses = su.initialize_binary_matrix(num_rows, word_length)
		self.debug = debug
		self.sdm_address_method = sdm_address_method
		self.fmt = "0%sb" % word_length
		self.hits = np.zeros((num_rows,), dtype=np.int16)
		self.stored = {} # stores number of times value stored and read [<data>, <store_count>, <read_count>]

	def store(self, address, data):
		# store binary word data at top nact addresses matching address
		top_matches = su.find_matches(self.addresses, address, self.nact, index_only = True, distribute_ties=True,
			sdm_address_method=self.sdm_address_method)
		d = su.int2iar(data, self.word_length)
		for i in top_matches:
			self.data_array[i] += d
			self.hits[i] += 1
		if self.debug:
			print("store\n addr=%s\n data=%s" % (format(address, self.fmt), format(data, self.fmt)))
		# self.save_stored(address, data, action="store")

	def bind_store(self, addr, data):
		# store bundle addr and data.  This done to ensure that vector being stored is unique
		# reason is if multiple vectors are stored, even at different addresses, they can
		# interfere with each other in the counters and reduce recall performance
		self.store(addr, addr^data)

	def bind_recall(self, addr):
		# recall value at address addr and unbind it (xor) with addr (reverse of bind_store)
		return self.read(addr) ^ addr

	# def save_stored(self, address, data, action):
	# 	assert action in ("store", "read")
	# 	if address not in self.stored:
	# 		if action == "read":
	# 			print("attempt to read value that is not stored")
	# 			return
	# 		self.stored[address] = [data, 1, 0]
	# 		return
	# 	if action == "store":
	# 		print("storing multiple times at same address")
	# 		return
	# 	# reading at address
	# 	self.stored[address][2] += 1

	# def show_stored(self):
	# 	not_read_count = 0
	# 	for b in self.stored.keys():
	# 		if self.stored[b][2] == 0:
	# 			not_read_count += 1
	# 	print("Numer of items stored = %s, not_read_count=%s" % (len(self.stored), not_read_count))


	def add_noise(self):
		# add noise to counters to test noise resiliency
		if self.noise_percent == 0:
			return
		for i in range(self.num_rows):
			add_noise(self.data_array[i], self.noise_percent)

	def show_hits(self):
		# display histogram of overlapping hits
		midpoint = int(self.num_rows / 2)
		first_half_count = np.sum(self.hits[0:midpoint])
		second_half_count = np.sum(self.hits[midpoint:])
		print("first_half_count: %s, second_half_count %s" % (first_half_count, second_half_count))
		mean_hits = np.mean(self.hits)
		stdev_hits = np.std(self.hits)
		predicted_mean = np.sum(self.hits) / self.num_rows
		predicted_stdev = math.sqrt(predicted_mean)
		print("hits mean=%s, predicted_mean=%s, stdev=%s, predicted_stdev=%s, sum=%s, num_rows=%s" % (
			mean_hits, predicted_mean, stdev_hits, predicted_stdev, np.sum(self.hits), self.num_rows))
		values, counts = np.unique(self.hits, return_counts=True)
		vc = [(values[i], counts[i]) for i in range(len(values))]
		vc.sort(key = lambda y: (y[0], y[1]))
		print("hits - counts:")
		pp.pprint(vc)

	def read(self, address):
		top_matches = su.find_matches(self.addresses, address, self.nact, index_only = True,
			distribute_ties=True, sdm_address_method=self.sdm_address_method)
		i = top_matches[0]
		isum = np.int32(self.data_array[i].copy())  # np.int32 is to convert to int32 to have range for sum
		for i in top_matches[1:]:
			isum += self.data_array[i]
		isum2 = su.iar2int(isum)
		if self.debug:
			print("read\n addr=%s\n top_matches=%s\n data=%s\nisum=%s," % (format(address, self.fmt), top_matches,
				format(isum2, self.fmt), isum))
		# self.save_stored(address, None, action="read")
		return isum2

	def test():
		# test sdm
		word_length = 512
		num_rows = 10
		nact = 5
		debug = False
		sdm = Sdm(address_length=word_length, word_length=word_length, num_rows=num_rows, nact=nact, debug=debug)
		fmt = sdm.fmt
		a1 = random.getrandbits(word_length)
		a2 = random.getrandbits(word_length)
		s1 = random.getrandbits(word_length)
		s2 = random.getrandbits(word_length)
		sr = random.getrandbits(word_length)
		sdm.store(a1, s1)
		sdm.store(a2, s2)
		r1 = sdm.read(a1)
		r2 = sdm.read(a2)
		hamming_r1 = gmpy2.popcount(s1^r1)
		hamming_r2 = gmpy2.popcount(s2^r2)
		hamming_to_random1 = gmpy2.popcount(sr^s1)
		hamming_to_random2 = gmpy2.popcount(sr^s2)
		if debug:
			print("a1 = %s" % format(a1, fmt))
			print("s1 = %s" % format(s1, fmt))
			print("r1 = %s, diff=%s" % (format(r1, fmt), hamming_r1))
			print("a2 = %s" % format(a2, fmt))
			print("s2 = %s" % format(s2, fmt))
			print("r2 = %s, diff=%s" % (format(r2, fmt), hamming_r2))
			print("random distance: %s, %s" % (hamming_to_random1, hamming_to_random2))
		hamming_percent_1 = round(100. * hamming_r1 / hamming_to_random1)
		hamming_percent_2 = round(100. * hamming_r2 / hamming_to_random2)
		print("percent hamming reduction from match to random is: %s and %s" % (hamming_percent_1, hamming_percent_2))
		if hamming_percent_1 < 10 and hamming_percent_2 < 10:
			print("sdm test passed.")
		else:
			print("sdm test failed.")


class Bundle:
	# bundle in hd vector

	def __init__(self, word_length, noise_percent=0, debug=False):
		self.word_length = word_length
		self.noise_percent = noise_percent
		self.debug = debug
		self.fmt = "0%sb" % word_length
		self.bundle = np.zeros((self.word_length, ), dtype=np.int16)

	def add(self, v):
		# add binary vector v to bundle
		d = su.int2iar(v, self.word_length)
		self.bundle += d
		if self.debug:
			print("add  %s" % format(v, self.fmt))
			print("bundle=%s" % self.bundle)

	def bind_store(self, addr, data):
		# add bind of address and data to bundle
		self.add(addr^data)

	def bind_recall(self, addr):
		data = self.bin_bundle ^ addr
		return data

	def binarize(self):
		# convert from bundle to binary then to int
		if self.noise_percent > 0:
			add_noise(self.bundle, self.noise_percent)
		self.bin_bundle = su.iar2int(self.bundle)
		return self.bin_bundle

	def test():
		word_length = 1000
		debug = False
		bun = Bundle(word_length, debug=debug)
		fmt = bun.fmt
		a1 = random.getrandbits(word_length)
		a2 = random.getrandbits(word_length)
		s1 = random.getrandbits(word_length)
		s2 = random.getrandbits(word_length)
		sr = random.getrandbits(word_length)
		bun.add(a1^s1)
		bun.add(a2^s2)
		b = bun.binarize()
		hamming_to_s1 = gmpy2.popcount(a1^b^s1)
		hamming_to_s2 = gmpy2.popcount(a2^b^s2)
		hamming_to_random1 = gmpy2.popcount(a1^b^sr)
		hamming_to_random2 = gmpy2.popcount(a2^b^sr)
		if debug:
			print("a1 = %s" % format(a1, fmt))
			print("s1 = %s" % format(s1, fmt))
			print("a1^s=%s" % format(a1^s1, fmt))
			print("binb=%s" % format(b, fmt))
			print("recalling from bundle:")
			print("a1^b=%s" % format(a1^b, fmt))
			print("  s1=%s, diff=%s" % (format(s1, fmt), hamming_to_s1))
			print("a2^b=%s" % format(a2^b, fmt))
			print("  s2=%s, diff=%s" % (format(s2, fmt), hamming_to_s2))
		hamming_percent_1 = round(100. * hamming_to_s1 / hamming_to_random1)
		hamming_percent_2 = round(100. * hamming_to_s2 / hamming_to_random2)
		print("percent hamming reduction from match to random is: %s and %s" % (hamming_percent_1, hamming_percent_2))
		if hamming_percent_1 < 60 and hamming_percent_2 < 60:
			print("bundle test passed.")
		else:
			print("bundle test failed.")


if __name__ == "__main__":
	# test sdm and bundle
	Sdm.test()
	Bundle.test()
