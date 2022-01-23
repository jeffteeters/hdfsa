# bundle class using numpy int arrays rather than python integers
# Started this because I thought the integer version had a bug,
# but later found that integer version worked.  So this is not
# needed.  Saving in this file just in case it might be useful later.

class BundleNp:
	# bundle in hd vector using numpy rather than python long integers

	def __init__(self, word_length, noise_percent=0, debug=False):
		self.word_length = word_length
		self.noise_percent = noise_percent
		self.debug = debug
		# self.fmt = "0%sb" % word_length
		self.bundle = np.zeros((self.word_length, ), dtype=np.int16)
		self.k = 0   # count of entries added

	def add(self, v):
		# add numpy vector v to bundle
		self.k += 1
		self.bundle += v
		if self.debug:
			print("add  %s" % format(v, self.fmt))
			print("bundle=%s" % self.bundle)

	def bind_store(self, addr, data):
		# add bind of address and data to bundle
		bind = np.logical_xor(addr,data).astype(np.int8)
		self.add(bind)


	def binarize(self):
		# convert from bundle to binary then to int
		if self.noise_percent > 0:
			add_noise(self.bundle, self.noise_percent)
		if (self.k % 2) == 0: # If even number then break ties
			superposition += np.random.randint(low = 0, high = 2, size =(self.word_length, 1))
			self.add(superposition)
		self.bin_bundle = self.bundle > (self.k / 2).astype(np.int8)
		return self.bin_bundle

	def bind_recall(self, addr):
		data = np.logical_xor(self.bin_bundle,addr).astype(np.int8)
		return data

	def sformat(self, v):
		# return string representation of int array containing binary values
		result = ""
		for i in len(v):
			result += "%s" % i
		return result

	def test():
		word_length = 1000
		debug = False
		bun = BundleNp(word_length, debug=debug)
		fmt = bun.fmt
		a1 = np.random.randint(low = 0, high = 2, size =word_length)
		a2 = np.random.randint(low = 0, high = 2, size =word_length)
		d1 = np.random.randint(low = 0, high = 2, size =word_length)
		d2 = np.random.randint(low = 0, high = 2, size =word_length)
		sr = np.random.randint(low = 0, high = 2, size =word_length)
		bun.bind_store(a1,d1)
		bun.bind_store(a2,d2)
		b = bun.binarize()
		f1 = bun.bind_recall(a1)
		f2 = bun.bind_recall(a2)
		hamming_to_s1 = gmpy2.popcount(d1^b^s1)
		hamming_to_s2 = gmpy2.popcount(a2^b^s2)
		hamming_to_random1 = gmpy2.popcount(a1^b^sr)
		hamming_to_random2 = gmpy2.popcount(a2^b^sr)
		if debug:
			print("a1 = %s" % bun.sformat(a1))
			print("d1 = %s" % bun.sformat(d1))
			print("f1=%s" % bun.sformat(a^s1, fmt))
			print("binb=%s" % bun.sformat(b, fmt))
			print("recalling from bundle:")
			print("a1^b=%s" % bun.sformat(a1^b, fmt))
			print("  s1=%s, diff=%s" % (bun.sformat(s1, fmt), hamming_to_s1))
			print("a2^b=%s" % format(a2^b, fmt))
			print("  s2=%s, diff=%s" % (bun.sformat(s2, fmt), hamming_to_s2))
		hamming_percent_1 = round(100. * hamming_to_s1 / hamming_to_random1)
		hamming_percent_2 = round(100. * hamming_to_s2 / hamming_to_random2)
		print("percent hamming reduction from match to random is: %s and %s" % (hamming_percent_1, hamming_percent_2))
		if hamming_percent_1 < 60 and hamming_percent_2 < 60:
			print("bundle test passed.")
		else:
			print("bundle test failed.")


