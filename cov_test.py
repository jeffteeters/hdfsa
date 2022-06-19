# script to test overlap weighting fraction

import itertools
import numpy as np

class Sdm_store_history:

	# Record complete history of writing to sdm for purposes of calculating
	# statistics of overlaps (one item, two item, ...)

	def __init__(self, nrows, nact, k, epochs=10, ncols=None, d=None, debug=False):
		# nrows - number of rows
		# nact - activaction count
		# k - number of items (transitions) to store in sdm
		# ncols - number of columns in each row of sdm
		# d - size of item memory (d-1 are distractors)
		self.nrows = nrows
		self.nact = nact
		self.k = k
		self.ncols = ncols
		self.d = d
		self.epochs = epochs
		rng = np.random.default_rng()
		thl = np.empty((self.k, self.nact), dtype=np.uint16)  # transition hard locations
		selected_rows = np.empty(self.k * self.nact, dtype=np.uint16)
		checked_found = np.zeros(self.nact - 1, dtype=[('checked', np.uint32), ('found', np.uint32)])  #,0 for checked, ,1 for found
		for epoch_id in range(self.epochs):
			for i in range(k):
				thl[i,:] = rng.choice(self.nrows, size=self.nact, replace=False)
			sdm_rows = {}  # contain index of all items stored in each row
			for i in range(nrows):
				sdm_rows[i] = []
			# simulate storing all transitions into sdm
			for i in range(k):
				for j in range(self.nact):
					sdm_rows[thl[i,j]].append(i)
			# import pdb; pdb.set_trace()
			# now find number of overlaps of different sizes
			for i in range(k):
				for nc in range(nact, 1, -1):  # number in overlap from same item
					assert nc >= 2
					seli = list(itertools.combinations(range(nact), nc))  # counters to compare
					for si in range(len(seli)):
						min_length = 10e6  # larger than number in any counter
						for hli in seli[si]:  # hard location index
							cl = len(sdm_rows[thl[i,hli]]) - 1
							if cl < min_length:
								min_length = cl
						# min_length is the maximum number of overlaps of length nc
						# now find actual number of overlaps
						cs = set(sdm_rows[thl[i,seli[si][0]]])
						for hli in seli[si][1:]:  # hard location index
							cs = cs.intersection(set(sdm_rows[thl[i,hli]]))
						found_common = len(cs) - 1
						assert found_common >= 0
						assert min_length >= 0
						assert min_length >= found_common
						checked_found[nc-2]['checked'] += min_length
						checked_found[nc-2]['found'] += found_common
		self.check_found = checked_found
		print("ovl\tchecked\tfound\tratio")
		for ovl in range(nact-1):
			checked = checked_found[ovl]['checked']
			found = checked_found[ovl]['found']
			ratio = found/checked
			print("%s\t%s\t%s\t%s" % (ovl+2, checked, found, ratio))



def main():
	# nrows = 6; nact = 2; k = 5; d = 27; ncols = 33  # original test case
	nrows=50; nact=5; k=1000;
	ssh = Sdm_store_history(nrows, nact, k, epochs=1000)


if __name__ == "__main__":
	main()
