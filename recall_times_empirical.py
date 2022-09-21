# script to find time it takes to recall data from memory empirically

from build_eedb import Empirical_error_db, get_sdm_ee, get_bundle_ee
import numpy as np
import time
from timeit import default_timer as timer

def update_recall_times():
	edb = Empirical_error_db()
	names = edb.get_memory_names()
	for name in names:
		mi = edb.get_minfo(name)
		mtype = mi["mtype"]
		bits_per_counter = mi["bits_per_counter"]
		match_method = mi["match_method"]
		ndims = len(mi["dims"])
		recall_times = np.empty(ndims, dtype=int)
		needed_epochs = 3
		for dim in mi["dims"]:
			if mtype == "sdm":
				(dim_id, ie, nrows, ncols, nact, pe, epochs, mean, std, recall_time_mean, recall_time_std,
					 match_counts, distract_counts, pmf_error) = dim
			else:
				(dim_id, ie, ncols, pe, epochs, mean, std, recall_time_mean, recall_time_std,
					match_counts, distract_counts, pmf_error) = dim
			if recall_time_mean is None:
				# need to get recall time
				print("%s, starting ie=%s, %s, needed_epochs=%s" % (time.ctime(), ie, name, needed_epochs))
				if mtype == "sdm":
					fee = get_sdm_ee(nrows, ncols, nact, bits_per_counter, match_method, needed_epochs)
				else:
					fee = get_bundle_ee(ncols, bits_per_counter, match_method, needed_epochs) # find empirical error
				# save recall time
				edb.add_recall_time(dim_id, fee.recall_time_mean, fee.recall_time_std)
				print("%s-%s, time=%s, std=%s" % (mi["short_name"], ie, fee.recall_time_mean, fee.recall_time_std))

def compare_operations(nrows=1000, ncols=512):
	rng = np.random.default_rng()
	# compare different methods of converting binary to bipolar
	b1 = rng.integers(0, high=2, size=(nrows, ncols), dtype=np.int8)
	b2 = rng.integers(0, high=2, size=(nrows, ncols), dtype=np.int8)
	# convert to +1, -1; then multiply
	b1b = b1*2-1
	b2b = b2*2-1
	print("binary xor vs bipolar mult")
	start_time = time.perf_counter_ns()
	b1b_b2b_mult = b1b * b2b
	mult_time = time.perf_counter_ns() - start_time
	# normal xor
	start_time = time.perf_counter_ns()
	b12_xor = np.logical_xor(b1, b2)
	xor_time = time.perf_counter_ns() - start_time
	assert np.all(-(b12_xor*2-1) == b1b_b2b_mult)
	mult_xor_ratio = mult_time / xor_time
	print("xor_time=%s, mult_time=%s, mult_time/xor_time=%s" % (xor_time, mult_time, mult_xor_ratio))
	print("dot product vs hamming distance")
	# compare dot product to hamming distance
	# hamming distance
	im_address = rng.integers(0, high=2, size=ncols, dtype=np.int8)
	start_time = time.perf_counter_ns()
	ham_distances = np.count_nonzero(b1!=im_address, axis=1)
	ham_time = time.perf_counter_ns() - start_time
	# dot product distance
	im_address_bipolar = im_address*2-1
	start_time = time.perf_counter_ns()	
	dot_distances = np.dot(b1b, im_address_bipolar)
	dot1_time = time.perf_counter_ns() - start_time
	start_time = time.perf_counter_ns()
	dot_distances2 = np.sum(b1b[:,] * im_address_bipolar, axis=1)
	dot2_time = time.perf_counter_ns() - start_time
	assert np.all(dot_distances == dot_distances2)
	# following constants used to convert hamming distances to dot product distances, using a * ham + b == dot
	# solving for ham == 0  => dot == ncols;  ham == ncols => dot == -ncols.
	a = -2
	b = ncols
	assert np.all(dot_distances2 == a*ham_distances+b)
	print("dot vs hamming")
	print("ham_time=%s, dot1=%s, dot2=%s, dot1/ham=%s, dot2/ham=%s" % (ham_time, dot1_time, dot2_time,
		dot1_time/ham_time, dot2_time/ham_time))

	print("dot product of binary vector vs non-binary (sv) with item memory")
	# create superposition vector, test recall time using dot product (with non-binary values)
	sv = np.sum(b1b, axis=0)
	sv1 = b1b[0]
	reps = 100
	sv = np.tile(sv, reps)
	sv1 = np.tile(sv1, reps)
	im_address_bipolar = np.tile(im_address_bipolar, reps)

	# dot_distances2 = np.dot(b2b, im_address_bipolar)


	# time dot with sum vector
	start_time = time.perf_counter_ns()
	s_timer = timer()
	dot_sv_distances = np.dot(sv, im_address_bipolar)
	dot_sv_timer = timer() - s_timer
	dot_sv_time = time.perf_counter_ns() - start_time

	# time dot with binary vector
	start_time = time.perf_counter_ns()
	s_timer = timer()
	dot_bp_distance = np.dot(sv1, im_address_bipolar)
	dot_bp_timer = timer() - s_timer
	dot_bp_time = time.perf_counter_ns() - start_time




	dot1_time_normalized = dot1_time / nrows
	print("dot1_time_normalized=%s, dot_sv_time=%s, dot_sv_time / dot1_time_normalized=%s" %(
		dot1_time_normalized, dot_sv_time, dot_sv_time / dot1_time_normalized))
	print("dot_bp_time=%s, dot_sv_time=%s, dot_sv_time/dot_bp_time=%s" %(dot_bp_time, dot_sv_time,
		dot_sv_time/dot_bp_time))
	print("dot_bp_timer=%s, dot_sv_timer=%s, dot_sv_timer/dot_bp_timer=%s" %(dot_bp_timer, dot_sv_timer,
		dot_sv_timer/dot_bp_timer))
	print("sum dot_distances2=%s" % np.sum(dot_distances2))



def test_timing(nrows=1000, ncols=512, nact=3, method="hamming"):
	if method == "hamming":
		im_address = rng.integers(0, high=2, size=(self.nrows, self.ncols), dtype=np.int8)
		address = rng.integers(0, high=2, size=(self.ncols), dtype=np.int8)
		start_time = time.perf_counter_ns()
		hl_match = np.count_nonzero(address!=im_address, axis=1)
		transition_hard_locations = np.argpartition(hl_match, self.nact)[0:self.nact]
		recall_time = time.perf_counter_ns() - start_time
	else:
		assert method == "dot"





def main():
	# update_recall_times()
	# test_timing()
	compare_operations()


main()

