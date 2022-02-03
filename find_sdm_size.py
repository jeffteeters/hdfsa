# script to find sdm dimensions for various number of items

import math
from scipy.stats import norm
from scipy import special
import sys
from overlap import sdm_delta_empirical

def system_size(m, n, codebook_length):
	# m - number rows, n - number of columns in sdm
	sdm_size = m * (1 + 1/8) * n
	item_memory_size = codebook_length * n / 8
	return sdm_size + item_memory_size

def fraction_rows_activated(m, k):
	# compute optimal fraction rows to activate in sdm
	# m is number rows, k is number items stored in sdm
	return 1.0 / ((2*m*k)**(1/3))

def single_bit_error_rate_wrong(m, k):
	# m - number of rows, k - number of items stored
	p = fraction_rows_activated(m, k)  # optimized activation count for sdm
	mean = p * m
	std = math.sqrt(p*m*(1. + p*k + (1. + p*p*m)))
	delta = norm.cdf(0, loc=mean, scale=std)
	nact = round(mean)
	return (delta, nact)

def single_bit_error_rate(m, k):
	# m - number of rows, k - number of items stored
	p = fraction_rows_activated(m, k)  # optimized activation count for sdm
	nact = round(p * m)
	mean = nact
	p = nact/m    # modify p to actual value
	std = math.sqrt(p*m*(1. + p*k + (1. + p*p*m)))
	delta = norm.cdf(0, loc=mean, scale=std)
	return (delta, nact)

def dplen(delta, per):
	# calculate vector length requred to store bundle at per accuracy
	# delta - mean of match distribution (single bit error rate, 0< delta < 0.5)
	# per - desired probability of error on recall (e.g. 0.01)
	n = (-2*(-0.25 - delta + delta**2)*special.erfinv(-1 + 2*per)**2)/(0.5 - delta)**2
	return n # round(n)

def num_columns(m, k, per):
	# calculate number of columns in sdm to attain per probability error on recall of single item
	# m - number of rows
	# k - number of items stored
	delta, nact = single_bit_error_rate(m, k)
	# empirical_delta = sdm_delta_empirical(round(m), round(k), nact)
	# num_cols = dplen(empirical_delta, per)
	num_cols = dplen(delta, per)
	return (num_cols, nact)

def columns_and_size(m, k, per, codebook_length):
	num_cols, nact = num_columns(m, k, per)
	size = system_size(m, num_cols, codebook_length)
	return (num_cols, nact, size)

def next_m(cur_m, cur_n, step, k, per, codebook_length):
	# advance current m in direction step until n (number of columns) changes
	# return new_m, new_n, new_size
	m = cur_m
	n = cur_n
	count = 0
	while n == cur_n:
		m += step
		n, nact, size = columns_and_size(m, k, per, codebook_length)
		count += 1
		if count>100:
			sys.exit("failed to change n in next_m")
	return (m, n, nact, size)


def find_optimal_sdm_2d_search(k, per, codebook_length):
	# Find size (# rows and columns) of sdm needed to store k items with final error (perf)
	# per is probability of at least one error when recalling all items
	# This makes an estimate of rows and columns and finds the minimum size sdm with
	# the specified error rate

	m_1 = k  # a guess for current m (number of rows)
	n_1, nact_1, size_1 = columns_and_size(m_1, k, per, codebook_length)
	m_2, n_2, nact_2, size_2 = next_m(m_1, n_1, +1, k, per, codebook_length)
	if size_1 < size_2:
		cur_m, cur_n, cur_nact, cur_size = m_2, n_2, nact_2, size_2
		tst_m, tst_n, tst_nact, tst_size = m_1, n_1, nact_1, size_1
		step = -1
	else:
		cur_m, cur_n, cur_nact, cur_size = m_1, n_1, nact_1, size_1
		tst_m, tst_n, tst_nact, tst_size = m_2, n_2, nact_2, size_2
		step = 1
	# print("cur_m=%s, cur_n=%s, cur_size=%s" % (cur_m, cur_n, cur_size))
	# print("tst_m=%s, tst_n=%s, tst_size=%s, step=%s" % (tst_m, tst_n, tst_size, step))
	while cur_size > tst_size:
		cur_size = tst_size
		cur_m = tst_m
		cur_n = tst_n
		cur_nact = tst_nact
		if step == -1 and cur_m == 1:
			# don't go smaller than m == 1
			break
		tst_m, tst_n, tst_nact, tst_size = next_m(cur_m, cur_n, step, k, per, codebook_length)
		# print("trying: m=%s, n=%s, size=%s" % (tst_m, tst_n, tst_size))
	return( (round(cur_m), round(cur_n), cur_nact, round(cur_size)))


def find_optimal_sdm_search_up(k, per, codebook_length):
	# Find size (# rows and columns) of sdm needed to store k items with final error (perf)
	# per is probability of at least one error when recalling all items
	# This makes an estimate of rows and columns and finds the minimum size sdm with
	# the specified error rate

	cur_m = 1  # a guess for current m (number of rows)
	cur_n, cur_size = columns_and_size(cur_m, k, per, codebook_length)
	tst_m, tst_n, tst_size = next_m(cur_m, cur_n, +1, k, per, codebook_length)
	step = 1
	while cur_size > tst_size:
		cur_size = tst_size
		cur_m = tst_m
		cur_n = tst_n
		tst_m, tst_n, tst_size = next_m(cur_m, cur_n, step, k, per, codebook_length)
		# print("trying: m=%s, n=%s, size=%s" % (tst_m, tst_n, tst_size))
	return( (cur_m, cur_n, cur_size))

def bunlen(k, per):
	# calculated bundle length needed to store k items with accuracy per
	# This calculates the mean distance to the matching vector using
	# approximation on page 3 in Pentti's paper (referenced above)
	# then calls dplen to calculate the vector length.
	return dplen(0.5 - 0.4 / math.sqrt(k - 0.44), per)

def find_bundle_size(k, per, codebook_length):
	# find superposition vector length (bundle length) and total size for superposition system
	bl = bunlen(k, per)
	bsize = bl * codebook_length
	return (bl, bsize)

def main():
	kvals = [5, 10, 20, 50, 100, 250, 500, 750, 1000, 2000, 3000]
	desired_percent_errors = [0.01] # [10, 1, .1, .01, .001]  # 0.01] # 
	codebook_lengths = [10, 36, 110, 200, 500, 1000, 2000, 3000]
	for desired_percent_error in desired_percent_errors:
		for codebook_length in codebook_lengths:
			for k in kvals:
				perf = desired_percent_error / 100
				per = 1 - math.exp(math.log(1-perf)/(k* codebook_length))
				m, n, nact, size = find_optimal_sdm_2d_search(k, per, codebook_length)
				bl, bsize = find_bundle_size(k, per, codebook_length)
				# print("desired_percent_error=%s%%, codebook_length=%s, k=%s found: m=%s, n=%s, size=%s" % (
				print("p_error=%s%%, code_book_len D=%s, num_items K=%s found: m=%s, n=%s, nact=%s, size=%s; bl=%s, bsize=%s, sdmsize/bsize=%s" % (
					desired_percent_error, codebook_length, k, round(m), round(n), nact, round(size),
					round(bl), round(bsize), size/bsize))
				# sys.exit("stopping after one for testing.")


if __name__ == "__main__":
	# test sdm and bundle
	main()
