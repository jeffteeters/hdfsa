# script to find sdm dimensions for various number of items

import math
from scipy.stats import norm
from scipy import special
import sys

def sdm_size(m, n):
	# m - number rows, n - number of columns in sdm
	size = m * (1 + 1/8) * n
	return size

def single_bit_error_rate(m, k):
	# m - number of rows, k - number of items stored
	p = 1.0 / ((2*m*k)**(1/3))  # optimized activation count for sdm
	mean = p * m
	std = math.sqrt(p*m*(1. + p*k + (1. + p*p*m)))
	delta = norm.cdf(0, loc=-mean, scale=std)
	return delta

def dplen(delta, per):
	# calculate vector length requred to store bundle at per accuracy
	# delta - mean of match distribution (single bit error rate, 0< delta < 0.5)
	# per - desired probability of error on recall (e.g. 0.000001)
	n = (-2*(-0.25 - delta + delta**2)*special.erfinv(-1 + 2*per)**2)/(0.5 - delta)**2
	return round(n)

def num_columns(m, k, per):
	# calculate number of columns in sdm to attain per probability error on recall of single item
	# m - number of rows
	# k - number of items stored
	delta = single_bit_error_rate(m, k)
	num_cols = dplen(delta, per)
	return num_cols

def columns_and_size(m, k, per):
	num_cols = num_columns(m, k, per)
	size = sdm_size(m, num_cols)
	return (num_cols, size)

def next_m(cur_m, cur_n, step, k, per):
	# advance current m in direction step until n (number of columns) changes
	# return new_m, new_n, new_size
	m = cur_m
	n = cur_n
	count = 0
	while n == cur_n:
		m += step
		n, size = columns_and_size(m, k, per)
		count += 1
		if count>100:
			sys.exit("failed to change n in next_m")
	return (m, n, size)


def find_sdm_size(k, per):
	# Find size (# rows and columns) of sdm needed to store k items with final error (perf)
	# per is probability of at least one error when recalling all items
	# This makes an estimate of rows and columns and finds the minimum size sdm with
	# the specified error rate

	m_1 = 2*k  # a guess for current m (number of rows)
	n_1, size_1 = columns_and_size(m_1, k, per)
	m_2, n_2, size_2 = next_m(m_1, n_1, +1, k, per)
	if size_1 < size_2:
		cur_m, cur_n, cur_size = m_2, n_2, size_2
		tst_m, tst_n, tst_size = m_1, n_1, size_1
		step = -1
	else:
		cur_m, cur_n, cur_size = m_1, n_1, size_1
		tst_m, tst_n, tst_size = m_2, n_2, size_2
		step = 1
	print("cur_m=%s, cur_n=%s, cur_size=%s" % (cur_m, cur_n, cur_size))
	print("tst_m=%s, tst_n=%s, tst_size=%s, step=%s" % (tst_m, tst_n, tst_size, step))
	while cur_size > tst_size:
		cur_size = tst_size
		cur_m = tst_m
		cur_n = tst_n
		if step == -1 and cur_m == 1:
			# don't go smaller than m == 1
			break
		tst_m, tst_n, tst_size = next_m(cur_m, cur_n, step, k, per)
		print("trying: m=%s, n=%s, size=%s" % (tst_m, tst_n, tst_size))
	return( (cur_m, cur_n, cur_size))



def main():
	kvals = [1000] # [5, 10, 20, 50, 100, 250, 500, 750, 1000, 2000, 3000]
	desired_percent_errors = [0.1] # [10, 1, .1, .01, .001]
	codebook_sizes = [110] # [3, 36,100, 200, 500, 1000] # [1000]
	for desired_percent_error in desired_percent_errors:
		for codebook_size in codebook_sizes:
			for k in kvals:
				perf = desired_percent_error / 100
				per = 1 - math.exp(math.log(1-perf)/(k* codebook_size))
				m, n, size = find_sdm_size(k, per)
				print("for desired_percent_error=%s%%, codebook_size=%s, k=%s found: m=%s, n=%s, size=%s" % (
					desired_percent_error, codebook_size, k, m, n, size))
				sys.exit("stopping after one for testing.")

main()


