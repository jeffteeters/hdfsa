

# script to calculate error rate for particular transitions being stored
import numpy as np
import itertools

def calc_result(bundle_sum, bun, s0a0, s0a2, s2a0, s2a2, p0, p1):
	read1 = "match" if s0a0^bun == p0 else "fail"
	read2 = "match" if s0a2^bun == p1 else "fail"
	read3 = "match" if s2a0^bun == p1 else "fail"
	read4 = "match" if s2a2^bun == p0 else "fail"
	result = "sum=%s, %s %s %s %s" % (bundle_sum, read1, read2, read3, read4)
	fail_count = result.count("fail")
	return (result, fail_count)

sums_found = np.empty(2**6, dtype=int)
sidx = 0

possible_options = list(itertools.product([0, 1], repeat=6))  # bit patterns
error_count = 0
for opt in possible_options:
	s0a0, s0a2, s2a0, s2a2, p0, p1 = opt
	term1 = s0a0^p0
	term2 = s0a2^p1
	term3 = s2a0^p1
	term4 = s2a2^p0
	bundle_sum = (term1*2-1)+(term2*2-1)+(term3*2-1)+(term4*2-1)
	sums_found[sidx] = bundle_sum
	sidx += 1
	if bundle_sum == 0:
		(result_0, fail_count_0) = calc_result(bundle_sum, 0, s0a0, s0a2, s2a0, s2a2, p0, p1)
		(result_1, fail_count_1) = calc_result(bundle_sum, 1, s0a0, s0a2, s2a0, s2a2, p0, p1)
		result = "sum is zero; if bun=0, %s; if bun=1, %s" % (result_0, result_1)
		error_count += (fail_count_0 + fail_count_1) / 2
	else:
		bun = 1 if bundle_sum > 0 else 0
		(result, fail_count) = calc_result(bundle_sum, bun, s0a0, s0a2, s2a0, s2a2, p0, p1)
		error_count += fail_count
	print("s0-%s, s2-%s, a0-%s, a2-%s, p0-%s, p1-%s; result = %s" % (s0a0, s0a2, s2a0, s2a2, p0, p1, result))
print("error_count=%s, error_rate=%s" % (error_count, (error_count / (len(possible_options) * 4))))
assert sums_found.size == sidx
values, counts = np.unique(sums_found, return_counts=True)
print("values=%s, counts=%s" % (values, counts))
