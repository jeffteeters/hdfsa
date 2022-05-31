

# script to calculate error rate for particular transitions being stored

import itertools

def calc_result(bun, s0, s2, a0, a2, p0, p1):
	read1 = "match" if s0^a0^bun == p0 else "fail"
	read2 = "match" if s0^a2^bun == p1 else "fail"
	read3 = "match" if s2^a0^bun == p1 else "fail"
	read4 = "match" if s2^a2^bun == p0 else "fail"
	result = "%s %s %s %s" % (read1, read2, read3, read4)
	fail_count = result.count("fail")
	return (result, fail_count)
	return result


possible_options = list(itertools.product([0, 1], repeat=6))  # bit patterns
error_count = 0
for opt in possible_options:
	s0, s2, a0, a2, p0, p1 = opt
	term1 = s0^a0^p0
	term2 = s0^a2^p1
	term3 = s2^a0^p1
	term4 = s2^a2^p0
	bundle_sum = (term1*2-1)+(term2*2-1)+(term3*2-1)+(term4*2-1)
	if bundle_sum == 0:
		(result_0, fail_count_0) = calc_result(0, s0, s2, a0, a2, p0, p1)
		(result_1, fail_count_1) = calc_result(1, s0, s2, a0, a2, p0, p1)
		result = "sum is zero; if bun=0, %s; if bun=1, %s" % (result_0, result_1)
		error_count += (fail_count_0 + fail_count_1) / 2
	else:
		bun = 1 if bundle_sum > 0 else 0
		(result, fail_count) = calc_result(bun, s0, s2, a0, a2, p0, p1)
		error_count += fail_count
	print("s0-%s, s2-%s, a0-%s, a2-%s, p0-%s, p1-%s; result = %s" % (s0, s2, a0, a2, p0, p1, result))
	print("error_count=%s, error_rate=%s" % (error_count, (error_count / (len(possible_options) * 4))))
