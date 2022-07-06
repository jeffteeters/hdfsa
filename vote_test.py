# test "voting" by selected items
# compares error rate of individual counters (binarized) to error rate of sum of counters
# sum of counters (then binarized) has smaller error rate

import numpy as np

rng = np.random.default_rng()
# import pdb; pdb.set_trace()

num_targets = 1000000
nact = 3
num_overlaps = 4
target = np.full(num_targets, -1, dtype=np.int8)
target[1::2] = 1   # set even elements to target 1
counters = np.zeros((num_targets, nact), dtype=np.int16)  # each target get it's own set of counters
target_add = np.repeat(target, nact).reshape((num_targets, nact))
counters += target_add
for i in range(num_overlaps):
	distractor = rng.integers(0, high=2, size=(num_targets, nact), dtype=np.int8)*2-1
	counters += distractor
# Find average error for each counter
tresholded_counters = counters.copy()
tresholded_counters[tresholded_counters>0] = 1
tresholded_counters[tresholded_counters<0] = -1
counter_error_count = np.sum(tresholded_counters !=  target_add)
counter_error_rate = counter_error_count / target_add.size
# find average error rate for "voting" (each group of nact)
counter_sums = np.sum(tresholded_counters, axis=1)
counter_sums[counter_sums>0] = 1
counter_sums[counter_sums<0] = -1
sum_error_count = np.sum(counter_sums !=  target)
sum_error_rate = np.sum(counter_sums !=  target) / target.size

print("counter_errors=%s/%s (%s), sum_errors=%s/%s (%s)" % (counter_error_count, target_add.size, counter_error_rate,
	sum_error_count, target.size, sum_error_rate))

