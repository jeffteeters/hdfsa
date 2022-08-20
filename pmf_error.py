# compute pmf_error from dictionary of match_counts and distractor_counts

import numpy as np

def get_counts_pmf(counts):
	# convert counts dictionary (key is distance, value is count at that distance)
	# to numpy array of pmf (probability mass function i.e. normalized counts) and start index
	# convert from dict to array, with zeros in unused distances
	start_index = min(counts.keys())
	end_index = max(counts.keys())
	length = end_index - start_index + 1
	counts_arr = np.zeros(length, dtype=np.int64)
	for key, val in counts.items():
		counts_arr[key - start_index] = val
	counts_pmf = counts_arr / counts_arr.sum()  # normalize to make pmf
	return (start_index, counts_pmf)


def pmf_error(match_counts, distract_counts):
	# compute pmf_error using distribution from match_counts and distractor_counts
	match_start_idx, match_pmf = get_counts_pmf(match_counts)
	distract_start_idx, distract_pmf = get_counts_pmf(distract_counts)
	if distract_start_idx <= match_start_idx:
	# distract_cdf are all distract values less then or equal to current match value
		end_idx = min(match_start_idx - distract_start_idx + 1, distract_pmf.size)
		distract_cdf = distract_pmf[0:end_idx].sum()
	else:
		distract_cdf = 0
	distract_match_offset = distract_start_idx - match_start_idx
	distract_0_based_idx = -distract_match_offset
	p_err = 0.0
	for i in range(0, len(match_pmf)):
		p_err += match_pmf[i] * distract_cdf
		distract_0_based_idx += 1
		if distract_0_based_idx >= 0 and distract_0_based_idx < distract_pmf.size:
			distract_cdf += distract_pmf[distract_0_based_idx]
	assert p_err >= 0 and p_err <= 1
	return p_err
