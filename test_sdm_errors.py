# scrip to compare predicted and found errors using match and distractor hamming
# distributions.  Not sure why they are so different and not sure where the
# numbers came from.  Was written Feb 6, 2022.

from scipy.stats import norm
import math
data = {
  'sdm_distractor_hamming_mean': [   11.216644444444444,
									   20.712533333333333,
									   34.36386666666667,
									   30.575634343434345,
									   27.486571717171717,
									   29.487553535353534],
	'sdm_distractor_hamming_stdev': [   6.207766527355593,
										7.278889862276165,
										6.930315856850457,
										4.887149136966791,
										3.7247809551750497,
										3.837606589065071],
	'sdm_error': [63.66, 44.78, 19.2, 5.02, 3.08, 8.9],
	'sdm_match_hamming_mean': [   5.4552,
								  10.3502,
								  18.0756,
								  12.5834,
								  10.447,
								  13.961],
	'sdm_match_hamming_stdev': [   3.610623253685714,
								   4.369642859061041,
								   5.313598734031509,
								   4.0478045140887,
								   3.540070256292083,
								   3.9094418715654835],
	'sim_error': [0.84, 0.52, 0.56, 0.86, 0.54, 0.72]}

def perror(mm, ms, dm, ds):
	# mm - match mean, ms - match stdev, dm - distractor mean, ds - distractor stdev
	# cm - combined mean, cs - combined stdev
	cm = dm - mm
	cs = math.sqrt(ms**2 + ds**2)
	perr = norm.cdf(0, loc=cm, scale=cs)
	return perr * 100.0


def compare():
	global data
	sdm_error = data['sdm_error']
	print("found\tpredicted")
	for i in range(len(sdm_error)):
		found_error = sdm_error[i]
		mm = data['sdm_match_hamming_mean'][i]
		ms = data['sdm_match_hamming_stdev'][i]
		dm = data['sdm_distractor_hamming_mean'][i]
		ds = data['sdm_distractor_hamming_stdev'][i]
		predicted_error = perror(mm, ms, dm, ds)
		print("%s\t%s" % (found_error, predicted_error))

compare()



