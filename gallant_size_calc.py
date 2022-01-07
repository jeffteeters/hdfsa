from scipy import special
import math

def gallen(s, per):
	# calculate required length using equation in Gallant paper
	# solved to give length.  (Page 39: "This is equivalent to the probability
	# of a random vector ["per"] being negative when selected from
	# NormalDistibution(D, √((2S-1)D)" solved for D in terms of per and s 
	return round(2*(-1 + 2*s)*special.erfcinv(2*per)**2)

def dplen(mm, per):
	# calculate vector length requred to store bundle at per accuracy
	# mm - mean of match distribution (single bit error rate, 0< mm < 0.5)
	# per - desired probability of error on recall (e.g. 0.000001)
	# This derived from equations in Pentti's 1997 paper, "Fully Distributed
	# Representation", page 3, solving per = probability random vector has
	# smaller hamming distance than hamming to matching vector, (difference
	# in normal distributions) is less than zero; solve for length (N)
	# in terms of per and mean distance to match (mm, denoted as delta in the
	# paper.)
	n = (-2*(-0.25 - mm + mm**2)*special.erfinv(-1 + 2*per)**2)/(0.5 - mm)**2
	return round(n) + 1

def bunlen(k, per):
	# calculated bundle length needed to store k items with accuracy per
	# This calculates the mean distance to the matching vector using
	# approximation on page 3 in Pentti's paper (referenced above)
	# then calls dplen to calculate the vector length.
	return dplen(0.5 - 0.4 / math.sqrt(k - 0.44), per)

def bunlenf(k, perf, n):
	# bundle length from final probabability error (perf) taking
	# into account number of items in bundle (k) and number of other items (n)
	per = perf/(k*n)
	return bunlen(k, per)

def gallenf(k, perf, n):
	# bundle length from formula in Gallant paper from final probability error
	# (perf) taking into account number of items in bundle (k) and number of other items (n)
	per = perf/(k*n)
	return gallen(k, per)

# calculate vector lengths for cases given in Gallant paper, and also using
# equations for binary vectors from Pentti's 1997 paper, "Fully Distributed Representation" 
# S-number of items in bundle, N-number of other items, perf-desired percent error
cases = [{"label":"Small", "S":20, "N":1000, "perf":1.6},
	{"label":"Medium", "S":100, "N":100000, "perf":1.8},
	{"label":"Large", "S":1000, "N":1000000, "perf":1.0}]

for case in cases:
	dgal = gallenf(case["S"], case["perf"]/100.0, case["N"])
	dbun = bunlenf(case["S"], case["perf"]/100.0, case["N"])
	print("%s (S=%s, N=%s, perf=%s), D is: Gallant: %s, bunlen: %s, ratio: %s" % (
		case["label"],case["S"],case["N"],case["perf"], dgal, dbun, round(dbun / dgal,3)))
