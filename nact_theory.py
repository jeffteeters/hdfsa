# script to calculate optimal nact using theory of overlaps and voting
# This used for binary SDM, each counter is binarized and votes.
# Unsure of meaning of output for larger number rows because at least three
# counters are required to vote and nact==1 output is always 0.25

from scipy.stats import binom
import scipy


def nact_vote(nact, perr):
	# calculate new probability of error given voting by nact counters, each with probability error perr
	p_corr = 1.0 - perr
	p_err_new = binom.cdf((nact-1)/2, nact, p_corr)
	return p_err_new


#Compute expected Hamming distance
def  expectedHamming (K):
    if (K % 2) == 0: # If even number then break ties so add 1
        K+=1
    deltaHam = 0.5 - (scipy.special.binom(K-1, 0.5*(K-1)))/2**K  # Pentti's formula for the expected Hamming distance
    return deltaHam

def main():
	k = 1000
	for nrows in range(50, 2000, 50):
		print("nrows=%s" % nrows)
		for nact in range(1,13,2):
			ave_overlaps = max(int(k*nact / nrows), 3)  # make sure at least three
			eham = expectedHamming(ave_overlaps)
			vham = nact_vote(nact, eham)
			print("nact=%s, eham=%s, vham=%s" % (nact, eham, vham))


	# for n
	# perr = 0.4
	# print("original perr=%s" % perr)
	# print("nact\tnew_perr")
	# for nact in range(3,14,2):
	# 	print("%s\t%s" % (nact, nact_vote(nact, perr)))


if __name__ == "__main__":
	main()


