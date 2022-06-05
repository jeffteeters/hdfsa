# make plot of empirical and theoretical match hamming distribution for binarized sdm

import fast_sdm_empirical
import binarized_sdm_analytical

# import sdm_ae as sdm_anl
# import sdm_analytical_jaeckel as sdm_jaeckel
import numpy as np
import matplotlib.pyplot as plt
import sdm_ae as sdm_anl


def main():
	actions=10; states=100; choices=10; d=100; k=1000; ncols=512
	# actions=3; states=3; choices=3; d=3; k=9; ncols=15
	nrows = 200
	nact = 1
	bits_per_counter = 1
	epochs = 10
	fast_emp = fast_sdm_empirical.Fast_sdm_empirical(nrows, ncols, nact, actions=actions,
			hl_selection_method="random", roll_address=True,
			states=states, choices=choices, epochs=epochs, bits_per_counter=bits_per_counter)
	empirical_match_hamming_distribution = fast_emp.match_hamming_distribution
	empirical_distractor_hamming_distribution = fast_emp.distractor_hamming_distribution
	bsm = binarized_sdm_analytical.Binarized_sdm_analytical(nrows, ncols, nact, k, d)
	predicted_match_hamming_distribution = bsm.match_hamming_distribution
	anl = sdm_anl.Sdm_error_analytical(nrows, nact, k, ncols, d)
	xvals = np.arange(ncols+1)
	assert len(xvals) == len(empirical_match_hamming_distribution)
	assert len(xvals) == len(predicted_match_hamming_distribution)
	plt.errorbar(xvals, empirical_match_hamming_distribution, yerr=None, fmt="-o", label="Empirical")
	plt.errorbar(xvals, predicted_match_hamming_distribution, yerr=None, fmt="-o", label="poission prediction")
	plt.errorbar(xvals, empirical_distractor_hamming_distribution, yerr=None, fmt="-o", label="Empirical distractor")
	plt.errorbar(xvals, anl.hdist, yerr=None, fmt="-o", label="sdm_ae prediction")
	
	empirical_distractor_hamming_distribution
	title = "Found and predicted match hamming distribution nrows=%s, nact=%s, ncols=%s" % (nrows, nact, ncols)
	plt.title(title)
	plt.xlabel("hamming distance")
	plt.ylabel("fraction present")
	plt.grid()
	plt.legend(loc="upper right")
	plt.show()

	# plot probability and error distributions from binarized and sdm_ae
	num_vals = 100
	xvals = np.arange(num_vals)
	plt.errorbar(xvals, bsm.prob_overlap[0:num_vals], yerr=None, fmt="-o", label="bsm prob_overlap")
	# import pdb; pdb.set_trace()
	plt.errorbar(xvals, anl.cop_prb[0:num_vals], yerr=None, fmt="-o", label="sdm_ae cop_prb")
	plt.title("probability overlaps")
	plt.legend(loc="lower right")
	plt.show()
	plt.errorbar(xvals, bsm.delt_overlap[0:num_vals], yerr=None, fmt="-o", label="bsm delt_overlap")
	plt.errorbar(xvals, anl.cop_err[0:num_vals], yerr=None, fmt="-o", label="sdm_ae cop_err")
	plt.title("error with number overlaps")
	plt.legend(loc="lower right")
	plt.show()


if __name__ == "__main__":
	# test sdm and bundle
	main()
