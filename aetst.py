import numpy as np
import matplotlib.pyplot as plt

import sdm_ee as sdm_emp
import sdm_ae as sdm_anl
import sdm_analytical_jaeckel as sdm_jaeckel
import bundle_analytical
import bundle_empirical
from scipy.stats import norm
from astropy import modeling
import fast_sdm_empirical
import sdm_bc
from scipy.stats import binom



# test sdm analytical vs empirical

def hist_err(hdist, ncols, d):
	# compute overall error rate, by integrating over all match hamming distances with distractor distribution
	# hdist is the match hamming distribution
	# ncols is the number of columns in each row of the sdm
	# d is number of item in item memory
	assert len(hdist) == ncols + 1
	n = ncols
	h = np.arange(len(hdist))
	# distractor_pmf = binom.pmf(h, n, 0.5)
	# self.plot(binom.pmf(h, n, 0.5), "distractor pmf", "hamming distance", "probability")
	ph_corr = binom.sf(h, n, 0.5) ** (d-1)
	# ph_corr = binom.sf(h, n, 0.5) ** (d-1)
	# ph_corr = (binom.sf(h, n, 0.5) + match_hammings_area) ** (d-1)
	# self.plot(ph_corr, "probability correct", "hamming distance", "fraction correct")
	# self.plot(ph_corr * hdist, "p_corr weighted by hdist", "hamming distance", "weighted p_corr")
	# hdist = hdist / np.sum(hdist)  # renormalize to increase due to loss of terms
	p_corr = np.dot(ph_corr, hdist)
	perr = 1 - p_corr
	return perr



def main():
	# nrows=6; ncols=33; nact=2; k=6; d=27
	nrows=6; ncols=33; nact=2; k=6; d=3 # should match 3 states two choices per state
	# nrows=1; ncols=128; nact=1; k=7; d=27
	# nrows=97; ncols=512; nact=2; k=1000; d=100  # size = 50 with bc=3.5
	# nrows=195; ncols=512; nact=3; k=1000; d=100  # size = 100 with bc=3.5
	# nrows=508; ncols=512; nact=5; k=1000; d=100  # size = 100 with bc=3.5
	# nrows=1376; ncols=512; nact=10; k=1000; d=100  # size = 800 with bc=8
	emp = sdm_emp.Sdm_ee(nrows, ncols, nact, k, d, ntrials=100000)
	actions=2; states=3; choices=2
	fast_emp = fast_sdm_empirical.Fast_sdm_empirical(nrows, ncols, nact, actions=actions,
		states=states, choices=choices, epochs=100000)
	bc=8
	sdm_mem = sdm_bc.Sparse_distributed_memory(ncols, nrows, nact, bc)
	sdm_bc_ri= sdm_bc.empirical_response(sdm_mem, actions, states, choices, ntrials=100000)
	anl = sdm_anl.Sdm_error_analytical(nrows, nact, k, ncols, d)

	jaeckel_error = sdm_jaeckel.SdmErrorAnalytical(nrows,k,d,nact,word_length=ncols)
	if nrows == 1:
		bundle_err_analytical = bundle_analytical.BundleErrorAnalytical(ncols,d,k)
		bundle_err_empirical = bundle_empirical.AccuracyEmpirical(ncols,d,k)
		bundle_err_empirical2 = bundle_empirical.bundle_error_empirical(ncols, k, d)
	else:
		bundle_err_analytical = None
		bundle_err_empirical = None
		bundle_err_empirical2 = None

	print("for k=%s, d=%s, sdm size=(%s, %s, %s), bundle_err_analytical=%s, bundle_err_empirical=%s, "
		"bundle_err_empirical2=%s, jaeckel=%s, numerical=%s, fraction=%s, empirical=%s, fast_empirical=%s"
		" sdm_bc_empirical=%s, fast_hist=%s, sdm_bc_hist=%s" % (
			k, d, nrows, ncols, nact, bundle_err_analytical, bundle_err_empirical, bundle_err_empirical2,
			jaeckel_error, anl.perr, anl.perr_fraction, emp.perr, fast_emp.mean_error, sdm_bc_ri["error_rate"],
			hist_err(fast_emp.ehdist, ncols, d), hist_err(sdm_bc_ri["ehdist"], ncols, d)))
	sdm_emp.plot(anl.hdist, "Perdicted match vs distractor hamming distribution", "hamming distance",
				"relative frequency", label="match", data2=anl.distractor_pmf, label2="distractor")
	sdm_emp.plot(anl.hdist, "Perdicted vs empirical match hamming distribution", "hamming distance",
				"relative frequency", label="predicted", data2=emp.ehdist, label2="found")
	sdm_emp.plot(anl.hdist, "Perdicted vs fast empirical match hamming distribution", "hamming distance",
				"relative frequency", label="predicted", data2=fast_emp.ehdist, label2="found")
	sdm_emp.plot(anl.hdist, "Perdicted vs sdm_bc empirical match hamming distribution", "hamming distance",
				"relative frequency", label="predicted", data2=sdm_bc_ri["ehdist"], label2="found")
	sdm_emp.plot(fast_emp.ehdist, "fast empirical vs sdm_bc empirical match hamming distribution", "hamming distance",
				"relative frequency", label="Fast empirical", data2=sdm_bc_ri["ehdist"], label2="sdm_bc empirical")
	# try to fit gaussian to perdicted distribution
	# from: https://stackoverflow.com/questions/44480137/how-can-i-fit-a-gaussian-curve-in-python
	# mean,std=norm.fit(anl.hdist)

	# fitter = modeling.fitting.LevMarLSQFitter()
	# model = modeling.models.Gaussian1D()   # depending on the data you need to give some initial values
	# x = np.arange(ncols + 1)
	# fitted_model = fitter(model, x, anl.hdist*(ncols+1))
	# # import pdb; pdb.set_trace()
	# # y = norm.pdf(x, mean, std)
	# sdm_emp.plot(anl.hdist, "Perdicted match hamming distribution vs fitted gaussian", "hamming distance",
	# 			"relative frequency", label="predicted", data2=fitted_model(x), label2="fitted gaussian")


	# if ee.error_rate_vs_hamming is not None:
	# 	plot(ee.error_rate_vs_hamming, "error rate vs hamming distances found", "hamming distance", "error rate")
	# 	print("error_rate_vs_hamming=%s" % error_rate_vs_hamming)
	# print("Bit error rate = %s" % ee.bit_error_rate)
	# print("overall perr=%s" % ee.overall_perr)


if __name__ == "__main__":
	# test sdm and bundle
	main()