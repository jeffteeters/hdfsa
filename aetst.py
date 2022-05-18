import numpy as np
import matplotlib.pyplot as plt

import sdm_ee as sdm_emp
import sdm_ae as sdm_anl
import sdm_analytical_jaeckel as sdm_jaeckel
import bundle_analytical
import bundle_empirical
from scipy.stats import norm
from astropy import modeling



# test sdm analytical vs empirical

def main():
	# nrows=6; ncols=33; nact=2; k=6; d=27
	# nrows=1; ncols=128; nact=1; k=7; d=27
	# nrows=97; ncols=512; nact=2; k=1000; d=100  # size = 50 with bc=3.5
	# nrows=195; ncols=512; nact=3; k=1000; d=100  # size = 100 with bc=3.5
	# nrows=508; ncols=512; nact=5; k=1000; d=100  # size = 100 with bc=3.5
	nrows=1376; ncols=512; nact=10; k=1000; d=100  # size = 800 with bc=8
	emp = sdm_emp.Sdm_ee(nrows, ncols, nact, k, d, ntrials=100000)
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
		"bundle_err_empirical2=%s, jaeckel=%s, numerical=%s, fraction=%s, empirical=%s" % (k, d, nrows, ncols, nact,
		bundle_err_analytical, bundle_err_empirical, bundle_err_empirical2, jaeckel_error, anl.perr, anl.perr_fraction, emp.perr))
	sdm_emp.plot(anl.hdist, "Perdicted match vs distractor hamming distribution", "hamming distance",
				"relative frequency", label="match", data2=anl.distractor_pmf, label2="distractor")
	sdm_emp.plot(anl.hdist, "Perdicted vs empirical match hamming distribution", "hamming distance",
				"relative frequency", label="predicted", data2=emp.ehdist, label2="found")
	# try to fit gaussian to perdicted distribution
	# from: https://stackoverflow.com/questions/44480137/how-can-i-fit-a-gaussian-curve-in-python
	# mean,std=norm.fit(anl.hdist)
	fitter = modeling.fitting.LevMarLSQFitter()
	model = modeling.models.Gaussian1D()   # depending on the data you need to give some initial values
	x = np.arange(ncols + 1)
	fitted_model = fitter(model, x, anl.hdist*(ncols+1))
	# import pdb; pdb.set_trace()
	# y = norm.pdf(x, mean, std)
	sdm_emp.plot(anl.hdist, "Perdicted match hamming distribution vs fitted gaussian", "hamming distance",
				"relative frequency", label="predicted", data2=fitted_model(x), label2="fitted gaussian")


	# if ee.error_rate_vs_hamming is not None:
	# 	plot(ee.error_rate_vs_hamming, "error rate vs hamming distances found", "hamming distance", "error rate")
	# 	print("error_rate_vs_hamming=%s" % error_rate_vs_hamming)
	# print("Bit error rate = %s" % ee.bit_error_rate)
	# print("overall perr=%s" % ee.overall_perr)


if __name__ == "__main__":
	# test sdm and bundle
	main()