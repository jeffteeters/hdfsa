import numpy as np
import matplotlib.pyplot as plt

import sdm_ee as sdm_emp
import sdm_ae as sdm_anl
import sdm_analytical_jaeckel as sdm_jaeckel
import bundle_analytical
import bundle_empirical



# test sdm analytical vs empirical

def main():
	# nrows=6; ncols=33; nact=2; k=6; d=27
	nrows=1; ncols=128; nact=1; k=7; d=27
	emp = sdm_emp.Sdm_ee(nrows, ncols, nact, k, d)
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
		"bundle_err_empirical2=%s, jaeckel=%s, numerical=%s, empirical=%s" % (k, d, nrows, ncols, nact,
		bundle_err_analytical, bundle_err_empirical, bundle_err_empirical2, jaeckel_error, anl.perr, emp.perr))
	sdm_emp.plot(anl.hdist, "Perdicted vs empirical match hamming distribution", "hamming distance",
				"relative frequency", label="predicted", data2=emp.ehdist, label2="found")

	# if ee.error_rate_vs_hamming is not None:
	# 	plot(ee.error_rate_vs_hamming, "error rate vs hamming distances found", "hamming distance", "error rate")
	# 	print("error_rate_vs_hamming=%s" % error_rate_vs_hamming)
	# print("Bit error rate = %s" % ee.bit_error_rate)
	# print("overall perr=%s" % ee.overall_perr)


if __name__ == "__main__":
	# test sdm and bundle
	main()