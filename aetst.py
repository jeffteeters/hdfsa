import numpy as np
import matplotlib.pyplot as plt

import sdm_ee as sdm_emp
import sdm_ae as sdm_anl


# test sdm analytical vs empirical

def main():
	nrows=6; ncols=33; nact=2; k=6; d=27
	emp = sdm_emp.Sdm_ee(nrows, ncols, nact, k, d)
	anl = sdm_anl.Sdm_error_analytical(nrows, nact, k, ncols, d)

	print("for k=%s, d=%s, sdm size=(%s, %s, %s), error analytical=%s, empirical=%s" % (k, d, nrows, ncols, nact,
		anl.perr, emp.perr))
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