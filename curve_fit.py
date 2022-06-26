# script to find equation fitting relationship between
# number of rows and error rate for different versions of
# sdm and for bundle


# import math
import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
from scipy.stats import linregress
import numpy as np
import sympy as sym


bundle_dims = [
	[1,24002],
	[2, 40503],
	[3, 55649],
	[4, 70239],
	[5, 84572],
	[6, 98790],
	[7, 112965],
	[8, 127134],
	[9, 141311],
	]

sdm_binarized_nact_optimum_dims = [
	[1, 50, 1,], # 200],  # comment out epochs
	[2, 97, 1,], #  200],
	[3, 158, 3,], #  300],
	[4, 208, 3,], #  400],
	[5, 262, 3,], # 3000],
	[6, 315, 5,], # 10000],
	[7, 368, 5,]] # 40000]]

#output from sdm_anl.Sdm_error_analytical:
# SDM sizes, nact:
sdm_8bit_counter_dims = [
	[1, 51, 1],
	[2, 86, 2],
	[3, 125, 2],
	[4, 168, 2],
	[5, 196, 3],
	[6, 239, 3],
	[7, 285, 3],
	[8, 312, 4],
	[9, 349, 4]]

class Line_fit():
	# fit line to size of sdm or bundle

	def __init__(self, mem_type):
		assert mem_type in ("bundle", "binarized_sdm", "8bit_counter_sdm")  # add other options here
		if mem_type == "bundle":
			x = np.empty(len(bundle_dims), dtype=np.float64)
			y = np.empty(len(bundle_dims), dtype=np.float64)
			for i in range(len(bundle_dims)):
				perr, width= bundle_dims[i]
				assert perr == i + 1
				x[i] = width
				y[i] = 10**(-perr)

		elif mem_type in ("binarized_sdm","8bit_counter_sdm"):
			sdm_dims = sdm_binarized_nact_optimum_dims if mem_type == "binarized_sdm" else sdm_8bit_counter_dims
			x = np.empty(len(sdm_dims), dtype=np.float64)
			y = np.empty(len(sdm_dims), dtype=np.float64)
			for i in range(len(sdm_dims)):
				perr, nrows, nact = sdm_dims[i]
				assert perr == i + 1
				x[i] = nrows
				y[i] = 10**(-perr)
		else:
			sys.exit("invalid dims specified: %s" % dims_using)
		ylog = np.log(y)  # store log of error for regression

		result = linregress(x, y=ylog)
		# print("%s linregress slope=%s, intercept=%s, exp intercept=%s" % (mem_type,
		# 	result.slope, result.intercept, np.exp(result.intercept)))
		k = np.exp(result.intercept)
		m = -result.slope
		self.x = x
		self.y = y
		self.yreg = k * np.exp(-m *x)
		self.k = k
		self.m = m
		print("%s for err=k*exp(-m*x), k=%s, m=%s" % (mem_type, k, m))


def plot_sdm_vs_bundle():
	bundle = Line_fit("bundle")
	bin_sdm = Line_fit("binarized_sdm")
	full_sdm = Line_fit("8bit_counter_sdm")
	bin_sdm_bits = bin_sdm.x * 512  # bits per row
	full_sdm_bits = full_sdm.x * 512 * 8

	plt.plot(bundle.x, bundle.y, 'ro',label="Bundle data")
	plt.plot(bundle.x, bundle.yreg,label="Bundle Line_fit")
	plt.plot(bin_sdm_bits, bin_sdm.y, 'go',label="1-bit sdm data")
	plt.plot(bin_sdm_bits, bin_sdm.yreg,label="1-bit sdm Line_fit")
	plt.plot(full_sdm_bits, full_sdm.y, 'bo',label="8-bit sdm data")
	plt.plot(full_sdm_bits, full_sdm.yreg,label="8-bit sdm Line_fit")
	plt.title("Bundle vs 1-bit sdm vs 8-bit sdm error vs size (bits)")
	plt.xlabel("bits (bundle width or sdm rows * 512 or sdm rows * 512*8")
	plt.ylabel("Fraction error")
	plt.yticks(10.0**-(np.arange(9.0, 0.0, -1.0)))
	plt.yscale('log')
	plt.grid()
	plt.legend(loc='upper right')
	plt.show()


def plot_single_fit(mem_type):
	sys.exit("not implemented")
	plt.plot(x, y, 'ro',label="Original Data")
	plt.plot(x, yreg, 'g-', label="linregress")

	plt.title("found vs linregress for %s" % dims_using)

	plt.yscale('log')
	plt.grid()
	plt.legend(loc='upper right')
	plt.show()

def main():
	plot_type = "plot_sdm_vs_bundle"
	if plot_type == "plot_sdm_vs_bundle":
		plot_sdm_vs_bundle()
	else:
		sys.exit("plot_type %s not implemented" % plot_type)



if __name__ == "__main__":
	# test sdm and bundle
	main()


# def bine(x):
# 	m = 0.04344500175460464
# 	k=0.8778013135972488
# 	return k * np.exp(-m*x)

# yb = [bine(nrows) for nrows in x]
# plt.plot(x, y, 'g-', label="bine calculated")

# function to optomize
# def func(x, k, m):
# 	return k + np.exp(-m * x)

# popt, pcov = curve_fit(func, x, y, p0=[0.8778013135972488, 0.04344500175460464])

# print("popt=%s" % popt)

