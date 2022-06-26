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

dims_using = "8bit_counter_sdm" # "bundle"
assert dims_using in ("bundle", "binarized_sdm", "8bit_counter_sdm")  # add other options here

if dims_using == "bundle":
	x = np.empty(len(bundle_dims), dtype=np.float64)
	y = np.empty(len(bundle_dims), dtype=np.float64)
	for i in range(len(bundle_dims)):
		perr, width= bundle_dims[i]
		assert perr == i + 1
		x[i] = width
		y[i] = 10**(-perr)

elif dims_using in ("binarized_sdm","8bit_counter_sdm"):
	sdm_dims = sdm_binarized_nact_optimum_dims if dims_using == "binarized_sdm" else sdm_8bit_counter_dims
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

print("linregress slope=%s, intercept=%s, exp intercept=%s" % (result.slope, result.intercept, np.exp(result.intercept)))

k = np.exp(result.intercept)
m = -result.slope
yreg = k * np.exp(-m *x)
print("for err=k*exp(-m*x), k=%s, m=%s" % (k, m))


"""
Make plots
"""
plt.plot(x, y, 'ro',label="Original Data")
plt.plot(x, yreg, 'g-', label="linregress")

plt.title("found vs linregress for %s" % dims_using)

plt.yscale('log')
plt.grid()
plt.legend(loc='upper right')
plt.show()



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

