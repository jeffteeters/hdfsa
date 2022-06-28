# script to find equation fitting relationship between
# number of rows and error rate for different versions of
# sdm and for bundle


import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.text import Text
# from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy import special
import numpy as np
import sympy as sym

mdims = {
	"bun_k1000_d100":
		[[1,24002],
		[2, 40503],
		[3, 55649],
		[4, 70239],
		[5, 84572],
		[6, 98790],
		[7, 112965],
		[8, 127134],
		[9, 141311]],
	"bun_k750_d100":
		# Bundle sizes, for k=750, d=100:
		[[1, 18012],
		[2, 30389],
		[3, 41750],
		[4, 52693],
		[5, 63443],
		[6, 74108],
		[7, 84740],
		[8, 95367],
		[9, 106002]],
	"bun_k500_d100":	
		# 	Bundle sizes, for k=500, d=100:
		[[1, 12020],
		[2, 20273],
		[3, 27848],
		[4, 35144],
		[5, 42313],
		[6, 49424],
		[7, 56513],
		[8, 63599],
		[9, 70690]],
	"bun_k250_d100":
		# 	Bundle sizes, for k=250, d=100:
		[[1, 6025],
		[2, 10153],
		[3, 13942],
		[4, 17593],
		[5, 21179],
		[6, 24736],
		[7, 28283],
		[8, 31827],
		[9, 35374]],
	"bun_k125_d100":
		# 	Bundle sizes, for k=125, d=100:
		[[1, 3000],
		[2, 5050],
		[3, 6931],
		[4, 8743],
		[5, 10523],
		[6, 12290],
		[7, 14051],
		[8, 15811],
		[9, 17572]],
	"sdm_k1000_d100_c1": # sdm_binarized_nact_optimum_dims
		[[1, 50, 1,], # 200],  # comment out epochs
		[2, 97, 1,], #  200],
		[3, 158, 3,], #  300],
		[4, 208, 3,], #  400],
		[5, 262, 3,], # 3000],
		[6, 315, 5,], # 10000],
		[7, 368, 5,], # 40000]]
		[8, 424, 7,],
		[9, 476, 7]],
	"sdm_k1000_d100_c8":
		#output from sdm_anl.Sdm_error_analytical:
		# SDM sizes, nact:
		# sdm_8bit_counter_dims = [
		[[1, 51, 1],
		[2, 86, 2],
		[3, 125, 2],
		[4, 168, 2],
		[5, 196, 3],
		[6, 239, 3],
		[7, 285, 3],
		[8, 312, 4],
		[9, 349, 4]]
	}

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
	# into account number of items in bundle (k) and number of other items (n) in item memory?
	per = perf/(k*n)
	return bunlen(k, per)

def cdf_dims(k):
	# calculate width of bundle needed to store k items at different precisions (perr)
	# assuming only one distractor in item memory
	dims = []
	for i in range(1,10):
		per = 10**(-i)
		w = bunlen(k, per)  # width of bundle
		dims.append([i, w])
	return dims


class Line_fit():
	# fit line to size of sdm or bundle

	def __init__(self, mem_type):
		if mem_type not in mdims:
			print("Invalid mem_type: %s" %(mem_type))
			sys.exit("options are: %s" % sorted(mem_type.keys()))
		dims = mdims[mem_type]
		nsizes = len(dims)
		x = np.empty(nsizes, dtype=np.float64)
		y = np.empty(nsizes, dtype=np.float64)
		for i in range(nsizes):
			perr, width = dims[i][0:2]
			assert perr == i + 1
			x[i] = width
			y[i] = 10**(-perr)
		ylog = np.log(y)  # store log of error for regression
		result = linregress(x, y=ylog)
		self.k = np.exp(result.intercept)
		self.m = -result.slope
		self.x = x
		self.y = y
		self.yreg = self.k * np.exp(-self.m *x)
		print("%s for err=k*exp(-m*x), k=%s, m=%s" % (mem_type, self.k, self.m))


def plot_sdm_vs_bundle():
	bundle = Line_fit("bun_k1000_d100")
	bin_sdm = Line_fit("sdm_k1000_d100_c1") #("binarized_sdm")
	full_sdm = Line_fit("sdm_k1000_d100_c8") # ("8bit_counter_sdm")
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
	plt.yscale('log')
	yticks = (10.0**-(np.arange(9.0, 0, -1.0)))
	ylabels = [10.0**(-i) for i in range(9, 0, -1)]
	plt.yticks(yticks, ylabels)
	plt.grid()
	plt.legend(loc='upper right')
	plt.show()

def vary_num_items():
	# plot bundle sizes required for different number of items
	wanted_dims = ["bun_k125_d100", "bun_k250_d100", "bun_k500_d100", "bun_k750_d100", "bun_k1000_d100"]
	colors = cm.rainbow(np.linspace(0, 1, len(wanted_dims)))
	for i in range(len(wanted_dims)):
		dim = wanted_dims[i]
		obs = Line_fit(dim)
		# plt.plot(obs.x, obs.y, 'o', c=colors[i], label="%s data" % dim)
		plt.plot(obs.x, obs.y, 'o', c=colors[i])
		plt.plot(obs.x, obs.yreg, c=colors[i], label=dim)
	finalize_plot("bundle error vs size for different k (num items stored)")

def finalize_plot(title):
	plt.title(title)
	plt.xlabel("bits (bundle width or sdm rows * 512 or sdm rows * 512*8")
	plt.ylabel("Fraction error")
	plt.yscale('log')
	yticks = (10.0**-(np.arange(9.0, 0, -1.0)))
	ylabels = [10.0**(-i) for i in range(9, 0, -1)]
	plt.yticks(yticks, ylabels)
	plt.grid()
	plt.legend(loc='upper right')
	plt.show()

def plot_cdf_dims():
	kvals = [25, 100, 250, 500, 750, 1000]
	colors = cm.rainbow(np.linspace(0, 1, len(kvals)))
	for i in range(len(kvals)):
		k = kvals[i]
		dim = "cdf_k%s_d2" % k
		mdims[dim] = cdf_dims(k)
		print("%s=\n%s" % (dim, mdims[dim]))
		obs = Line_fit(dim)
		# plt.plot(obs.x, obs.y, 'o', c=colors[i], label="%s data" % dim)
		plt.plot(obs.x, obs.y, 'o', c=colors[i])
		plt.plot(obs.x, obs.yreg, c=colors[i], label=dim)
	finalize_plot("cdf error vs size for different k (num items stored)")

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
	plot_type = "cdf_dims" # "vary_num_items" # "plot_sdm_vs_bundle"
	if plot_type == "plot_sdm_vs_bundle":
		plot_sdm_vs_bundle()
	elif plot_type == "vary_num_items":
		vary_num_items()
	elif plot_type == "cdf_dims":
		plot_cdf_dims()
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

