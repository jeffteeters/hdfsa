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
from scipy.stats import norm
import bundle_analytical

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
	# bundle, test varying d
	"bun_k1000_d2":
	# Bundle sizes, for k=1000, d=2:
		[[1, 5239],
		[2, 17084],
		[3, 30087],
		[4, 43541],
		[5, 57236],
		[6, 71081],
		[7, 85027],
		[8, 99047],
		[9, 113122]],
	"bun_k1000_d5":
		# Bundle sizes, for k=1000, d=5:
		[[1, 10697],
		[2, 24217],
		[3, 37877],
		[4, 51643],
		[5, 65503],
		[6, 79443],
		[7, 93449],
		[8, 107510],
		[9, 121616]],
	# Bundle sizes, for k=1000, d=10:
	"bun_k1000_d10":
		[[1, 14059],
		[2, 28399],
		[3, 42427],
		[4, 56380],
		[5, 70342],
		[6, 84340],
		[7, 98383],
		[8, 112467],
		[9, 126590]],
	"bun_k1000_d25":
		# Bundle sizes, for k=1000, d=25:
		[[1, 18147],
		[2, 33407],
		[3, 47886],
		[4, 62084],
		[5, 76182],
		[6, 90260],
		[7, 104351],
		[8, 118466],
		[9, 132610]],
	"bun_k1000_d50":	
		# Bundle sizes, for k=1000, d=50:
		[[1, 21107],
		[2, 37004],
		[3, 51817],
		[4, 66206],
		[5, 80418],
		[6, 94562],
		[7, 108692],
		[8, 122833],
		[9, 136993]],
	"bun_k1000_d75":
		# Bundle sizes, for k=1000, d=75:
		[[1, 22807],
		[2, 39060],
		[3, 54068],
		[4, 68574],
		[5, 82855],
		[6, 97042],
		[7, 111198],
		[8, 125354],
		[9, 139533]],
	"bun_k1000_d125":
		# Bundle sizes, for k=1000, d=125:
		[[1, 24923],
		[2, 41614],
		[3, 56867],
		[4, 71523],
		[5, 85897],
		[6, 100141],
		[7, 114332],
		[8, 128510],
		[9, 142694]],
	"bun_k1000_d150":
		# Bundle sizes, for k=1000, d=150:
		[[1, 25674],
		[2, 42516],
		[3, 57858],
		[4, 72568],
		[5, 86977],
		[6, 101242],
		[7, 115446],
		[8, 129633],
		[9, 143822]],
	"bun_k1000_d175":
		# Bundle sizes, for k=1000, d=175:
		[[1, 26306],
		[2, 43276],
		[3, 58692],
		[4, 73448],
		[5, 87887],
		[6, 102171],
		[7, 116387],
		[8, 130581],
		[9, 144774]],
	"bun_k1000_d200":
		# Bundle sizes, for k=1000, d=200:
		[[1, 26852],
		[2, 43933],
		[3, 59412],
		[4, 74209],
		[5, 88674],
		[6, 102974],
		[7, 117201],
		[8, 131401],
		[9, 145598]],
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

def gal_dims(d, k):
	# calculate dims (width of bundles) needed to store k items with d items in item memory using Galant model
	# (8 bit counters)
	dims = []
	print("d=%s,k=%s" % (d,k))
	for i in range(1,10):
		perf = 10**(-i)
		w = bundle_analytical.gallenf(d, k, perf)
		dims.append([i, w])
	return dims

class Line_fit():
	# fit line to size of sdm or bundle

	def __init__(self, mem_type, bpx=1):
		# bpx is bits per x counter (e.g. 1, 8, 512, 512*8)
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
		self.xbits = x * bpx
		self.y = y
		self.yreg = self.k * np.exp(-self.m *x)
		print("%s for err=k*exp(-m*x), k=%s, m=%s" % (mem_type, self.k, self.m))


def sdm_vs_bundle():
	k = 1000
	d = 100
	mdims["bun_k1000_d100_c8"] = gal_dims(d, k)
	# print("gal_dims=%s" % mdims["bun_k1000_d100_c8"])
	gal_bun = Line_fit("bun_k1000_d100_c8", bpx=8)
	bundle = Line_fit("bun_k1000_d100")
	bin_sdm = Line_fit("sdm_k1000_d100_c1", bpx=512) #("binarized_sdm")
	full_sdm = Line_fit("sdm_k1000_d100_c8", bpx=512*8) # ("8bit_counter_sdm")
	num_wanted_dims = 4
	colors = cm.rainbow(np.linspace(0, 1, num_wanted_dims))
	plt.plot(bundle.xbits, bundle.yreg, c=colors[0], label="bun_k1000_d100")
	plt.plot(bundle.xbits, bundle.y, 'o', c=colors[0])
	plt.plot(bin_sdm.xbits, bin_sdm.yreg, c=colors[1], label="sdm_k1000_d100_c1")
	plt.plot(bin_sdm.xbits, bin_sdm.y, 'o', c=colors[1])
	plt.plot(full_sdm.xbits, full_sdm.yreg, c=colors[2], label="sdm_k1000_d100_c8")
	plt.plot(full_sdm.xbits, full_sdm.y, 'o',c=colors[2])
	plt.plot(gal_bun.xbits, gal_bun.yreg, c=colors[3], label="bun_k1000_d100_c8")
	plt.plot(gal_bun.xbits, gal_bun.y, 'o',c=colors[3])
	plt.title("Bundles and sdm errors vs size (bits)")
	plt.xlabel("bits (in bundle or sdm")
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

def vary_item_memory():
	# plot bundle sizes required for different size of item memory (d)
	wanted_dims = ["bun_k1000_d2", "bun_k1000_d5", "bun_k1000_d10",
		"bun_k1000_d25", "bun_k1000_d50", "bun_k1000_d75", "bun_k1000_d100", "bun_k1000_d125",
		"bun_k1000_d150", "bun_k1000_d175", "bun_k1000_d200"]
	colors = cm.rainbow(np.linspace(0, 1, len(wanted_dims)))
	for i in range(len(wanted_dims)):
		dim = wanted_dims[i]
		obs = Line_fit(dim)
		# plt.plot(obs.x, obs.y, 'o', c=colors[i], label="%s data" % dim)
		plt.plot(obs.x, obs.y, 'o', c=colors[i])
		plt.plot(obs.x, obs.yreg, c=colors[i], label=dim)
	finalize_plot("bundle error vs size with different item memory size (d)")

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
	kvals = [25, 100, 250, 500, 750, 1000, 1500, 2000]
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

def compare_cdf_to_exp_fit():
	def cr(n, f, m):
		return f * np.exp(-m * n)
	def cd(n, k):
		return norm.cdf(-0.4 * math.sqrt(n*2) / math.sqrt((k - 0.76)))
	k = 1000
	e1 = 0.01
	e2 = 10.0**(-6)
	n1 = 16911 # for e=0.01
	# n1 = 29837  # for e=0.001
	n2 = 70576
	m = (np.log(e1) - np.log(e2)) / (n2 - n1)
	f = np.exp(np.log(e1) + m * n1)
	n = np.arange(1, 200, 5) * 1000
	ecr = [cr(i, f, m) for i in n]
	ecd = [cd(i, k) for i in n]
	plt.plot(n, ecr, label="exp_fit")
	plt.plot(n, ecd, label="cdf fit")
	plt.title("cdf vs exp error vs vector width")
	plt.xlabel("bundle width")
	plt.ylabel("Fraction error")
	plt.yscale('log')
	# yticks = (10.0**-(np.arange(9.0, 0, -1.0)))
	# ylabels = [10.0**(-i) for i in range(9, 0, -1)]
	# plt.yticks(yticks, ylabels)
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
	plot_type = "sdm_vs_bundle" # "vary_num_items" # "vary_item_memory" # "compare_cdf_to_exp_fit" # "cdf_dims" #
	if plot_type == "sdm_vs_bundle":
		sdm_vs_bundle()
	elif plot_type == "vary_num_items":
		vary_num_items()
	elif plot_type == "cdf_dims":
		plot_cdf_dims()
	elif plot_type == "compare_cdf_to_exp_fit":
		compare_cdf_to_exp_fit()
	elif plot_type == "vary_item_memory":
		vary_item_memory()
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

