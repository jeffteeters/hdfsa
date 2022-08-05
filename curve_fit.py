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
	"bun_k1000_d100_c1#S1":
		[[1,24002],
		[2, 40503],
		[3, 55649],
		[4, 70239],
		[5, 84572],
		[6, 98790],
		[7, 112965],
		[8, 127134],
		[9, 141311]],

	"bun_k1000_d100_c8#S2":
	[[1, 15221],
	[2, 25717],
	[3, 35352],
	[4, 44632],
	[5, 53750],
	[6, 62794],
	[7, 71812],
	[8, 80824],
	[9, 89843]],

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
	"sdm_k1000_d100_c1_ham#A2": # sdm_binarized_nact_optimum_dims
		[[1, 50, 1,], # 200],  # comment out epochs
		[2, 97, 1,], #  200],
		[3, 158, 3,], #  300],
		[4, 208, 3,], #  400],
		[5, 262, 3,], # 3000],
		[6, 315, 5,], # 10000],
		[7, 368, 5,], # 40000]]
		[8, 424, 7,],
		[9, 476, 7]],
	"sdm_k1000_d100_c8_ham#A1":
		# from run with prune threshold 10m, nact optimum
		[[1, 51, 1],
		[2, 86, 2],
		[3, 125, 2],
		[4, 168, 2],
		[5, 196, 3],
		[6, 238, 3],
		[7, 285, 3],
		[8, 311, 4],
		[9, 356, 4]],
		#output from sdm_anl.Sdm_error_analytical:
		# SDM sizes, nact:
		# sdm_8bit_counter_dims = 
		# from file: sdm_c8_ham_thres1m.txt
		# [[1, 50, 2], # 51, 1],
		# [2, 86, 2],
		# [3, 121, 3], # 125, 2],
		# [4, 157, 3], # 168, 2],
		# [5, 192, 4], # #196, 3],
		# [6, 228, 5], #239, 3],
		# [7, 265, 5], # 285, 3],
		# [8, 303, 5], # 312, 4],
		# [9, 340, 6]], # 349, 4]],
		# original
		# [[1, 51, 1],
		# [2, 86, 2],
		# [3, 125, 2],
		# [4, 168, 2],
		# [5, 196, 3],
		# [6, 239, 3],
		# [7, 285, 3],
		# [8, 312, 4],
		# [9, 349, 4]],
	"sdm_k1000_d100_c1_dot#A3":
		# from run with prune threshold 10m, nact optimum; same as above
		[[1, 51, 1],
		[2, 86, 2],
		[3, 125, 2],
		[4, 168, 2],
		[5, 196, 3],
		[6, 238, 3],
		[7, 285, 3],
		[8, 311, 4],
		[9, 356, 4]],
	# from empirical_size, non-thresholded sum, binarized counters
		# new, same as above
		# [[1, 50, 2], # 51, 1],
		# [2, 86, 2],
		# [3, 121, 3], # 125, 2],
		# [4, 157, 3], # 168, 2],
		# [5, 192, 4], # #196, 3],
		# [6, 228, 5], #239, 3],
		# [7, 265, 5], # 285, 3],
		# [8, 303, 5], # 312, 4],
		# [9, 340, 6]], # 349, 4]],
		# # original
		# [[1, 51, 1],
		# [2, 86, 2],
		# [3, 127, 2],
		# [4, 163, 2],
		# [5, 199, 3],
		# [6, 236, 3],
		# [7, 272, 3],
		# [8, 309, 4],
		# [9, 345, 4]],
	"sdm_k1000_d100_c8_dot#A4":
		# output from 10m prune thrshold, jaeckel optimum nact
		[[1, 31, 1],
		[2, 56, 1],
		[3, 79, 1],
		[4, 101, 2],
		[5, 129, 2],
		[6, 160, 2],
		[7, 177, 3],
		[8, 205, 3],
		[9, 239, 3]],

		# updatted; from sdm_c8_dot_threshold_10M
		# [[1, 30, 1],  # was 1, 31, 2
		# [2, 54, 2],
		# [3, 77, 2],
		# [4, 100, 3],  # was 102, 2]
		# [5, 123, 3],  # 129, 2],  # was 129, 2
		# [6, 147, 4], # was 160, 2], # try with nact=4?
		# [7, 172, 4], # was 178, 3],
		# [8, 198, 5], # was 205, 3],
		# [9, 224, 5]], # was 238, 3]],

	# original
	# "sdm_k1000_d100_c8_dot#A4":
	# 	[[1, 31, 2],
	# 	[2, 54, 2],
	# 	[3, 77, 2],
	# 	[4, 102, 2],
	# 	[5, 129, 2],
	# 	[6, 160, 2], # try with nact=4?
	# 	[7, 178, 3],
	# 	[8, 205, 3],
	# 	[9, 238, 3]],

	# from empirical_size and fast_sdm_empirical
	# Estimated for 5-9.
		# [[1, 31, 1],
  		# [2, 56, 1],
		# [3, 76, 2],
		# [4, 98, 2],
		# [5, 120, 2], # this and following estimated by multiplying above by 0.6
		# [6, 142, 2],
		# [7, 163, 2],
		# [8, 185, 3],
		# [9, 207, 3]]
	}

# from find_sizes, using dot match:
# 	SDM sizes for k=1000, d=100, ncols=512 (dot match)
# SDM size, nact
# original verson
# 1 - 31, 1
# 2 - 56, 1
# 3 - 80, 1
# 4 - 102, 2
# 5 - 129, 2
# 6 - 160, 2
# 7 - 178, 3
# 8 - 205, 3
# 9 - 238, 3

# SDM sizes for k=1000, d=100, ncols=512 (dot match)
# SDM size, nact
# # Not sure why this different than above.  This uses threshold 10,000,000 same as others below.  May be more accurate
# 1 - 30, 1 *
# 2 - 57, 1
# 3 - 87, 1
# 4 - 127, 1
# 5 - 178, 1
# 6 - 248, 1
# 7 - 269, 1
# 8 - 269, 1
# 9 - 269, 1


# desired_error=9, res.x=268.6283015804925, res.success=True, res.message=Solution found.
# SDM sizes for k=1000, d=100, ncols=512 (dot match)
# SDM size, nact
# 1 - 31, 2
# 2 - 54, 2 *
# 3 - 77, 2 *
# 4 - 102, 2
# 5 - 129, 2
# 6 - 160, 2
# 7 - 196, 2
# 8 - 236, 2
# 9 - 269, 2


# desired_error=9, res.x=238.20347314440252, res.success=True, res.message=Solution found.
# SDM sizes for k=1000, d=100, ncols=512 (dot match)
# SDM size, nact
# 1 - 32, 3
# 2 - 56, 3
# 3 - 78, 3
# 4 - 100, 3  **
# 5 - 123, 3  **
# 6 - 150, 3
# 7 - 176, 3  # why 178 in other?  probably threshold for pruning.  this threshold 10,000,000
# 8 - 205, 3
# 9 - 238, 3

# desired_error=9, res.x=225.85647584035365, res.success=True, res.message=Solution found.
# SDM sizes for k=1000, d=100, ncols=512 (dot match)
# SDM size, nact
# 1 - 42, 4
# 2 - 53, 4
# 3 - 79, 4
# 4 - 101, 4
# 5 - 124, 4
# 6 - 147, 4 **
# 7 - 172, 4 **
# 8 - 198, 4
# 9 - 226, 4

# desired_error=9, res.x=223.7670095015759, res.success=True, res.message=Solution found.
# SDM sizes for k=1000, d=100, ncols=512 (dot match)
# SDM size, nact
# 1 - 64, 5
# 2 - 65, 5
# 3 - 65, 5
# 4 - 99, 5
# 5 - 125, 5
# 6 - 148, 5
# 7 - 173, 5
# 8 - 198, 5 **
# 9 - 224, 5 **


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


def compute_bundle_size(ncols, bits_per_counter, item_memory_len, fip=1.0):
	# return size for bundle storage in bits
	# item_memory_len - number of items in item memory
	# fip - fraction of item memory that is realized physically
	counter_memory_size = ncols * bits_per_counter
	item_memory_size = item_memory_len*ncols * fip
	total_size = math.ceil(counter_memory_size + item_memory_size)
	return total_size

def compute_sdm_size(nrows, ncols, bits_per_counter, item_memory_len, fip=1.0):
	# return size of sdm storage in bits
	# item_memory_len - number of items in item memory
	# fip - fraction of item memory that is realized physically
	counter_memory_size = nrows * ncols * bits_per_counter
	address_memory_size = nrows * ncols * fip
	item_memory_size = item_memory_len*ncols * fip
	total_size = math.ceil(counter_memory_size + address_memory_size + item_memory_size)
	return total_size	
	# mem_type, either "bundle" or "sdm"

def compute_bundle_ops(ncols, bits_per_counter, item_memory_len, fimp=1.0, parallel=False):
	# return number of byte operations for recall from bundle
	# item_memory_len - number of items in item memory
	# fip - fraction of item memory that is realized physically
	# parallel - True if operations operate in parallel
	ops_for_one_match = None
	counter_memory_size = ncols * bits_per_counter
	item_memory_size = item_memory_len*ncols * fip
	total_size = math.ceil(counter_memory_size + item_memory_size)
	return total_size

def compute_sdm_ops(nrows, ncols, bits_per_counter, item_memory_len, fip=1.0):
	# return size of sdm storage in bits
	counter_memory_size = nrows * ncols * bits_per_counter
	address_memory_size = nrows * ncols * fip
	item_memory_size = item_memory_len*ncols * fip
	total_size = math.ceil(counter_memory_size + address_memory_size + item_memory_size)
	return total_size	
	# mem_type, either "bundle" or "sdm"


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
	#  mdims["bun_k1000_d100_c8"] = gal_dims(d, k)  # estimate, replaced by analytical calculation
	# print("gal_dims=%s" % mdims["bun_k1000_d100_c8"])
	gal_bun = Line_fit("bun_k1000_d100_c8#S2", bpx=8)
	bundle = Line_fit("bun_k1000_d100_c1#S1")
	bin_sdm = Line_fit("sdm_k1000_d100_c1_ham#A2", bpx=512) #("binarized_sdm")
	full_sdm = Line_fit("sdm_k1000_d100_c8_ham#A1", bpx=512*8) # ("8bit_counter_sdm")
	sdm_c8_dot = Line_fit("sdm_k1000_d100_c8_dot#A4", bpx=512*8)
	sdm_c1_dot = Line_fit("sdm_k1000_d100_c1_dot#A3", bpx=512)
	num_wanted_dims = 6
	colors = cm.rainbow(np.linspace(0, 1, num_wanted_dims))
	plt.plot(bundle.xbits, bundle.yreg, c=colors[0], label="bun_ham")
	plt.plot(bundle.xbits, bundle.y, 'o', c=colors[0])
	plt.plot(gal_bun.xbits, gal_bun.yreg, c=colors[1], label="bun_c8_dot")
	plt.plot(gal_bun.xbits, gal_bun.y, 'o',c=colors[1])
	plt.plot(bin_sdm.xbits, bin_sdm.yreg, c=colors[2], label="sdm_c1_ham")
	plt.plot(bin_sdm.xbits, bin_sdm.y, 'o', c=colors[2])
	plt.plot(full_sdm.xbits, full_sdm.yreg, c=colors[3], label="sdm_c8_ham")
	plt.plot(full_sdm.xbits, full_sdm.y, 'o',c=colors[3])
	plt.plot(sdm_c8_dot.xbits, sdm_c8_dot.yreg, c=colors[4], label="sdm_c8_dot")
	plt.plot(sdm_c8_dot.xbits, sdm_c8_dot.y, 'o',c=colors[4])
	plt.plot(sdm_c1_dot.xbits, sdm_c1_dot.yreg, c=colors[5], label="sdm_c1_dot")
	plt.plot(sdm_c1_dot.xbits, sdm_c1_dot.y, 'o',c=colors[5])
	plt.title("Bundles and sdm errors vs size (bits)")
	plt.xlabel("bits (in bundle or sdm")
	plt.ylabel("Fraction error")
	plt.yscale('log')
	# plt.xscale('log')
	yticks = (10.0**-(np.arange(9.0, 0, -1.0)))
	ylabels = [10.0**(-i) for i in range(9, 0, -1)]
	plt.yticks(yticks, ylabels)
	plt.grid()
	plt.legend(loc='upper right')
	plt.show()

# def sdm_and_bundle_rows():
# 	# display line fit to bundle length and SDM number of rows
# 	# k = 1000
# 	# d = 100
# 	#  mdims["bun_k1000_d100_c8"] = gal_dims(d, k)  # estimate, replaced by analytical calculation
# 	# print("gal_dims=%s" % mdims["bun_k1000_d100_c8"])
# 	gal_bun = Line_fit("bun_k1000_d100_c8") #, bpx=8)
# 	bundle = Line_fit("bun_k1000_d100")
# 	bin_sdm = Line_fit("sdm_k1000_d100_c1") #, bpx=512) #("binarized_sdm")
# 	full_sdm = Line_fit("sdm_k1000_d100_c8") #, bpx=512*8) # ("8bit_counter_sdm")
# 	sdm_c8_dot = Line_fit("sdm_k1000_d100_c8_dot") #, bpx=512*8)
# 	sdm_c1_dot = Line_fit("sdm_k1000_d100_c1_dot") #, bpx=512)
# 	num_wanted_dims = 6
# 	colors = cm.rainbow(np.linspace(0, 1, num_wanted_dims))
# 	plt.plot(bundle.xbits, bundle.yreg, c=colors[0], label="bun_ham")
# 	plt.plot(bundle.xbits, bundle.y, 'o', c=colors[0])
# 	plt.plot(gal_bun.xbits, gal_bun.yreg, c=colors[1], label="bun_c8_dot")
# 	plt.plot(gal_bun.xbits, gal_bun.y, 'o',c=colors[1])
# 	plt.plot(bin_sdm.xbits, bin_sdm.yreg, c=colors[2], label="sdm_c1_ham")
# 	plt.plot(bin_sdm.xbits, bin_sdm.y, 'o', c=colors[2])
# 	plt.plot(full_sdm.xbits, full_sdm.yreg, c=colors[3], label="sdm_c8_ham")
# 	plt.plot(full_sdm.xbits, full_sdm.y, 'o',c=colors[3])
# 	plt.plot(sdm_c8_dot.xbits, sdm_c8_dot.yreg, c=colors[4], label="sdm_c8_dot")
# 	plt.plot(sdm_c8_dot.xbits, sdm_c8_dot.y, 'o',c=colors[4])
# 	plt.plot(sdm_c1_dot.xbits, sdm_c1_dot.yreg, c=colors[5], label="sdm_c1_dot")
# 	plt.plot(sdm_c1_dot.xbits, sdm_c1_dot.y, 'o',c=colors[5])
# 	plt.title("Bundles and sdm errors vs size (bits)")
# 	plt.xlabel("bits (in bundle or sdm")
# 	plt.ylabel("Fraction error")
# 	plt.yscale('log')
# 	yticks = (10.0**-(np.arange(9.0, 0, -1.0)))
# 	ylabels = [10.0**(-i) for i in range(9, 0, -1)]
# 	plt.yticks(yticks, ylabels)
# 	plt.grid()
# 	plt.legend(loc='upper right')
# 	plt.show()

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
	plot_type = "sdm_vs_bundle" # "vary_item_memory"; #"vary_num_items" # "vary_num_items" # "vary_item_memory" # "compare_cdf_to_exp_fit" # "cdf_dims" #
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

