# find sizes of bundle and sdm which give specific error rates

# from scipy.optimize import bracket
from scipy.optimize import minimize_scalar
import sdm_ae as sdm_anl
import sdm_analytical_jaeckel as sdm_jaeckel
import bundle_analytical
import binarized_sdm_analytical
# import bundle_empirical
import math


def get_dk():
	# return size of item memory (d) and number of items stored (k)
	d = 100
	k = 1000
	return (d, k)

def get_binarized():
	# used to set binarized option (currently for bundle, possibly for sdm in future)
	# if false, uses dot product for superposition vector
	binarized = False
	return binarized

def bundle_error_vs_width(n, desired_error=1):
	# n is bundle size (number of components)
	# desired_error in negative power of 10, e.g. 1 means 0.1 error, 2 means 0.01 error, 3 means 1/10**3 error
	# return difference between found_error and desired_error squared (to minimize)
	d, k = get_dk()
	binarized = get_binarized()
	# d=25  # size of item memory
	# k=1000  # number of items stored in bundle (number transitions)
	found_error = bundle_analytical.BundleErrorAnalytical(round(n),d,k,binarized=binarized)
	print("n=%s, found_error=%s" % (n, found_error))
	diff = (desired_error  + math.log10(found_error)) ** 2
	return diff

def get_sdm_activaction_count(nrows, k):
	# made separate function so can replace by constant for testing
	nact = sdm_jaeckel.get_sdm_activation_count(round(nrows), k)
	# nact = 3
	return nact

def get_sdm_ncols():
	# return number of columns used in sdm memory
	ncols = 512
	return ncols


def sdm_error_vs_nrows(nrows, desired_error=1):
	# nrows is number of rows in the sdm
	# desired_error in negative power of 10, e.g. 1 means 0.1 error, 2 means 0.01 error, 3 means 1/10**3 error
	# return difference between found_error and desired_error squared (to minimize)
	# d=100   # size of item memory
	# k=1000  # number of items stored in bundle (number transitions)
	d, k = get_dk()
	ncols = get_sdm_ncols()
	# found_error = sdm_jaeckel.SdmErrorAnalytical(round(nrows),k,d,nact=None,word_length=512)
	# nact = sdm_jaeckel.get_sdm_activation_count(round(nrows), k)
	nact = get_sdm_activaction_count(nrows, k)
	match_method = "hamming"  # use to save time to not compute dot match
	anl = sdm_anl.Sdm_error_analytical(round(nrows), nact, k, ncols=ncols, d=d, match_method=match_method)
	found_error = anl.perr
	# found_error = anl.perr_dot
	# if nact ==1:
	# 	bsm = binarized_sdm_analytical.Binarized_sdm_analytical(nrows, ncols, nact, k, d)
	# else:
	# 	bsm = binarized_sdm_analytical.Bsa_sample(nrows, ncols, nact, k, d)
	# found_error = bsm.p_err
	if found_error == 0.0:
		found_error = 10.0**(-20)
	print("nrows=%s, nact=%s, found_error=%s" % (nrows, nact, found_error))
	diff = (desired_error  + math.log10(found_error)) ** 2
	return diff

def find_bundle_size(desired_error):
	init_xa, init_xb = 200, 200000
	# xa, xb, xc, fa, fb, fc, funcalls = bracket(bundle_error_vs_width, xa=init_xa, xb=init_xb)
	fun = bundle_error_vs_width
	bounds = (init_xa, init_xb)
	options = {"disp": True, 'xatol': 0.1}
	res = minimize_scalar(fun, bracket=None, bounds=bounds, args=(desired_error,), method='Bounded', tol=None, options=options)
	# print("xa=%s, xb=%s, fa=%s, fb=%s, fc=%ss, funcalls=%s" % (xa, xb, xc, fa, fb, fc, funcalls))
	print("desired_error=%s, res.x=%s, res.success=%s, res.message=%s" % (desired_error, res.x, res.success, res.message))
	return round(res.x)

def find_sdm_size(desired_error):
	init_xa, init_xb = 25, 400
	# xa, xb, xc, fa, fb, fc, funcalls = bracket(bundle_error_vs_width, xa=init_xa, xb=init_xb)
	fun = sdm_error_vs_nrows
	bounds = (init_xa, init_xb)
	options = {"disp": True, 'xatol': 0.1}
	res = minimize_scalar(fun, bracket=None, bounds=bounds, args=(desired_error,), method='Bounded', tol=None, options=options)
	# print("xa=%s, xb=%s, fa=%s, fb=%s, fc=%ss, funcalls=%s" % (xa, xb, xc, fa, fb, fc, funcalls))
	print("desired_error=%s, res.x=%s, res.success=%s, res.message=%s" % (desired_error, res.x, res.success, res.message))
	return round(res.x)

def find_bundle_sizes(desired_error):
	bundle_sizes = [find_bundle_size(x) for x in desired_error]
	d, k = get_dk()
	binarized = get_binarized()
	print("Bundle sizes, for k=%s, d=%s, binarized=%s:" % (k, d, binarized))
	for i in range(len(desired_error)):
		print("%s - %s" % (desired_error[i], bundle_sizes[i]))
	# output for bundle sizes:

	# Bundle sizes, for k=1000, d=100, binarized=False:
	# 1 - 15221
	# 2 - 25717
	# 3 - 35352
	# 4 - 44632
	# 5 - 53750
	# 6 - 62794
	# 7 - 71812
	# 8 - 80824
	# 9 - 89843
	#  BELOW HAVE binarized=True (use hamming distance)
	# 	Bundle sizes, k=1000, d=100:
	# 1 - 24002
	# 2 - 40503
	# 3 - 55649
	# 4 - 70239
	# 5 - 84572
	# 6 - 98790
	# 7 - 112965
	# 8 - 127134
	# 9 - 141311
	# 	Bundle sizes, for k=750, d=100:
	# 1 - 18012
	# 2 - 30389
	# 3 - 41750
	# 4 - 52693
	# 5 - 63443
	# 6 - 74108
	# 7 - 84740
	# 8 - 95367
	# 9 - 106002
	# 	Bundle sizes, for k=500, d=100:
	# 1 - 12020
	# 2 - 20273
	# 3 - 27848
	# 4 - 35144
	# 5 - 42313
	# 6 - 49424
	# 7 - 56513
	# 8 - 63599
	# 9 - 70690
	# 	Bundle sizes, for k=250, d=100:
	# 1 - 6025
	# 2 - 10153
	# 3 - 13942
	# 4 - 17593
	# 5 - 21179
	# 6 - 24736
	# 7 - 28283
	# 8 - 31827
	# 9 - 35374
	# 	Bundle sizes, for k=125, d=100:
	# 1 - 3000
	# 2 - 5050
	# 3 - 6931
	# 4 - 8743
	# 5 - 10523
	# 6 - 12290
	# 7 - 14051
	# 8 - 15811
	# 9 - 17572
	# 	Bundle sizes, for k=1000, d=20:
	# 1 - 17175
	# 2 - 32221
	# 3 - 46592
	# 4 - 60729
	# 5 - 74793
	# 6 - 88851
	# 7 - 102929
	# 8 - 117037
	# 9 - 131176
	# Bundle sizes, for k=1000, d=25:
	# 1 - 18147
	# 2 - 33407
	# 3 - 47886
	# 4 - 62084
	# 5 - 76182
	# 6 - 90260
	# 7 - 104351
	# 8 - 118466
	# 9 - 132610

def find_sdm_sizes(desired_error):
	sdm_sizes = [find_sdm_size(x) for x in desired_error]
	d, k = get_dk()
	ncols = get_sdm_ncols()
	print("SDM sizes for k=%s, d=%s, ncols=%s (hamming match - jaeckel nact)" %(k, d, ncols))
	print("SDM size, nact")
	k = 1000
	ncols = get_sdm_ncols()
	for i in range(len(desired_error)):
		nrows = sdm_sizes[i]
		nact = get_sdm_activaction_count(nrows, k)
		print("%s - %s, %s" % (desired_error[i], nrows, nact))
	# output from sdm_jaeckel:
		# SDM sizes, nact
		# 1 - 51, 1
		# 2 - 88, 2
		# 3 - 122, 2
		# 4 - 155, 2
		# 5 - 188, 3
		# 6 - 221, 3
		# 7 - 255, 3
		# 8 - 288, 3
		# 9 - 323, 4
	# output from sdm_anl.Sdm_error_analytical:
		# SDM sizes, nact:
		# 1 - 51, 1
		# 2 - 86, 2
		# 3 - 125, 2
		# 4 - 168, 2
		# 5 - 196, 3
		# 6 - 239, 3
		# 7 - 285, 3
		# 8 - 312, 4
		# 9 - 349, 4
	# SDM sizes, nact=1 (binarized_sdm with nact=1):
		# 1 - 50 *
		# 2 - 97 *
		# 3 - 161
		# 4 - 254
		# 5 - 396
		# 6 - 619
		# 7 - 984
		# 8 - 1599
		# 9 - 2665
	# SDM sizes, nact = 3 (binarized_sdm with nact=3)
		# 1 - 66, 3
		# 2 - 112, 3
		# 3 - 158, 3 *
		# 4 - 208, 3 *
		# 5 - 262, 3 *
		# 6 - 323, 3
		# 7 - 391, 3
		# 8 - 471, 3
		# 9 - 559, 3
	# SDM sizes, nact = 5 (binarized_sdm with nact=5)
		# 1 - 81, 5
		# 2 - 127, 5
		# 3 - 172, 5
		# 4 - 219, 5
		# 5 - 265, 5
		# 6 - 315, 5 *
		# 7 - 368, 5 *
		# 8 - 425, 5
		# 9 - 486, 5
	# SDM sizes, nact = 7 (binarized_sdm with nact=7)
		# 1 - 95, 7
		# 2 - 141, 7
		# 3 - 186, 7
		# 4 - 231, 7
		# 5 - 277, 7
		# 6 - 324, 7
		# 7 - 372, 7
		# 8 - 424, 7 *
		# 9 - 476, 7 *



# Using sdm_ae, with nact==1 all the time:
# SDM sizes, nact
# 1 - 51, 1
# 2 - 99, 1
# 3 - 163, 1
# 4 - 255, 1
# 5 - 388, 1
# 6 - 583, 1
# 7 - 867, 1
# 8 - 1212, 1
# 9 - 1573, 1
# (base) Jeffs-MacBook:hdfsa jt$ 

# Increase pruning threshold to 10000000
# desired_error=9, res.x=2583.62255068014, res.success=True, res.message=Solution found.
# SDM sizes, nact
# 1 - 51, 1
# 2 - 99, 1
# 3 - 164, 1
# 4 - 260, 1
# 5 - 405, 1
# 6 - 633, 1
# 7 - 999, 1
# 8 - 1581, 1
# 9 - 2584, 1

# If no pruning:
# SDM sizes, nact
# 1 - 51, 1
# 2 - 99, 1
# 3 - 164, 1
# 4 - 259, 1
# 5 - 406, 1
# 6 - 636, 1
# 7 - 1014, 1
# 8 - 1651, 1
# 9 - 2763, 1

# Now try sdm_ae with nact=2; reduces rows greatly
# SDM sizes, nact
# 1 - 50, 2 *
# 2 - 86, 2 *
# 3 - 125, 2
# 4 - 168, 2
# 5 - 219, 2
# 6 - 278, 2
# 7 - 348, 2
# 8 - 433, 2
# 9 - 536, 2

# # nact = 3, sdm_ae:
# SDM sizes, nact
# 1 - 52, 3
# 2 - 87, 3
# 3 - 121, 3 *
# 4 - 157, 3 *
# 5 - 196, 3
# 6 - 238, 3
# 7 - 285, 3
# 8 - 338, 3
# 9 - 396, 3

# SDM sizes, nact = 4
# 1 - 55, 4
# 2 - 90, 4
# 3 - 123, 4
# 4 - 157, 4 
# 5 - 191, 4 *
# 6 - 229, 4
# 7 - 268, 4
# 8 - 311, 4
# 9 - 357, 4

# SDM sizes, nact = 5
# 1 - 59, 5
# 2 - 94, 5
# 3 - 127, 5
# 4 - 160, 5
# 5 - 193, 5
# 6 - 228, 5 *
# 7 - 265, 5 *
# 8 - 303, 5 *
# 9 - 344, 5

# SDM sizes, nact=6
# 1 - 61, 6
# 2 - 97, 6
# 3 - 131, 6
# 4 - 164, 6
# 5 - 197, 6
# 6 - 231, 6
# 7 - 266, 6
# 8 - 305, 6
# 9 - 340, 6 *

# SDM sizes, nact=7
# 1 - 63, 7
# 2 - 102, 7
# 3 - 136, 7
# 4 - 169, 7
# 5 - 201, 7
# 6 - 235, 7
# 7 - 269, 7
# 8 - 307, 7
# 9 - 341, 7

	# 	If use ncols = 256 (reduce by half), increases size more (more than doubles):
	# 	SDM sizes, nact=1 (binarized_sdm with nact=1):
	# 1 - 106
	# 2 - 236
	# 3 - 468
	# 4 - 927
	# 5 - 1947

	# If use ncols = 1024, reduces by more than half number of rows needed:
	# 	SDM sizes, nact=1 (binarized_sdm with nact=1):
	# 1 - 24
	# 2 - 44
	# 3 - 66
	# 4 - 94
	# 5 - 130
	# 6 - 177
	# 7 - 240
	# 8 - 323
	# 9 - 438

def main():
	desired_error = range(1,10) # range(1,10)
	# find_bundle_sizes(desired_error)
	find_sdm_sizes(desired_error)

main()
