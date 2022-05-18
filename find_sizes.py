# find sizes of bundle and sdm which give specific error rates

# from scipy.optimize import bracket
from scipy.optimize import minimize_scalar
import sdm_ae as sdm_anl
import sdm_analytical_jaeckel as sdm_jaeckel
import bundle_analytical
# import bundle_empirical
import math



def bundle_error_vs_width(n, desired_error=1):
	# n is bundle size (number of components)
	# desired_error in negative power of 10, e.g. 1 means 0.1 error, 2 means 0.01 error, 3 means 1/10**3 error
	# return difference between found_error and desired_error squared (to minimize)
	d=100   # size of item memory
	k=1000  # number of items stored in bundle (number transitions)
	found_error = bundle_analytical.BundleErrorAnalytical(round(n),d,k)
	print("n=%s, found_error=%s" % (n, found_error))
	diff = (desired_error  + math.log10(found_error)) ** 2
	return diff

def sdm_error_vs_nrows(nrows, desired_error=1):
	# nrows is number of rows in the sdm
	# desired_error in negative power of 10, e.g. 1 means 0.1 error, 2 means 0.01 error, 3 means 1/10**3 error
	# return difference between found_error and desired_error squared (to minimize)
	d=100   # size of item memory
	k=1000  # number of items stored in bundle (number transitions)
	ncols = 512
	# found_error = sdm_jaeckel.SdmErrorAnalytical(round(nrows),k,d,nact=None,word_length=512)
	nact = sdm_jaeckel.get_sdm_activation_count(round(nrows), k)
	anl = sdm_anl.Sdm_error_analytical(round(nrows), nact, k, ncols=ncols, d=d)
	found_error = anl.perr
	print("nrows=%s, nact=%s, found_error=%s" % (nrows, nact, found_error))
	diff = (desired_error  + math.log10(found_error)) ** 2
	return diff

def find_bundle_size(desired_error):
	init_xa, init_xb = 5000, 200000
	# xa, xb, xc, fa, fb, fc, funcalls = bracket(bundle_error_vs_width, xa=init_xa, xb=init_xb)
	fun = bundle_error_vs_width
	bounds = (init_xa, init_xb)
	options = {"disp": True, 'xatol': 0.1}
	res = minimize_scalar(fun, bracket=None, bounds=bounds, args=(desired_error,), method='Bounded', tol=None, options=options)
	# print("xa=%s, xb=%s, fa=%s, fb=%s, fc=%ss, funcalls=%s" % (xa, xb, xc, fa, fb, fc, funcalls))
	print("desired_error=%s, res.x=%s, res.success=%s, res.message=%s" % (desired_error, res.x, res.success, res.message))
	return round(res.x)

def find_sdm_size(desired_error):
	init_xa, init_xb = 5, 350
	# xa, xb, xc, fa, fb, fc, funcalls = bracket(bundle_error_vs_width, xa=init_xa, xb=init_xb)
	fun = sdm_error_vs_nrows
	bounds = (init_xa, init_xb)
	options = {"disp": True, 'xatol': 0.1}
	res = minimize_scalar(fun, bracket=None, bounds=bounds, args=(desired_error,), method='Bounded', tol=None, options=options)
	# print("xa=%s, xb=%s, fa=%s, fb=%s, fc=%ss, funcalls=%s" % (xa, xb, xc, fa, fb, fc, funcalls))
	print("desired_error=%s, res.x=%s, res.success=%s, res.message=%s" % (desired_error, res.x, res.success, res.message))
	return round(res.x)

def find_bundle_sizes(desired_sizes):
	bundle_sizes = [find_bundle_size(x) for x in desired_sizes]
	print("Bundle sizes:")
	for i in range(len(desired_sizes)):
		print("%s - %s" % (desired_sizes[i], bundle_sizes[i]))

def find_sdm_sizes(desired_sizes):
	bundle_sizes = [find_sdm_size(x) for x in desired_sizes]
	print("SDM sizes, nact:")
	k = 1000
	for i in range(len(desired_sizes)):
		nrows = bundle_sizes[i]
		nact = sdm_jaeckel.get_sdm_activation_count(nrows, k)
		print("%s - %s, %s" % (desired_sizes[i], nrows, nact))
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

def main():
	desired_sizes = range(1,10)
	# find_bundle_sizes(desired_sizes)
	find_sdm_sizes(desired_sizes)

main()
