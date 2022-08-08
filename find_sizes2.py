# script to find sizes of bundle and sdm for particular error rate

import sdm_ae as sdm_anl
import sdm_analytical_jaeckel as sdm_jaeckel
import bundle_analytical
import binarized_sdm_analytical

mem_info = [
	{
		"name": "bun_k1000_d100_c1#S1",
		"short_name": "S1",
 		"mtype": "bundle",
 		"binarize_counters": True,  # use hamming match
  	},
  	{
		"name": "bun_k1000_d100_c8#S2",
		"short_name": "S2",
 		"mtype": "bundle",
 		"binarize_counters": False,  # used dot product match
  	},
  	{
  		"name": "sdm_k1000_d100_c8_ham#A1",
		"short_name": "A1",
 		"mtype": "sdm",
 		"bits_per_counter": 8,
 		"match_method": "hamming",
 	},
 	{
  		"name": "sdm_k1000_d100_c1_ham#A2",
		"short_name": "A2",
 		"mtype": "sdm",
 		"bits_per_counter": 1,
 		"match_method": "hamming",
 	},
 	{
  		"name": "sdm_k1000_d100_c1_dot#A3",
		"short_name": "A3",
 		"mtype": "sdm",
 		"bits_per_counter": 1,
 		"match_method": "dot",
 	},
 	{
  		"name": "sdm_k1000_d100_c8_dot#A4",
		"short_name": "A4",
 		"mtype": "sdm",
 		"bits_per_counter": 8,
 		"match_method": "dot",
 	},
]

def bundle_error(ncols, k, d, binarize_counters):
	# return predicted error for bundle
	# ncols - width of bundle
	# k - number of items stored
	# d - number items in item memory (are d-1 distractors)
	# binarize_counters - True if are binarizing counters (hamming match), False if dot product match
	error = bundle_analytical.BundleErrorAnalytical(ncols,d,k,binarized=binarize_counters)
	return error

def sdm_error(nrows, k, d, nact, ncols, match_method="hamming", bits_per_counter=8):
	# return perdicted error for sdm
	# k - number of items stored
	# d - number items in item memory (are d-1 distractors)
	# match_method - "hamming" if threshold sums, "dot" if use dot product
	# bits_per_counter - 8 if full size, 1 if binarize counters
	if bits_per_counter == 8:
		assert match_method in ("hamming", "dot")
		ae = sdm_anl.Sdm_error_analytical(nrows, nact, k, ncols=ncols, d=d, match_method=match_method)
		error = ae.perr if match_method == "hamming" else ae.perr_dot
	else:
		assert bits_per_counter == 1
		if nact ==1:
			bsm = binarized_sdm_analytical.Binarized_sdm_analytical(nrows, ncols, nact, k, d)
		else:
			bsm = binarized_sdm_analytical.Bsa_sample(nrows, ncols, nact, k, d)
		error = bsm.p_err
	return error

def get_sdm_ncols():
	ncols = 512
	return ncols

def get_sdm_nact(nrows, k):
	# nact = sdm_jaeckel.get_sdm_activation_count(nrows, k)
	# nact = 3
	# return closest odd number for model "sdm_k1000_d100_c1_ham#A2"
	fnact = sdm_jaeckel.fraction_rows_activated(nrows, k)*nrows
	# round to closest odd integer
	nact = 2 * int(fnact/2) + 1
	# print("nrows=%s, k=%s, fnact=%s, nact=%s" % (nrows, k, fnact, nact))
	return nact

def get_number_items_stored():
	k = 1000  # number items stored
	return k

def get_item_memory_size():
	d = 100
	return d

def get_error(mi, size):
	# return error for memory with properties given in dictionary mi (mem_info dictionary above)
	# size is number of rows (for sdm) or width of bundle
	# nact is activaction count for sdm, or None if use optimum
	# ncols is the width of sdm
	k = get_number_items_stored()  # number items stored
	d = get_item_memory_size()  # number of items in item memory (d - 1) distractors
	if mi["mtype"] == "bundle":
		ncols = size
		binarize_counters = mi["binarize_counters"]
		error = bundle_error(ncols, k, d, binarize_counters)
	else:
		nrows = size
		assert mi["mtype"] == "sdm"
		nact = get_sdm_nact(nrows, k)
		ncols = get_sdm_ncols()
		match_method = mi["match_method"]
		bits_per_counter = mi["bits_per_counter"]
		error = sdm_error(nrows, k, d, nact, ncols, match_method=match_method, bits_per_counter=bits_per_counter)
	return error

def get_cache_error(size, mi, sizes_tested):
	if size in sizes_tested:
		error = sizes_tested[size]
	else:
		error = get_error(mi, size)
		sizes_tested[size] = error
	return error

def find_sizes(mi, ie_start=1, ie_end=10):
	# find sizes for memory with properties in dictionary mi
	mtype = mi["mtype"]
	assert mtype in ("sdm", "bundle")
	# following stores size => error; size is ncols (bundle) or nrows (sdm)
	sizes_tested = {}
	sizes = []
	for ie in range(ie_start, ie_end):  # ie is error in powers of 10
		desired_error = 10**(-ie)
		low, hi = (1000, 200000) if mtype == "bundle" else (20, 600)
		# find new low and hi if there were tests already
		for size, error in sizes_tested.items():
			if size > low and error > desired_error:
				low = size
			if size < hi and error < desired_error:
				hi = size
		print("ie=%s, start search with low=%s, hi=%s" % (ie, low, hi))
		# now do binary search for best error matching desired_error
		while low < hi-1:
			mid = int((low + hi)/2)
			mid_error = get_cache_error(mid, mi, sizes_tested)
			print("ie=%s, low=%s, hi=%s, mid=%s, error=%s" % (ie, low, hi, mid, mid_error))
			if mid_error < desired_error:
				# mid_error too small, decrease size to increase error.  Decrease size by reducing hi
				hi = mid
			elif mid_error > desired_error:
				# need to increase size to reduce error.  Increase size by increasing low
				low = mid
			else:
				# found exact match
				low = mid
				hi = mid
		if low == hi:
			size = low
		else:
			low_error = get_cache_error(low, mi, sizes_tested)
			hi_error = get_cache_error(hi, mi, sizes_tested)
			low_diff = abs(low_error - desired_error)
			hi_diff = abs(hi_error - desired_error)
			if low_diff < hi_diff:
				error = low_error
				size = low
				diff = low_diff
				other = hi
				other_diff = hi_diff
				other_error = hi_error
			else:
				error = hi_error
				size = hi
				diff = hi_diff
				other = low
				other_diff = low_diff
				other_error = low_error
		print("Selected %s, size=%s, error=%s, diff=%s; other=%s, error=%s diff=%s" % (ie, size, error, diff,
			 other, other_error, other_diff ))
		sizes.append((ie, size, error))
	return sizes

def lookup_mem_name(mem_name):
	global mem_info
	for mi in mem_info:
		if mi["name"] == mem_name:
			return mi
	sys.exit("Error: Unable to find memory '%s' in mem_info" % mem_name)

def main():
	# mem_name = "sdm_k1000_d100_c8_ham#A1"
	# mem_name = "sdm_k1000_d100_c8_dot#A4"
	mem_name = "sdm_k1000_d100_c1_ham#A2" # "bun_k1000_d100_c8#S2" # "bun_k1000_d100_c1#S1"
	mi = lookup_mem_name(mem_name)
	assert mi["mtype"] in ("bundle", "sdm")
	sizes = find_sizes(mi)
	k = get_number_items_stored()
	print("For k=%s, %s dimensions are:" % (k, mem_name))
	for size in sizes:
		ie, nrows, error = size
		if mi["mtype"] == "sdm":
			nact = get_sdm_nact(nrows, k)
			print("[%s, %s, %s, %s]," % (ie, nrows, nact, error))
		else:
			print("[%s, %s, %s]," % (ie, nrows, error))


if __name__ == "__main__":
	main()