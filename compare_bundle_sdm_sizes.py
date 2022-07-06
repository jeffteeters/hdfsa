

# compare ratio (number of bits required) of bundle to binarized 1-nact sdm
# this is not too useful because size of sdm is greatly reduced when nact (activaction count)
# is greater than 1.

bundle_sizes = [
	(1, 24002),
	(2, 40503),
	(3, 55649),
	(4, 70239),
	(5, 84572),
	(6, 98790),
	(7, 112965),
	(8, 127134),
	(9, 141311)]

binarized_sdm_sizes = [  # assumes nact = 1, ncols=512
	(1, 50),
	(2, 97),
	(3, 161),
	(4, 254),
	(5, 396),
	(6, 619),
	(7, 984),
	(8, 1599),
	(9, 2665)]


ncols = 512
print("bundle size (bits) vs binarized_sdm_size (1 bit counters)")
print("bun_width - width of bundle to obtain desired error rate with item memory size (d) == 100")
print("sdm_bits - bits in binarized_sdm, with ncols=512.  Is nrows in sdm * 512")
print("ratio - ratio: sdm_bits / bun_width")
print("ier\tbun_width\tsdm_bits\tratio")
for i in range(len(bundle_sizes)):
	ier, bun_width = bundle_sizes[i]
	ier2, nrows = binarized_sdm_sizes[i]
	assert ier == ier2
	sdm_bits = ncols * nrows
	ratio = sdm_bits / bun_width
	print("%s\t%s\t%s\t%s" % (ier, bun_width, sdm_bits, ratio))
