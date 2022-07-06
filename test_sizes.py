# empiriclly test sizes found for desired error rates
# test done using sdm_bc, which has been replaced by fast_sdm_empirical
# and fast_bundle_empirical


import sdm_bc

sizes = (
		("sdm_numerical_estimate", (
		# (1, 51, 1),
		# (2, 86, 2),
		# (3, 125, 2),
		# (4, 168, 2),
		# (5, 196, 3),
		(6, 239, 3),
		# (7, 285, 3),
		# (8, 312, 4),
		# (9, 349, 4)
		)),
	("sdm_jackel", (
		# format is: desired error rate (10e-n), nrows, nact:
		# (1, 51, 1),
		# (2, 88, 2),
		# (3, 122, 2),
		# (4, 155, 2),
		# (5, 188, 3),
		(6, 221, 3),
		# (7, 255, 3),
		# (8, 288, 3),
		# (9, 323, 4)
		)),
	# ("bundle_analytical", (
	# 	(1, 24002),
	# 	(2, 40503),
	# 	(3, 55649),
	# 	(4, 70239),
	# 	(5, 84572),
	# 	(6, 98790),
	# 	(7, 112965),
	# 	(8, 127134),
	# 	(9, 141311))),
	)

def test_sdm(vals):
	word_length = 512
	states = 100
	choices = 10
	actions = 10
	bc = 8
	for inf in vals[0:5]:
		expected_error, nrows, nact = inf
		mem = sdm_bc.Sparse_distributed_memory(word_length, nrows, nact, bc)
		ri = sdm_bc.empirical_response(mem, actions, states, choices, ntrials=10000000)
		error_rate = ri["error_rate"]
		print("%s, nrows=%s, nact=%s, found_error=%s" % (expected_error, nrows, nact, error_rate))

def test_bundle(vals):
	states = 100
	choices = 10
	actions = 10
	for inf in vals[0:5]:
		expected_error, word_length = inf
		mem = sdm_bc.Bundle_memory(word_length)
		ri = sdm_bc.empirical_response(mem, actions, states, choices, ntrials=6000)
		error_rate = ri["error_rate"]
		print("%s, word_length=%s, found_error=%s" % (expected_error, word_length, error_rate))

def main():
	global sizes
	for name_vals in sizes:
		name, vals = name_vals
		print("testing %s" % name)
		if name.startswith("sdm_"):
			test_sdm(vals)
		elif name.startswith("bundle_"):
			test_bundle(vals)
		else:
			sys.exit("Unknown key prefix: %s" % name)


main()

