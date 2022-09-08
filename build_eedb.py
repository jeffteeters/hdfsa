# build (or update) empirical error database (database of empiricl errors)

import fast_sdm_empirical
import fast_bundle_empirical
import numpy as np
import matplotlib.pyplot as plt
import os.path
import sqlite3
import math
# from scipy.stats import linregress
from mdie import mdie
import pprint
import time
pp = pprint.PrettyPrinter(indent=4)
from multiprocessing import Pool

roll_address = False

class Empirical_error_db():
	# Class to create and update empirical error database

	db_schema = """
	create table memory(
		id integer primary key,
		name text not null,
		short_name text not null,
		mtype text CHECK( mtype in ('sdm', 'bundle')) not null,
		bits_per_counter integer CHECK (bits_per_counter > 0 and bits_per_counter <= 8) not null,
		match_method text CHECK (match_method in ('dot', 'hamming')) not null
	);
	create table dim(
		-- dimensions of memory
		id integer primary key,
		mem_id integer not null,   -- foreign key to memory table
		ie integer CHECK (ie > 0 and ie < 20) not null,  -- desired power of 10 error (10**(-ie))
		size integer not null,  -- width of bundle or nrows of sdm
		ncols integer not null default 0,  -- ncols if sdm; not used for bundle
		nact integer not null default 0,  -- activaction count for sdm; not used for bundle
		pe real not null   -- predicted error from numerical estimation (should be close to 10**(-ie))
	);
	create table error(
		-- count of epoch errors
		-- error table used to calculate the mean and standard deviation of errors
		id integer primary key,
		dim_id integer not null,  -- foreign key to dim table
		nerrors integer not null, -- number of errors found in one epoch
		nepochs integer not null  -- number of epochs that have this (nerrors) of errors
	);
	create table stats(
		id integer primary key,
		dim_id integer not null,  -- foreign key to dim table
		epochs integer not null,  -- number of epochs used in calculating mean and std
		mean real not null,  -- mean error rate
		std real not null  -- standard deviation of error rate
	);
	create table pmf_stats(
		-- count of match and distractor distances that are used to calculate match and distractor distributions (pmf)
		-- which are used to calculate the pmf_error (error rate estimated from distributions)
		id integer primary key,
		dim_id integer not null,  -- foreign key to dim table
		match_counts text not null,    -- count of match distances, used to calculate match distribution, which
		    -- is used with distractor distribution to calculate count_error; format is: <count0_distance>;count0,count1,count2,...
		distract_counts text not null,  -- count of distractor distances, used to calculate distractor distribution
		    -- which is then used with match_distribution to calculate count_error; save format as above
		pmf_error real not null -- error calculated from match and distractor distributions made from match_counts and distract_counts
	);
	"""

	# def __init__(self, dbfile="empirical_error_noroll_v7.db"): #
	def __init__(self, dbfile="empirical_error_noroll.db"): #
	# def __init__(self, dbfile="empirical_error_rw1_roll.db"):
		self.dbfile = dbfile
		# import pdb; pdb.set_trace()
		if not os.path.isfile(dbfile):
			con = self.initialize_database()
			print("created new databases")
		else:
			con = sqlite3.connect(dbfile)
			print("opened existing database")
		self.con = con

	def initialize_database(self):
		con = sqlite3.connect(self.dbfile)
		cur = con.cursor()
		# Create database tables
		cur.executescript(Empirical_error_db.db_schema )
		con.commit()
		# insert memory dimension information from file mdie.py
		for mi in mdie:
			sql = "insert into memory (name, short_name, mtype, bits_per_counter, match_method) values " \
				"(:name, :short_name, :mtype, :bits_per_counter, :match_method)"
			cur.execute(sql, mi)  # fills values in from mi dictionary
			mem_id = cur.lastrowid
			mtype = mi["mtype"]
			assert mtype in ('sdm', 'bundle')
			dims = mi["dims"]
			for dim in dims:
				if mtype == "bundle":
					ie, width, pe = dim
					sql = "insert into dim (mem_id, ie, size, pe) values (%s, %s, %s, %s)" % (mem_id, ie, width, pe)
				else:
					ie, nrows, nact, pe = dim
					ncols = 512
					sql = "insert into dim (mem_id, ie, size, ncols, nact, pe) values (%s, %s, %s, %s, %s, %s)" % (
						mem_id, ie, nrows, ncols, nact, pe)
				cur.execute(sql)
		con.commit()
		cur.close()
		return con

	def get_minfo(self, name):
		# return dictionary of memory properties
		sql = "select id, short_name, mtype, bits_per_counter, match_method from memory where name = '%s'" % name
		cur = self.con.cursor()
		res = cur.execute(sql)
		row = res.fetchone()
		if row is None:
			sys.exit("Could not find memory with name '%s' in database" % name)
		mem_id, short_name, mtype, bits_per_counter, match_method = row
		# get dims
		if mtype == "sdm":
			sql = ("select dim.id, ie, size, ncols, nact, pe, epochs, mean, std,\n"
				" match_counts, distract_counts, pmf_error from dim\n"
				" left join stats on dim.id = stats.dim_id\n"
				" left join pmf_stats on dim.id = pmf_stats.dim_id\n"
				" where mem_id = %s order by ie" % mem_id)
		else:
			sql = ("select dim.id, ie, size, pe, epochs, mean, std,\n"
				" match_counts, distract_counts, pmf_error from dim\n"
				" left join stats on dim.id = stats.dim_id\n"
				" left join pmf_stats on dim.id = pmf_stats.dim_id\n"
				" where mem_id = %s order by ie" % mem_id)
		cur = self.con.cursor()
		res = cur.execute(sql)
		dims = res.fetchall()
		info = {"mem_id": mem_id, "short_name": short_name, "mtype": mtype, 
			"bits_per_counter":bits_per_counter, "match_method": match_method, "dims":dims}
		return info

	def get_memory_names(self, mtype=None):
		# return list of memory names
		# if mtype specified, return only those with match mtype
		where_clause = ("where mtype = '%s' " % mtype) if mtype is not None else ""
		sql = "select name from memory %sorder by id" % where_clause
		cur = self.con.cursor()
		res = cur.execute(sql)
		mnames = res.fetchall()
		mnames = [x[0] for x in mnames]  # return just name, not name in tuple
		return mnames

	def add_stats(self, dim_id, epochs, mean, std):
		# add statistics to stats table
		sql = "select id from stats where dim_id = %s" % dim_id
		cur = self.con.cursor()
		res = cur.execute(sql)
		row = res.fetchone()
		if row is None:
			sql = "insert into stats (dim_id, epochs, mean, std) values (%s, %s, %s, %s)" % (
				dim_id, epochs, mean, std)
		else:
			stat_id = row[0]
			sql = "update stats set epochs=%s, mean=%s, std=%s where id = %s" % (
				epochs, mean, std, stat_id)
		cur = self.con.cursor()
		res = cur.execute(sql)
		self.con.commit()

	def add_error(self, dim_id, nerrors, nepochs):
		# add errors to error table
		# if dim_id already has nerrors in table, add nepochs to value stored 
		sql = "select id from error where dim_id = %s and nerrors = %s" % (dim_id, nerrors)
		cur = self.con.cursor()
		res = cur.execute(sql)
		row = res.fetchone()
		if row is None:
			sql = "insert into error (dim_id, nerrors, nepochs) values (%s, %s, %s)" % (
				dim_id, nerrors, nepochs)
		else:
			error_id = row[0]
			sql = "update error set nepochs = nepochs + %s where id = %s" % (nepochs, error_id)
		cur = self.con.cursor()
		res = cur.execute(sql)
		self.con.commit()

	def add_multi_error(self, dim_id, fail_counts):
		# fail_counts is array containing numper of failures in each epoch
		binned_fail_counts = np.bincount(fail_counts)  # count of number of times each error count occurred
		for nerrors in range(len(binned_fail_counts)):
			nepochs = binned_fail_counts[nerrors]
			self.add_error(dim_id, nerrors, nepochs)

	def calculate_stats(self, dim_id, items_per_epoch):
		# use entries in error table to calculate stats
		# items_per_epoch - number of items stored per epoch; used to calculate error rate per one item recall
		sql = "select nerrors, nepochs from error where dim_id = %s" % dim_id
		cur = self.con.cursor()
		res = cur.execute(sql)
		nee = res.fetchall()
		if len(nee) == 0:
			# no entries for this dim, don't calculate anything
			return
		values = np.empty(len(nee), dtype=np.float64)
		weights = np.empty(len(nee), dtype=np.uint32)
		for i in range(len(nee)):
			values[i], weights[i] = nee[i]
		values = values / items_per_epoch  # make values be error rate for recalling one item; instead of num errors
		# calculate mean and std using method at:
		# https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
		mean, epochs = np.average(values, weights=weights, returned=True)
		# Fast and numerically precise:
		variance = np.average((values-mean)**2, weights=weights)
		std = math.sqrt(variance)
		self.add_stats(dim_id, epochs, mean, std)
		# print("dim_id=%s, values=%s, weights=%s, mean=%s, variance=%s" % (dim_id, values, weights, mean, variance))
		return (mean, std, epochs)

	def update_stats(self):
		# routine to recalculate all stats; used to fix error in stats due to bug, not including number of transitions
		num_transitions = 1000  # 100 states, 10 actions per state
		sql = "select distinct dim_id from error"
		cur = self.con.cursor()
		res = cur.execute(sql)
		dim_ids = res.fetchall()
		for i in range(len(dim_ids)):
			dim_id = dim_ids[i][0]
			self.calculate_stats(dim_id, num_transitions)

	def get_pmf_counts(self, dim_id, kind, additional_counts):
		# add counts to previously stored match_counts or distract_counts
		# these are used to calculate match and distractor distributions which are then used to calculate count_error
		# kind is the name of the field, either "match_counts" or "distract_counts"
		# additional_counts is a dictionary, with keys distance (hamming or negagive of dot product) and
		# values counts of occurances of that distance.
		assert kind in ("match_counts", "distract_counts")
		sql = "select %s from pmf_stats where dim_id = %s" % (kind, dim_id)
		cur = self.con.cursor()
		res = cur.execute(sql)
		row = res.fetchone()
		counts = {}
		if row is not None and row[0] != "":
			# convert orig_counts_str to ints, then put in counts dictionary
			orig_counts_str = row[0]
			start_index, counts_str = orig_counts_str.split(";", 1)
			start_index = int(start_index)
			# orig_counts = counts_str.split(",")
			orig_counts = [int(x) for x in counts_str.split(",")]  # convert to integers
			#  map(int, orig_counts)  # not sure why this does not work
			index = start_index
			for count in orig_counts:
				if count > 0:
					# save count in dict
					counts[index] = count
				index += 1
			print("recovered %s %s counts, dim_id=%s, start_index=%s" % (len(counts), kind, dim_id, start_index))
		# now add in additional_counts
		for key, val in additional_counts.items():
			if key in counts:
				counts[key] += val
			else:
				counts[key] = val
		# convert from dict to array, with zeros in unused distances
		start_index = min(counts.keys())
		end_index = max(counts.keys())
		length = end_index - start_index + 1
		counts_arr = np.zeros(length, dtype=np.int64)
		for key, val in counts.items():
			counts_arr[key - start_index] = val
		# convert from array to string for storing into database
		counts_str = ",".join(["%s" % x for x in counts_arr])
		counts_str = "%s;%s" % (start_index, counts_str)  # put starting index in front
		# normalize array to make pmf
		counts_pmf = counts_arr / counts_arr.sum()
		# return start_index, array and string
		return (start_index, counts_pmf, counts_str)

	def update_pmf_stats(self, dim_id, match_counts, distract_counts, name):
		# match_counts and distract_counts are both dictionaries
		# with key=distance (hamming or negative of dot product) and values counts of occurances of that distance
		match_start_idx, match_pmf, match_counts_str = self.get_pmf_counts(dim_id, "match_counts", match_counts)
		distract_start_idx, distract_pmf, distract_counts_str = self.get_pmf_counts(dim_id, "distract_counts", distract_counts)
		if distract_start_idx <= match_start_idx:
			# distract_cdf are all distract values less then or equal to current match value
			end_idx = min(match_start_idx - distract_start_idx + 1, distract_pmf.size)
			distract_cdf = distract_pmf[0:end_idx].sum()
		else:
			distract_cdf = 0
		distract_match_offset = distract_start_idx - match_start_idx
		distract_0_based_idx = -distract_match_offset
		if False:
			match_xvals = np.arange(len(match_pmf)) + match_start_idx
			distract_xvals = np.arange(len(distract_pmf)) + distract_start_idx
			plt.plot(match_xvals, match_pmf, label="match_pmf")
			plt.plot(distract_xvals, distract_pmf, label="distract_pmf")
			plt.title("%s match and distract pmf" % name)
			plt.xlabel("hamming or dot product distance")
			plt.ylabel("Frequency")
			plt.grid()
			plt.legend(loc='upper right')
			plt.show(block=False)
		# import pdb; pdb.set_trace()
		# distract_idx = distract_start_idx - match_start_idx
		p_err = 0.0
		for i in range(0, len(match_pmf)):
			p_err += match_pmf[i] * distract_cdf
			distract_0_based_idx += 1
			match_idx = i + match_start_idx
			# distract_idx = match_idx # + distract_match_offset
			# distract_0_based_idx = distract_idx - distract_start_idx
			if distract_0_based_idx >= 0 and distract_0_based_idx < distract_pmf.size:
				distract_cdf += distract_pmf[distract_0_based_idx]
			# if match_idx == 11700:
			# 	print("at match_index = 11700, distract_cdf = %s, p_err=%s" % (distract_cdf, p_err))
			# 	import pdb; pdb.set_trace()
		assert p_err >= 0 and p_err <= 1
		# store in database
		sql = "select id from pmf_stats where dim_id = %s" % dim_id
		cur = self.con.cursor()
		res = cur.execute(sql)
		row = res.fetchone()
		if row is not None:
			# update existing entry
			pmf_stats_id = row[0]
			sql = ("update pmf_stats set match_counts='%s',\n"
				"	distract_counts='%s',\n"
				"	pmf_error=%s where id = %s") % (match_counts_str, distract_counts_str, p_err, pmf_stats_id)
		else:
			# add new entry
			sql = ("insert into pmf_stats (dim_id, match_counts, distract_counts, pmf_error) values\n"
				"	(%s, '%s', '%s', %s)" % (dim_id, match_counts_str, distract_counts_str, p_err))
		cur = self.con.cursor()
		res = cur.execute(sql)
		self.con.commit()
		return p_err


def get_bundle_ee(ncols, bits_per_counter, match_method, needed_epochs):
	# return bundle_empirical_error object
	global roll_address
	assert bits_per_counter == 8
	assert match_method in ("dot", "hamming")
	binarize_counters = match_method == "hamming"
	fbe = fast_bundle_empirical.Fast_bundle_empirical(ncols, epochs=needed_epochs, binarize_counters=binarize_counters,
		roll_address=roll_address)
	return fbe

def get_sdm_ee(nrows, ncols, nact, bits_per_counter, match_method, needed_epochs):
	# return sdm_empirical_error object
	global roll_address
	assert bits_per_counter in (1, 8)
	assert match_method in ("dot", "hamming")
	assert ncols == 512  # for current simulations
	threshold_sum = match_method == "hamming"
	fse = fast_sdm_empirical.Fast_sdm_empirical(nrows, ncols, nact, epochs=needed_epochs, roll_address=roll_address,
			bits_per_counter=bits_per_counter, threshold_sum=threshold_sum)
	return fse

def fill_eedb(mem_name=None):
	# main routine to populate the Empirical Error database
	edb = Empirical_error_db()
	if mem_name is None:
		# no name specified, process all of them
		mem_names = edb.get_memory_names("sdm") + edb.get_memory_names("bundle")  # process sdm first
	else:
		mem_names = [ mem_name, ]  # process just this one
	for name in mem_names:
		mi = edb.get_minfo(name)
		mtype = mi["mtype"]  # "bundle" or "sdm"
		bits_per_counter = mi["bits_per_counter"]
		match_method = mi["match_method"]
		for dim in mi["dims"]:
			if mtype == "sdm":
				# continue  # skip sdm for now
				dim_id, ie, nrows, ncols, nact, pe, epochs, mean, std, match_counts, distract_counts, pmf_error = dim
				wanted_epochs = get_epochs(ie, bundle=False)
			else:
				dim_id, ie, ncols, pe, epochs, mean, std, match_counts, distract_counts, pmf_error = dim
				# if ie == 6:
				# 	wanted_epochs = 300
				# elif ie == 7:
				# 	wanted_epochs = 300
				# 	# if ie == 6 else None  # try 100 epochs with bundle, for pe == 6
				# else:
				# 	wanted_epochs = None
				wanted_epochs = get_epochs(ie, bundle=True)
			if epochs is None:
				epochs = 0  # have no epochs
			if wanted_epochs is not None and wanted_epochs > epochs:
				# need more epochs
				needed_epochs = wanted_epochs if ie < 4 else wanted_epochs - epochs # if ie < 4, store stats directly
				print("%s, starting ie=%s, %s, needed_epochs=%s" % (time.ctime(), ie, name, needed_epochs))
				if mtype == "sdm":
					fee = get_sdm_ee(nrows, ncols, nact, bits_per_counter, match_method, needed_epochs)
				else:
					fee = get_bundle_ee(ncols, bits_per_counter, match_method, needed_epochs) # find empirical error
				# update pmf_stats
				pmf_error = edb.update_pmf_stats(dim_id, fee.match_counts, fee.distract_counts, name)
				# store empirical count stats
				if ie < 4:
					# store stats directly
					edb.add_stats(dim_id, wanted_epochs, fee.mean_error, fee.std_error)
					print("%s ie=%s, fresh epochs=%s, mean=%s, pmf_error=%s, std=%s" % (name,
						ie, wanted_epochs, fee.mean_error, pmf_error, fee.std_error))
				else:
					# add errors to error table and recalculate stats
					edb.add_multi_error(dim_id, fee.fail_counts)
					items_per_epoch = fee.num_transitions
					mean, std, new_epochs = edb.calculate_stats(dim_id, items_per_epoch)
					print("%s ie=%s, added epochs=%s, mean=%s, pmf_error=%s, std=%s, new_epochs=%s" % (name, ie,
						needed_epochs, mean, pmf_error, std, new_epochs))
		print("%s, Finished %s" % (time.ctime(), name))

def get_epochs(ie, bundle=False):
	# ie is expected error rate, range 1 to 9 (10^(-ie))
	# return number of epochs required or None if not running this one because would require too many epochs
	num_transitions = 1000  # 100 states, 10 choices per state
	desired_fail_count = 100
	if bundle:
		# bundle takes longer, so use fewer epochs 
		minimum_fail_count = .001
		epochs_max = 9000
	else:
		epochs_max = 450000
		minimum_fail_count = 0.05
	expected_perr = 10**(-ie)  # expected probability of error
	desired_epochs = max(round(desired_fail_count / (expected_perr *num_transitions)), 2)
	# if ie == 7:
	# 	print("ie=%s, desired_epochs=%s, desired_fail_count=%s, expected_perr=%s" % (
	# 		ie, desired_epochs, desired_fail_count, expected_perr))
	if desired_epochs <= epochs_max:
		return desired_epochs
	minimum_epochs = round(minimum_fail_count / (expected_perr *num_transitions))
	if minimum_epochs <= epochs_max:
		return epochs_max
	# requires too many epochs
	return None


# mem_info = [
# 	{
# 		"name": "bun_k1000_d100_c1#S1",
# 		"short_name": "S1",
#  		"mtype": "bundle",
#  		"binarize_counters": True,  # use hamming match
#   	},
#   	{
# 		"name": "bun_k1000_d100_c8#S2",
# 		"short_name": "S2",
#  		"mtype": "bundle",
#  		"binarize_counters": False,  # used dot product match
#   	},
#   	{
#   		"name": "sdm_k1000_d100_c8_ham#A1",
# 		"short_name": "A1",
#  		"mtype": "sdm",
#  		"bits_per_counter": 8,
#  		"match_method": "hamming",
#  	},
#  	{
#   		"name": "sdm_k1000_d100_c1_ham#A2",
# 		"short_name": "A2",
#  		"mtype": "sdm",
#  		"bits_per_counter": 1,
#  		"match_method": "hamming",
#  	},
#  	{
#   		"name": "sdm_k1000_d100_c1_dot#A3",
# 		"short_name": "A3",
#  		"mtype": "sdm",
#  		"bits_per_counter": 1,
#  		"match_method": "dot",
#  	},
#  	{
#   		"name": "sdm_k1000_d100_c8_dot#A4",
# 		"short_name": "A4",
#  		"mtype": "sdm",
#  		"bits_per_counter": 8,
#  		"match_method": "dot",
#  	},
# ]

# def get_bundle_perr(mi, ie):
# 	# return mean and standard deviation
# 	# mi entry in mem_info array (key-value pairs)
# 	# ip is desired error, range 1 to 9 (10^(-ie))
# 	global mdims
# 	assert mi["mtype"] == "bundle"
# 	name = mi["name"]
# 	dims = mdims[name]
# 	size = dims[ie - 1]
# 	assert size[0] == ie, "First component of %s dims does not match error %s" % (name, ie)
# 	ncols = size[1]
# 	binarize_counters = mi["binarize_counters"]
# 	epochs = get_epochs(ie, bundle=True)
# 	if epochs is None:
# 		# requires too many epochs, don't run
# 		mean_error = math.nan
# 		std_error = math.nan
# 		clm_error = math.nan
# 	else:
# 		fbe = fast_bundle_empirical.Fast_bundle_empirical(ncols, epochs=epochs, binarize_counters=binarize_counters)
# 		mean_error = fbe.mean_error
# 		std_error = fbe.std_error
# 		clm_error = std_error / math.sqrt(epochs) * 1.96  # 95% confidence interval for the mean (CLM)
# 		print("%s, ie=%s, epochs=%s, error mean=%s, std=%s, clm=%s" % (name, ie, epochs, mean_error, std_error, clm_error))
# 	mi["results"]["sizes"].append(ncols)
# 	mi["results"]["mean_errors"].append(mean_error)
# 	mi["results"]["std_errors"].append(std_error)
# 	mi["results"]["clm_errors"].append(clm_error)

# def get_sdm_perr(mi, ie):
# 	# return mean and standard deviation of error
# 	# mi entry in mem_info array (key-value pairs)
# 	# ie is desired error, range 1 to 9 (10^(-ie))
# 	global mdims
# 	assert mi["mtype"] == "sdm"
# 	name = mi["name"]
# 	dims = mdims[name]
# 	size = dims[ie - 1]
# 	assert size[0] == ie, "First component of %s dims does not match error %s" % (name, ie)
# 	nrows = size[1]
# 	nact = size[2]
# 	bits_per_counter = mi["bits_per_counter"]
# 	assert mi["match_method"] in ("hamming", "dot")
# 	epochs = get_epochs(ie, bundle=False)
# 	if epochs is None:
# 		# requires too many epochs, don't run
# 		mean_error = math.nan
# 		std_error = math.nan
# 		clm_error = math.nan
# 	else:
# 		threshold_sum = mi["match_method"] == "hamming"
# 		ncols = 512
# 		fse = fast_sdm_empirical.Fast_sdm_empirical(nrows, ncols, nact, epochs=epochs,
# 			bits_per_counter=bits_per_counter, threshold_sum=threshold_sum)
# 		mean_error = fse.mean_error
# 		std_error = fse.std_error
# 		clm_error = std_error / math.sqrt(epochs) * 1.96  # 95% confidence interval for the mean (CLM)
# 		print("%s, ie=%s, epochs=%s, error mean=%s, std=%s, clm=%s" % (name, ie, epochs, mean_error, std_error, clm_error))
# 	mi["results"]["sizes"].append(nrows)
# 	mi["results"]["mean_errors"].append(mean_error)
# 	mi["results"]["std_errors"].append(std_error)
# 	mi["results"]["clm_errors"].append(clm_error)


# def plot_fit(mtype="sdm"):
# 	assert mtype in ("sdm", "bundle")
# 	get_perr = get_sdm_perr if mtype=="sdm" else get_bundle_perr
# 	for mi in mem_info:
# 		if mi["mtype"] == mtype:
# 			mi["results"] = {"sizes":[], "mean_errors":[], "std_errors":[], "clm_errors":[]}
# 			for ie in range(1, 10):
# 				get_perr(mi, ie) # calculates and appends to variables above in "results"
# 	# now make plot
# 	desired_errors = [10**(-i) for i in range(1, 10)]
# 	for mi in mem_info:
# 		if mi["mtype"] == mtype:
# 			name = mi["name"]
# 			x = mi["results"]["sizes"]
# 			y = mi["results"]["mean_errors"]
# 			ybar = mi["results"]["clm_errors"]
# 			plt.errorbar(x, y, yerr=ybar, fmt="-o", label=name)
# 			plt.errorbar(x, desired_errors, yerr=None, fmt="o", label="%s - Desired error" % name)
# 	plt.title("%s empirical vs. desired error" % mtype)
# 	xlabel = "sdm num rows" if mtype == "sdm" else "Superposition vector width"
# 	plt.xlabel("xlabel")
# 	plt.ylabel("Fraction error")
# 	plt.yscale('log')
# 	# xlabels = ["%s/%s" % (rows[i], nacts[i]) for i in range(num_steps)]
# 	# plt.xticks(rows[0:num_steps], xlabels)
# 	plt.grid()
# 	plt.legend(loc='upper right')
# 	plt.show()

def main():
	# plt.ion()
	edb = Empirical_error_db()
	mem_names = edb.get_memory_names()
	with Pool(6) as p:
		p.map(fill_eedb, mem_names)
	# fill_eedb()
	# plt.show()  # keep any plots open
	# plot_fit("sdm")
	# plot_fit("bundle")
	# edb = Empirical_error_db()
	# mem_name = "bun_k1000_d100_c1#S1"
	# mi = edb.get_minfo(mem_name)
	# print("minfo for %s is:" % mem_name)
	# pp.pprint(mi)


if __name__ == "__main__":
	# compare_sdm_ham_dot()
	main()

