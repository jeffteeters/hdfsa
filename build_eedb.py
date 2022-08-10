# build (or update) empirical error database (database of empiricl errors)

import fast_sdm_empirical
import fast_bundle_empirical
import numpy as np
import matplotlib.pyplot as plt
import os.path
import sqlite3
import math
from scipy.stats import linregress
from mdie import mdie
import pprint
pp = pprint.PrettyPrinter(indent=4)


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
	)
	"""


	def __init__(self, dbfile="empirical_error.db"):
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
			sql = ("select dim.id, ie, size, ncols, nact, pe, epochs, mean, std from dim left join stats on"
				" dim.id = stats.dim_id where mem_id = %s order by ie" % mem_id)
		else:
			sql = ("select dim.id, ie, size, pe, epochs, mean, std from dim left join stats on"
				" dim.id = stats.dim_id where mem_id = %s order by ie" % mem_id)
		cur = self.con.cursor()
		res = cur.execute(sql)
		dims = res.fetchall()
		info = {"mem_id": mem_id, "short_name": short_name, "mtype": mtype, 
			"bits_per_counter":bits_per_counter, "match_method": match_method, "dims":dims}
		return info

	def get_memory_names(self):
		# return list of memory names
		sql = "select name from memory order by id"
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
			sql = "update states set epochs=%s, mean=%s, std=%s where id = %s" % stat_id
		cur = self.con.cursor()
		res = cur.execute(sql)
		self.con.commit()

	def add_error(self, dim_id, nerrors, nepochs):
		# add errors to error table
		# if dim_id already has nerrors in table, add nepochs to value stored 
		sql = "select id from error where dim_id = %s" % dim_id
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

	def calculate_stats(self, dim_id):
		# use entries in error table to calculate stats
		sql = "select nerrors, nepochs from error where dim_id = %s" % dim_id
		cur = self.con.cursor()
		res = cur.execute(sql)
		nee = res.fetchall()
		if len(nee) == 0:
			# no entries for this dim, don't calculate anything
			return
		values = np.empty(len(nee), dtype=np.uint32)
		weights = np.empty(len(nee), dtype=np.uint32)
		for i in range(len(nee)):
			values[i], weights[i] = nee[i]
		# calculate mean and std using method at:
		# https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
		mean, epochs = np.average(values, weights=weights, returned=True)
		# Fast and numerically precise:
		variance = np.average((values-mean)**2, weights=weights)
		std = math.sqrt(variance)
		self.add_stats(dim_id, epochs, mean, std)
		return (mean, std)

def get_bundle_ee(ncols, bits_per_counter, match_method, needed_epochs):
	# return bundle_empirical_error object
	assert bits_per_counter == 8
	assert match_method in ("dot", "hamming")
	binarize_counters = match_method == "hamming"
	fbe = fast_bundle_empirical.Fast_bundle_empirical(ncols, epochs=needed_epochs, binarize_counters=binarize_counters)
	return fbe

def get_sdm_ee(nrows, ncols, nact, bits_per_counter, match_method, needed_epochs):
	# return sdm_empirical_error object
	assert bits_per_counter in (1, 8)
	assert match_method in ("dot", "hamming")
	assert ncols == 512  # for current simulations
	threshold_sum = match_method == "hamming"
	fse = fast_sdm_empirical.Fast_sdm_empirical(nrows, ncols, nact, epochs=needed_epochs,
			bits_per_counter=bits_per_counter, threshold_sum=threshold_sum)
	return fse

def fill_eedb():
	# main routine to populate the Empirical Error database
	edb = Empirical_error_db()
	mem_names = edb.get_memory_names()
	for name in mem_names:
		mi = edb.get_minfo(name)
		mtype = mi["mtype"]  # "bundle" or "sdm"
		bits_per_counter = mi["bits_per_counter"]
		match_method = mi["match_method"]
		for dim in mi["dims"]:
			if mtype == "sdm":
				dim_id, ie, nrows, ncols, nact, pe, epochs, mean, std = dim
				wanted_epochs = get_epochs(ie, bundle=False)
			else:
				dim_id, ie, ncols, pe, epochs, mean, std = dim
				wanted_epochs = get_epochs(ie, bundle=True)
			if epochs is None:
				epochs = 0  # have no epochs
			if wanted_epochs is not None and wanted_epochs > epochs:
				# need more epochs
				needed_epochs = wanted_epochs if ie < 4 else wanted_epochs - epochs # if ie < 4, store stats directly
				if mtype == "sdm":
					fee = get_sdm_ee(nrows, ncols, nact, bits_per_counter, match_method, needed_epochs)
				else:
					fee = get_bundle_ee(ncols, bits_per_counter, match_method, needed_epochs) # find empirical error
				if ie < 4:
					# store stats directly
					edb.add_stats(dim_id, wanted_epochs, fee.mean_error, fee.std_error)
					print("%s ie=%s, fresh epochs=%s, mean=%s, std=%s" % (name, ie, wanted_epochs, fee.mean_error, fee.std_error))
				else:
					# add errors to error table and recalculate stats
					print("%s adding multi_error, mean=%s, std=%s" % (name, fee.mean_error, fee.std_error))
					import pdb; pdb.set_trace()
					edb.add_multi_error(dim_id, fee.fail_counts)
					mean, std = edb.calculate_stats(dim_id)
					print("%s ie=%s, added epochs=%s, mean=%s, std=%s" % (name, ie, wanted_epochs, mean, std))



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

def get_bundle_perr(mi, ie):
	# return mean and standard deviation
	# mi entry in mem_info array (key-value pairs)
	# ip is desired error, range 1 to 9 (10^(-ie))
	global mdims
	assert mi["mtype"] == "bundle"
	name = mi["name"]
	dims = mdims[name]
	size = dims[ie - 1]
	assert size[0] == ie, "First component of %s dims does not match error %s" % (name, ie)
	ncols = size[1]
	binarize_counters = mi["binarize_counters"]
	epochs = get_epochs(ie, bundle=True)
	if epochs is None:
		# requires too many epochs, don't run
		mean_error = math.nan
		std_error = math.nan
		clm_error = math.nan
	else:
		fbe = fast_bundle_empirical.Fast_bundle_empirical(ncols, epochs=epochs, binarize_counters=binarize_counters)
		mean_error = fbe.mean_error
		std_error = fbe.std_error
		clm_error = std_error / math.sqrt(epochs) * 1.96  # 95% confidence interval for the mean (CLM)
		print("%s, ie=%s, epochs=%s, error mean=%s, std=%s, clm=%s" % (name, ie, epochs, mean_error, std_error, clm_error))
	mi["results"]["sizes"].append(ncols)
	mi["results"]["mean_errors"].append(mean_error)
	mi["results"]["std_errors"].append(std_error)
	mi["results"]["clm_errors"].append(clm_error)

def get_sdm_perr(mi, ie):
	# return mean and standard deviation of error
	# mi entry in mem_info array (key-value pairs)
	# ie is desired error, range 1 to 9 (10^(-ie))
	global mdims
	assert mi["mtype"] == "sdm"
	name = mi["name"]
	dims = mdims[name]
	size = dims[ie - 1]
	assert size[0] == ie, "First component of %s dims does not match error %s" % (name, ie)
	nrows = size[1]
	nact = size[2]
	bits_per_counter = mi["bits_per_counter"]
	assert mi["match_method"] in ("hamming", "dot")
	epochs = get_epochs(ie, bundle=False)
	if epochs is None:
		# requires too many epochs, don't run
		mean_error = math.nan
		std_error = math.nan
		clm_error = math.nan
	else:
		threshold_sum = mi["match_method"] == "hamming"
		ncols = 512
		fse = fast_sdm_empirical.Fast_sdm_empirical(nrows, ncols, nact, epochs=epochs,
			bits_per_counter=bits_per_counter, threshold_sum=threshold_sum)
		mean_error = fse.mean_error
		std_error = fse.std_error
		clm_error = std_error / math.sqrt(epochs) * 1.96  # 95% confidence interval for the mean (CLM)
		print("%s, ie=%s, epochs=%s, error mean=%s, std=%s, clm=%s" % (name, ie, epochs, mean_error, std_error, clm_error))
	mi["results"]["sizes"].append(nrows)
	mi["results"]["mean_errors"].append(mean_error)
	mi["results"]["std_errors"].append(std_error)
	mi["results"]["clm_errors"].append(clm_error)

def get_epochs(ie, bundle=False):
	# ie is expected error rate, range 1 to 9 (10^(-ie))
	# return number of epochs required or None if not running this one because would require too many epochs
	num_transitions = 1000  # 100 states, 10 choices per state
	desired_fail_count = 100
	minimum_fail_count = 40
	epochs_max = 1000 if not bundle else 5  # bundle takes longer, so use fewer epochs 
	expected_perr = 10**(-ie)  # expected probability of error
	desired_epochs = max(round(desired_fail_count / (expected_perr *num_transitions)), 2)
	# print("ie=%s, desired_epochs=%s, desired_fail_count=%s, expected_perr=%s" % (
	# 	ie, desired_epochs, desired_fail_count, expected_perr))
	if desired_epochs <= epochs_max:
		return desired_epochs
	minimum_epochs = round(minimum_fail_count / (expected_perr *num_transitions))
	if minimum_epochs <= epochs_max:
		return epochs_max
	# requires too many epochs
	return None

def plot_fit(mtype="sdm"):
	assert mtype in ("sdm", "bundle")
	get_perr = get_sdm_perr if mtype=="sdm" else get_bundle_perr
	for mi in mem_info:
		if mi["mtype"] == mtype:
			mi["results"] = {"sizes":[], "mean_errors":[], "std_errors":[], "clm_errors":[]}
			for ie in range(1, 10):
				get_perr(mi, ie) # calculates and appends to variables above in "results"
	# now make plot
	desired_errors = [10**(-i) for i in range(1, 10)]
	for mi in mem_info:
		if mi["mtype"] == mtype:
			name = mi["name"]
			x = mi["results"]["sizes"]
			y = mi["results"]["mean_errors"]
			ybar = mi["results"]["clm_errors"]
			plt.errorbar(x, y, yerr=ybar, fmt="-o", label=name)
			plt.errorbar(x, desired_errors, yerr=None, fmt="o", label="%s - Desired error" % name)
	plt.title("%s empirical vs. desired error" % mtype)
	xlabel = "sdm num rows" if mtype == "sdm" else "Superposition vector width"
	plt.xlabel("xlabel")
	plt.ylabel("Fraction error")
	plt.yscale('log')
	# xlabels = ["%s/%s" % (rows[i], nacts[i]) for i in range(num_steps)]
	# plt.xticks(rows[0:num_steps], xlabels)
	plt.grid()
	plt.legend(loc='upper right')
	plt.show()

def main():
	fill_eedb()
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

