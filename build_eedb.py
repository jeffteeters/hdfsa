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
		return con



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
	print("ie=%s, desired_epochs=%s, desired_fail_count=%s, expected_perr=%s" % (
		ie, desired_epochs, desired_fail_count, expected_perr))
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
	# plot_fit("sdm")
	# plot_fit("bundle")
	edb = Empirical_error_db()



if __name__ == "__main__":
	# compare_sdm_ham_dot()
	main()

