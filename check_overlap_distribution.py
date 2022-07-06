
# compare overlap disribution mean and variance to predicted which should be
# Poisson (mean and variance the same).  I think it shows they are (at least for some
# cases, probably large number overlaps).  Histogram plot has empty columns (broken),
# not sure why.

import numpy as np
import matplotlib.pyplot as plt

def get_overlaps(nrows, k, nact):
	# get overlaps when storing k items into nrows with nact rows selected
	rows = np.zeros(nrows, dtype=np.int32)
	rng = np.random.default_rng()
	for i in range(k):
		hard_locations = rng.choice(nrows, size=nact, replace=False)
		rows[hard_locations] += 1
	return rows

def get_overlap_counts(nrows, k, nact):
	# get count of each number of overlaps when storing k items into nrows with nact
	# rows selected
	rows = get_overlaps(nrows, k, nact)
	ave_num_hits = nact * k / nrows
	overlap_counts = np.zeros(int(ave_num_hits * 10), dtype=np.int32)
	for i in range(nrows):
		overlap_counts[rows[i]] += 1
	# trim zeros from end
	non_zeros = np.where(overlap_counts!=0)[0]
	last = non_zeros[-1]+1
	return overlap_counts[0:last]

def plot_histogram(vals):
	nbins = get_nbins(vals)
	n, bins, patches = plt.hist(vals, nbins, density=False, facecolor='g', alpha=0.75)
	plt.show()

# def plot_margin_hist(vals, title, margin_mean, margin_std, size, trial_count):
# 	nbins = get_nbins(vals)
# 	n, bins, patches = plt.hist(vals, nbins, density=False, facecolor='g', alpha=0.75)
# 	# from: https://www.geeksforgeeks.org/how-to-plot-normal-distribution-over-histogram-in-python/
# 	mu, std = norm.fit(vals)
# 	print("size=%s, mu=%s, margin_mean=%s, std=%s, margin_std=%s" %(size, mu, margin_mean, std, margin_std))
  
# 	# Plot the histogram.
# 	# plt.hist(data, bins=25, density=True, alpha=0.6, color='b')
  
# 	# Plot the PDF.
# 	xmin, xmax = plt.xlim()
# 	x = np.linspace(xmin, xmax, 100)
# 	p = norm.pdf(x, mu, std) * trial_count
  
# 	plt.plot(x, p, 'k', linewidth=2)
# 	title = ("%s, Fit Values: {:.2f} and {:.2f}".format(mu, std)) % title
# 	plt.title(title)

# 	plt.xlabel('Margin value')
# 	plt.ylabel('Count')
# 	plt.title(title)
# 	# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
# 	# plt.xlim(40, 160)
# 	# plt.ylim(0, 0.03)
# 	plt.grid(True)
# 	plt.show()

def get_nbins(xv):
	# calculate number of bins to use in histogram for values in xv
	xv_range = max(xv) - min(xv)
	if xv_range == 0:
		nbins = 3
	else:
		nbins = int(xv_range) + 1
	# if xv_range < 10:
	#   nbins = 10
	# elif xv_range > 100:
	#   nbins = 100
	# else:
	#   nbins = int(xv_range) + 1
	return nbins


def main():
	nrows = 51
	nact = 7
	k = 1000
	# overlap_counts = get_overlap_counts(nrows, k, nact)
	# print("overlap_counts=%s" % overlap_counts)
	overlaps = get_overlaps(nrows, k, nact)
	predicted_mean = nact*k / nrows
	print("mean=%s, variance=%s, predicted_mean=%s" % (np.mean(overlaps), np.var(overlaps), predicted_mean))
	plot_histogram(overlaps)

main()