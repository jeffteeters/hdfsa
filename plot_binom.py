from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt
import math

def plot_hamming(ncols=225):
	# alculate probability of different Hamming distances between two vectors
	x = np.arange(ncols + 1)  # all possible hamming distances between two random vectors
	pham = binom.pmf(x, ncols, 0.5)  # probability each hamming distance
	max_pham = pham.max()    # maximum probability
	cutoff_threshold = 10**6
	itemindex = np.where(pham > max_pham/cutoff_threshold)
	# find range of probabilities above the cutoff_threshold
	from_idx = itemindex[0][0]
	to_idx = itemindex[0][-1]
	width = to_idx - from_idx + 1
	total = pham[from_idx:to_idx+1].sum()
	print("most probable Hamming distances, from_idx=%s, to_idx=%s, width=%s, total=%s" % (
		from_idx, to_idx, width, total))

	fontsize=12
	fig, ax = plt.subplots()
	ax.plot(pham)
	ax.annotate('$h_{from}$', xy=(from_idx, 0.0), xytext=(from_idx, max_pham/7.0), horizontalalignment="center",
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=10), fontsize=fontsize)
	ax.annotate('$h_{to}$', xy=(to_idx, 0.0), xytext=(to_idx, max_pham/7.0), horizontalalignment="center",
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=10), fontsize=fontsize)
	plt.title("Hamming distance PMF ($p_{ham}$) if ncols=%s" % ncols)
	plt.xlabel("Hamming distance")
	plt.ylabel("Probability of Hamming distance")
	plt.show()


plot_hamming()