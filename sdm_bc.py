# script to generate plot of SDM recall performance vs bit counter widths
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from labellines import labelLine, labelLines
import sys

def sdm_error(size, bc):
	err = size * (1- bc / 8)
	nrows = size* bc / 10
	nact = size*bc*2
	info={"err":err, "nrows":nrows, "nact":nact}
	return info

def plot_info(sizes, bc_vals, resp_info):
	# fig1, ax1 = plt.subplots()
	# fig1, ax1 = plt.subplot(221)

	# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
	# fig.suptitle('SDM performance with different counter sizes')
	# ax1.plot(x, y)
	# ax2.plot(x, y**2, 'tab:orange')
	# ax3.plot(x, -y, 'tab:green')
	# ax4.plot(x, -y**2, 'tab:red')

	# for ax in fig.get_axes():
	#     ax.label_outer()
	plots_info = [
		{"subplot": 221, "key":"err","title":"SDM error with different counter bits", "ylabel":"Recall error"},
		{"subplot": 222, "key":"nrows","title":"Number rows in SDM vs. size and counter bits","ylabel":"Number rows"},
		{"subplot": 223, "key":"nact","title":"SDM activation count vs counter bits and size","ylabel":"Activation Count"},
		 ]
	for pi in plots_info:
		plt.subplot(pi["subplot"])
		log_scale = "scale" in pi and pi["scale"] == "log"
		yvals = [resp_info[bc_vals[0]][i][pi["key"]] for i in range(len(sizes))]
		plt.errorbar(sizes, yvals, yerr=None, label="%s bit" % bc_vals[0]) # fmt="-k"
		for i in range(1, len(bc_vals)):
			yvals = [resp_info[bc_vals[i]][j][pi["key"]] for j in range(len(sizes))]
			plt.errorbar(sizes, yvals, yerr=None, label='%s bit' % bc_vals[i], linewidth=1, fmt="-k",) # linestyle='dashed',
		labelLines(plt.gca().get_lines(), zorder=2.5)
		if log_scale:
			plt.xscale('log')
			log_msg = " (log scale)"
			# plt.xticks([2, 3, 4, 5, 10, 20, 40, 100, 200])
			ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
		else:
			log_msg = ""
			# plt.xticks([2, 25, 50, 75, 100, 125, 150, 175, 200])
		plt.title(pi["title"]+log_msg)
		plt.xlabel("Size (kB)")
		plt.ylabel(pi["ylabel"])
		plt.grid()
	plt.show()
	return

	# plt.xlim(xmin=2)
	# plt.yticks(np.arange(0, 91, 10))


	# plot error
	plt.subplot(221)
	err = [resp_info[bc_vals[0]][i]["err"] for i in range(len(sizes))]
	plt.errorbar(sizes, err, yerr=None, label="%s bit" % bc_vals[0]) # fmt="-k"
	for i in range(1, len(bc_vals)):
		err = [resp_info[bc_vals[i]][j]["err"] for j in range(len(sizes))]
		plt.errorbar(sizes, err, yerr=None, label='%s bit' % bc_vals[i], linewidth=1, fmt="-k",) # linestyle='dashed',
	# labelLines(plt.gca().get_lines(), zorder=2.5)
	labelLines(plt.gca().get_lines(), zorder=2.5)

	if log_scale:
		plt.xscale('log')
		log_msg = " (log scale)"
		# plt.xticks([2, 3, 4, 5, 10, 20, 40, 100, 200])
		ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
	else:
		log_msg = ""
		# plt.xticks([2, 25, 50, 75, 100, 125, 150, 175, 200])
	plt.title("SDM error with different counter bits")
	# ax1.set_title("SDM error with different counter bits")
	# plt.xlabel("Size (kB)")
	plt.xlabel("Size (kB)")
	plt.ylabel("Recall error")
	# plt.xlim(xmin=2)
	# plt.yticks(np.arange(0, 91, 10))
	plt.grid()
	# plt.legend(loc='upper right')
	# plt.show()

	# plot num rows
	plt.subplot(222)
	nrows = [resp_info[bc_vals[0]][i]["nrows"] for i in range(len(sizes))]
	plt.errorbar(sizes, nrows, yerr=None, label="%s bit" % bc_vals[0]) # fmt="-k"
	for i in range(1, len(bc_vals)):
		nrows = [resp_info[bc_vals[i]][j]["nrows"] for j in range(len(sizes))]
		plt.errorbar(sizes, nrows, yerr=None, label='%s bit' % bc_vals[i], linewidth=1, fmt="-k",) # linestyle='dashed',
	# labelLines(plt.gca().get_lines(), zorder=2.5)
	labelLines(plt.gca().get_lines(), zorder=2.5)
	plt.title("Number rows in SDM vs. size and counter bits")
	plt.xlabel("Size (kB)")
	plt.ylabel("Num rows")
	# plt.xlim(xmin=2)
	# plt.yticks(np.arange(0, 91, 10))
	plt.grid()

	# plot nact
	plt.subplot(223)
	nact = [resp_info[bc_vals[0]][i]["nact"] for i in range(len(sizes))]
	plt.errorbar(sizes, nrows, yerr=None, label="%s bit" % bc_vals[0]) # fmt="-k"
	for i in range(1, len(bc_vals)):
		nrows = [resp_info[bc_vals[i]][j]["nrows"] for j in range(len(sizes))]
		plt.errorbar(sizes, nrows, yerr=None, label='%s bit' % bc_vals[i], linewidth=1, fmt="-k",) # linestyle='dashed',
	# labelLines(plt.gca().get_lines(), zorder=2.5)
	labelLines(plt.gca().get_lines(), zorder=2.5)
	plt.title("Number rows in SDM vs. size and counter bits")
	plt.xlabel("Size (kB)")
	plt.ylabel("Num rows")
	# plt.xlim(xmin=2)
	# plt.yticks(np.arange(0, 91, 10))
	plt.grid()
	# plt.legend(loc='upper right')
	plt.show()


def main(start_size=50000, step_size=25000, stop_size=200001, bc_vals=[1,2,3,4,5,8]):
	resp_info = {}
	sizes = range(start_size, stop_size, step_size)
	for bc in bc_vals:
		resp_info[bc] = [sdm_error(size, bc) for size in sizes]
	# make plot
	plot_info(sizes, bc_vals, resp_info)


	# for size in sizes:
	# 	resp_info[size] = [sdm_error(size, bc) for bc in bc_vals]




if __name__ == "__main__":
	main()