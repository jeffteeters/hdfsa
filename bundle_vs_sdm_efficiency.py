import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from labellines import labelLine, labelLines
import sys


def bundle_efficiency(i):
    # effeciency of bundle (superposition memory)
    # i is the number of items in item memory
    return 1 / (i + 1)


def sdm_efficiency(i, r):
    # effeciency of sparse distibuted memory
    # i is the number of items in item memory
    # r is the number of rows in the sdm contents matrix
    return 1 / (1 + i/r + 1/8)

def make_plot(scale="linear"):
    assert scale in ("linear", "log")

    log_scale = scale == "log"


    # fig = plt.figure()
    fig1, ax1 = plt.subplots()

    ivals = np.arange(2, 201, 2)
    # rvals = np.arange(2, 2048)
    rvals = [10, 20, 40, 80, 160, 320, 640, 1280, 5120]

    bun_eff = [bundle_efficiency(i)*100 for i in ivals]

    # plt.errorbar(ivals, bun_eff, yerr=None, label='B') # fmt="-k"
    ax1.errorbar(ivals, bun_eff, yerr=None, label='B') # fmt="-k"

    for r in rvals:
        sdm_eff = [sdm_efficiency(i, r)*100 for i in ivals]
        ax1.errorbar(ivals, sdm_eff, yerr=None, label='r=%s' % r, linewidth=1, fmt="-k",) # linestyle='dashed',
        # plt.errorbar(ivals, sdm_eff, yerr=None, label='r=%s' % r, linewidth=1, fmt="-k",) # linestyle='dashed',  

    # labelLines(plt.gca().get_lines(), zorder=2.5)
    labelLines(plt.gca().get_lines(), zorder=2.5)

    if log_scale:
        plt.xscale('log')
        log_msg = " (log scale)"
        plt.xticks([2, 3, 4, 5, 10, 20, 40, 100, 200])
        ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    else:
        log_msg = ""
        plt.xticks([2, 25, 50, 75, 100, 125, 150, 175, 200])

    plt.title("Memory efficiency of superposition vector (B) and SDM with r rows" + log_msg)
    plt.xlabel("Number of vectors in item memory (i)")
    plt.ylabel("Percent total storage used for memory")
    # plt.xlim(xmin=2)
    plt.yticks(np.arange(0, 91, 10))
    ax1.grid()
    # plt.legend(loc='upper right')
    plt.show()

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("log", "linear"):
        sys.exit("Usage: %s [ log | linear]" % sys.argv[0])
    scale = sys.argv[1]
    make_plot(scale)

main()

