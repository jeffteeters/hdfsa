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
    # return 1 / (1 + i/r + 1/8)  # previous version, incorrect
    m = r
    # return 1/ (1 + i/(m * 8) + 1/8)
    # assume only one bit used, others are not
    return 1 / (9 + i/r)

def sdm_efficiency_bm(i, r, b_m=5, b_c=8):
    # compute efficiency assuming bm bits in each counter are memory bits, and bc-bm bits are fixed bits
    m = r  # number rows in sdm memory
    return  b_m / ( b_c  + i/m + 1) 

def make_mem_plot(scale="linear"):
    assert scale in ("linear", "log")

    log_scale = scale == "log"


    # fig = plt.figure()
    fig1, ax1 = plt.subplots()

    ivals = np.arange(2, 201, 2)
    # rvals = np.arange(2, 2048)
    rvals = [5, 10, 20, 40, 80, 161, 320, 1280]

    bun_eff = [bundle_efficiency(i)*100 for i in ivals]

    # plt.errorbar(ivals, bun_eff, yerr=None, label='B') # fmt="-k"
    ax1.errorbar(ivals, bun_eff, yerr=None, label='B') # fmt="-k"

    for r in rvals:
        sdm_eff = [sdm_efficiency_bm(i, r)*100 for i in ivals]
        ax1.errorbar(ivals, sdm_eff, yerr=None, label='m=%s' % r, linewidth=1, fmt="-k",) # linestyle='dashed',
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

    plt.title("Memory efficiency of superposition vector (B) and SDM with m rows" + log_msg)
    plt.xlabel("Number of vectors in item memory (i)")
    plt.ylabel("Percent total storage used for memory")
    # plt.xlim(xmin=2)
   # plt.yticks(np.arange(0, 91, 10))
    ax1.grid()
    # plt.legend(loc='upper right')
    plt.show()

# def cb_over_cc(i_s, r, i_a=10, wc=512):
#     i = i_a + i_s
#     cb = wc*((9*r+i)/(i+1)*(i_s+1)/8)
#     cc = wc*(r/8 +)

def get_sdm_activation_count(m, k):
        # compute number of rows in sdm to be active
        # m is number of rows in sdm.  k is number of items being stored (1000)
        # nact = round(sdm_num_rows / 100)  # originally used in hdfsa.py
        nact = round( m/((2*m*k)**(1/3)) )
        return nact

def make_cal_plot(scale="linear"):
    # plot of number of operations (calculations) used for sizes used in capacity plots in Frontiers paper
    assert scale in ("linear", "log")

    log_scale = scale == "log"

    # sizes used to make capacity plots.  (#kB, bundle_length, sdm_num_rows)
    cap_sizes = [(100, 7207, 161), (200, 14414, 335), (300, 21622, 509), (400, 28829, 682), (500, 36036, 856),
        (600, 43243, 1029), (700, 50450, 1203), (800, 57658, 1377), (900, 64865, 1550), (1000, 72072, 1724)]

    k = 1000  # number of items saved in memory (bundle or sdm)
    i_s = 100  # number of states in item memory
    w_c = 512  # width of each vector in sdm
    sizes = []
    bun_ops = []
    sdm_ops = []
    bun_sdm_ratio = []
    for cs in cap_sizes:
        size, bl, nrows = cs
        nact = get_sdm_activation_count(nrows, k)
        bundle_ops = bl * (i_s + 1) / 8
        sdm_tops = w_c *(nrows / 8 + nact + 1 + i_s / 8)
        ratio = bundle_ops / sdm_tops
        sizes.append(size)
        bun_ops.append(bundle_ops)
        sdm_ops.append(sdm_tops)
        bun_sdm_ratio.append(ratio)

    # fig = plt.figure()
    fig1, ax1 = plt.subplots()

    # plt.errorbar(ivals, bun_eff, yerr=None, label='B') # fmt="-k"
    ax1.errorbar(sizes, bun_ops, yerr=None, label='Superposition') # fmt="-k"
    ax1.errorbar(sizes, sdm_ops, yerr=None, label='Superposition') # fmt="-k"


    plt.title("Number operations for superposition and SDM during recall")
    plt.xlabel("Storage (bytes)")
    plt.ylabel("Number of operations")
    xaxis_labels = ["100k", "200k", "300k", "400k", "500k", "600k", "700k", "800k", "900k", "10^6" ]
    plt.xticks(sizes,xaxis_labels)
    # plt.xlim(xmin=2)
   # plt.yticks(np.arange(0, 91, 10))
    ax1.grid()
    # plt.legend(loc='upper right')
    plt.show()

    # plot ratio
    fig2, ax2 = plt.subplots()
    ax2.errorbar(sizes, bun_sdm_ratio, yerr=None, label='Bundle ops / sdm ops')
    plt.title("Ratio of number of computations for superposition over SDM per recall")
    xaxis_labels = ["100k", "200k", "300k", "400k", "500k", "600k", "700k", "800k", "900k", "10^6" ]
    ax2.grid()
    plt.xticks(sizes,xaxis_labels)
    plt.show()


def get_bundle_ops(wc, i, r, i_s):
    bun_ops = wc * (9 * r + i) * (i_s + 1) / ((i+1) * 8)
    return bun_ops

def get_sdm_ops(wc, i, r, i_s, nact):
    sdm_ops = wc * (r / 8 + nact + 1 + i_s / 8)
    return sdm_ops


def make_cali_plot(scale="linear"):
    assert scale in ("linear", "log")

    log_scale = scale == "log"


    # fig = plt.figure()
    fig1, ax1 = plt.subplots()

    ivals = np.arange(2, 403, 4)
    # rvals = np.arange(2, 2048)
    rvals = [5, 10, 20, 40, 80, 160, 320, 1280]


    # plt.errorbar(ivals, bun_eff, yerr=None, label='B') # fmt="-k"
    # ax1.errorbar(ivals, bun_eff, yerr=None, label='B') # fmt="-k"

    k = 1000  # number of items saved in memory (bundle or sdm)
    # i_s = 100  # number of states in item memory
    wc = 512  # width of each vector in sdm
    i_a = 10
    for r in rvals:
        nact = nact = get_sdm_activation_count(r, k)
        bun_ops = [get_bundle_ops(wc, i_a+i_s, r, i_s) for i_s in ivals]
        sdm_ops = [get_sdm_ops(wc, i_a+i_s, r, i_s, nact) for i_s in ivals]
        ratio = [bun_ops[i] / sdm_ops[i] for i in range(len(bun_ops))]
        ax1.errorbar(ivals, ratio, yerr=None, label='m=%s' % r, linewidth=1, fmt="-k",) # linestyle='dashed',
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

    plt.title("Ratio of superposition operations vs sdm operations" + log_msg)
    plt.xlabel("Number of vectors in item memory (i)")
    plt.ylabel("Ratio of operations")
    # plt.xlim(xmin=2)
   # plt.yticks(np.arange(0, 91, 10))
    ax1.grid()
    # plt.legend(loc='upper right')
    plt.show()

# def cb_over_cc(i_s, r, i_a=10, wc=512):
#     i = i_a + i_s
#     cb = wc*((9*r+i)/(i+1)*(i_s+1)/8)
#     cc = wc*(r/8 +)





def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("mem", "cal", "cali") or sys.argv[2] not in ("log", "linear"):
        print("plot calculations required or memory effeciency")
        sys.exit("Usage: %s (mem | cal | cali) (log | linear)" % sys.argv[0])
    mode = sys.argv[1]
    scale = sys.argv[2]
    if mode == "mem":
        make_mem_plot(scale)
    elif mode == "cal":
        make_cal_plot(scale)
    elif mode == "cali":
        k = 1000  # number of items saved in memory (bundle or sdm)
        i_s = 100  # number of states in item memory
        wc = 512  # width of each vector in sdm
        i_a = 10
        r = nrows = 1724
        nact = get_sdm_activation_count(r, k)
        i = 110
        bun_ops = get_bundle_ops(wc, i_s+i_a, r, i_s)
        sdm_ops = get_sdm_ops(wc, i_s+i_a, r, i_s, nact)
        print("bun_ops=%s, sdm_ops=%s" % (bun_ops, sdm_ops))
        make_cali_plot(scale)

main()

