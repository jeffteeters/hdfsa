from scipy import special
import math
import scipy.integrate as integrate
import numpy as np
from scipy.stats import norm
import scipy


def gallen(s, per):
	# calculate required length using equation in Gallant paper
	# solved to give length.  (Page 39: "This is equivalent to the probability
	# of a random vector ["per"] being negative when selected from
	# NormalDistibution(D, √((2S-1)D)" solved for D in terms of per and s 
	return round(2*(-1 + 2*s)*special.erfcinv(2*per)**2)

def dplen(mm, per):
	# calculate vector length requred to store bundle at per accuracy
	# mm - mean of match distribution (single bit error rate, 0< mm < 0.5)
	# per - desired probability of error on recall (e.g. 0.000001)
	# This derived from equations in Pentti's 1997 paper, "Fully Distributed
	# Representation", page 3, solving per = probability random vector has
	# smaller hamming distance than hamming to matching vector, (difference
	# in normal distributions) is less than zero; solve for length (N)
	# in terms of per and mean distance to match (mm, denoted as delta in the
	# paper.)
	n = (-2*(-0.25 - mm + mm**2)*special.erfinv(-1 + 2*per)**2)/(0.5 - mm)**2
	return round(n) + 1

def dplen2(mm, per, D):
	# calculate vector length based on equation 2.25 in the Frady paper
	beta = 1.08
	s2 = (4*(math.log(D-1) - math.log(2*per) + math.log((math.sqrt(beta-1)*math.sqrt((2*math.e)/math.pi))/beta)))/beta
	n = s2 * (mm*(1-mm) + 0.5*(1-0.5))/(0.5 - mm)**2
	# n = (s2*(1 + 4.*mm - 4.*mm**2))/(1. - 2.*mm)**2
	return round(n)

def DimensionalityAnalytical(acc,K,D):
    hamming_sup= expectedHamming(K)
    delHam=0.5-hamming_sup
    beta=1.08
    epsi=(1-acc)
    N =  (np.log(D-1) - np.log(2*epsi) +  np.log((np.sqrt(beta-1)*np.sqrt((2*np.e)/np.pi))/beta))/(beta*(delHam**2))  
    
    return np.round(N)


def cs2(per, D):
	beta = 1.08
	s2 = (4*(math.log(D-1) - math.log(2*per) + math.log((math.sqrt(beta-1)*math.sqrt((2*math.e)/math.pi))/beta)))/beta
	print("s2=%s, sqrt(s2)=%s" % (s2, math.sqrt(s2)))
	return s2


#Compute expected Hamming distance
def  expectedHamming (K):
    if (K % 2) == 0: # If even number then break ties so add 1
        K+=1
    deltaHam = 0.5 - (scipy.special.binom(K-1, 0.5*(K-1)))/2**K  # Pentti's formula for the expected Hamming distance
    return deltaHam


def bunlen(k, per):
	# calculated bundle length needed to store k items with accuracy per
	# This calculates the mean distance to the matching vector using
	# approximation on page 3 in Pentti's paper (referenced above)
	# then calls dplen to calculate the vector length.
	return dplen(0.5 - 0.4 / math.sqrt(k - 0.44), per)

def bunlenf(k, perf, n):
	# bundle length from final probabability error (perf) taking
	# into account number of items in bundle (k) and number of other items (n)
	per = perf/(k*n)
	return bunlen(k, per)

def gallenf(k, perf, n):
	# bundle length from formula in Gallant paper from final probability error
	# (perf) taking into account number of items in bundle (k) and number of other items (n)
	per = perf/(k*n)
	return gallen(k, per)

	# Calculate analytical accuracy of the encoding according to the equation for p_corr from 2017 IEEE Tran paper
def  p_corr (N, D, dp_hit):
	# dp_hit is the expected hamming distance (per dimension) from recalled vector to matching vector in item memory 
	# print("p_corr, N=%s, D=%s, dp_hit=%s" % (N, D, dp_hit))
	dp_rej=0.5
	var_hit = 0.25*N
	var_rej=var_hit
	range_var=10 # number of std to take into account
	fun = lambda u: (1/(np.sqrt(2*np.pi*var_hit)))*np.exp(-((u-N*(dp_rej-dp_hit) )**2)/(2*(var_hit)))*((norm.cdf((u/np.sqrt(var_rej))))**(D-1) ) # analytical equation to calculate the accuracy
	acc = integrate.quad(fun, (-range_var)*np.sqrt(var_hit), N*dp_rej+range_var*np.sqrt(var_hit)) # integrate over a range of possible values
	return acc[0]


# calculate vector lengths for cases given in Gallant paper, and also using
# equations for binary vectors from Pentti's 1997 paper, "Fully Distributed Representation" 
# S-number of items in bundle, N-number of other items, perf-desired percent error
cases = [{"label":"Small", "S":20, "N":1000, "perf":1.6},
	{"label":"Medium", "S":100, "N":100000, "perf":1.8},
	{"label":"Large", "S":1000, "N":1000000, "perf":1.0}]

for case in cases:
	k = case["S"]  # number items stored in bundle
	perf = case["perf"]/100.0   # desired final percent error after recalling all k items once
	d = case["N"]    # size of item memory
	dgal = gallenf(k, perf, d)
	dbun = bunlenf(k, perf, d)
	b_delta = 0.5 - 0.4 / math.sqrt(case["S"] - 0.44)
	acc = perf / k
	dfra = dplen2(b_delta, acc, d)
	ndenis = DimensionalityAnalytical(acc,k,d)
	perr_frady = (1-p_corr(dbun, case["N"], b_delta)) * 100.0  # get error probability from Frady equation
	print("%s (S=%s, N=%s, perf=%s), D is: Gallant: %s, bunlen: %s, fralen=%s, ndenis=%s, ratio: %s, Frady perror=%s" % (
		case["label"],case["S"],case["N"],case["perf"], dgal, dbun, dfra, ndenis, round(dbun / dgal,3), perr_frady))
