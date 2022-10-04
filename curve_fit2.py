# script to find equation fitting relationship between
# number of rows and error rate for binarized sdm


# import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
import numpy as np
import sympy as sym

sdm_binarized_nact_optimum_dims = [
	[1, 50, 1, 200],
	[2, 97, 1, 200],
	[3, 158, 3, 300],
	[4, 208, 3, 400],
	[5, 262, 3, 3000],
	[6, 315, 5, 10000],
	[7, 368, 5, 40000]]

def bine(x):
	m = 0.04344500175460464
	k=0.8778013135972488
	return k * np.exp(-m*x)


"""
m = 0.04344500175460464
>>> math.log(0.1) + m*50
-0.13033500526381347
>>> math.exp(-0.13033500526381347)
0.8778013135972488
>>> k=0.8778013135972488
>>> def bine(nrows):
...     return k * math.exp(-m*nrows)
"""

using_binarized_sdm = True

if using_binarized_sdm:
	sdm_dims = sdm_binarized_nact_optimum_dims # sdm_binarized_nact_1_dims

x = np.empty(len(sdm_dims), dtype=np.float64)
y = np.empty(len(sdm_dims), dtype=np.float64)
for i in range(len(sdm_dims)):
	perr, nrows, nact, epochs = sdm_dims[i]
	x[i] = nrows
	y[i] = 10**(-perr)


ylog = np.log(y)

"""
Plot your data
"""
plt.plot(x, y, 'ro',label="Original Data")

yb = [bine(nrows) for nrows in x]
plt.plot(x, y, 'g-', label="bine calculated")

# function to optomize
def func(x, k, m):
	return k + np.exp(-m * x)

popt, pcov = curve_fit(func, x, y, p0=[0.8778013135972488, 0.04344500175460464])


# result = linregress(x, y=ylog, alternative='less')
result = linregress(x, y=ylog)

print("linregress slope=%s, intercept=%s, exp intercept=%s" % (result.slope, result.intercept, np.exp(result.intercept)))

k = np.exp(result.intercept)
m = -result.slope
yreg = k * np.exp(-m *x)

plt.plot(x, yreg, 'yo', label="linregress")

print("popt=%s" % popt)

# from: https://stackoverflow.com/questions/3433486/how-to-do-exponential-and-logarithmic-curve-fitting-in-python-i-found-only-poly
"""
The result is:
popt[0] = a , popt[1] = b, popt[2] = c and popt[3] = d of the function,
so f(x) = popt[0]*x**3 + popt[1]*x**2 + popt[2]*x + popt[3].
"""
print ("k = %s , m = %s" % (popt[0], popt[1]))

"""
Use sympy to generate the latex sintax of the function
"""
# xs = sym.Symbol('\lambda')    
# tex = sym.latex(func(xs,*popt)).replace('$', '')
# plt.title(r'$f(\lambda)= %s$' %(tex),fontsize=16)

plt.title("found vs curve fit k + np.exp(-m * x)")
"""
Print the coefficients and plot the funcion.
"""

plt.yscale('log')

plt.plot(x, func(x, *popt), label="Fitted Curve") #same as line above \/
#plt.plot(x, popt[0]*x**3 + popt[1]*x**2 + popt[2]*x + popt[3], label="Fitted Curve") 

plt.legend(loc='upper left')
plt.show()

