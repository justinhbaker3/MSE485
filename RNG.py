import numpy as np
import math
import matplotlib.pyplot as plt
import random
def my_LCG(m,a,c,x0,N):
    """
    Input:
    m, a, c are parameters of the LCG.
    x0: the initial pseduo-random number.
    N : number of random numbers to return

    Output:
    a list or numpy array of length N, containing the next N pseduo-random numbers in order (excluding x0).
    """
    ret = np.zeros(N)
    ret[0] = (a*x0+c) % m
    for i in range(1, N):
        ret[i] = (a*ret[i-1]+c) % m
    return ret

def my_GaussianRNG(m,a,c,x0,N):
    """
    Input:
    m, a, c, x0 are parameters of the LCG.
    N : number of Gaussian random numbers to return

    Output:
    a list or numpy array of length N, containing the next N Gaussian pseduo-random numbers in order (excluding x0).
    """
    rands = my_LCG(m, a, c, x0, 2*N)
    rands = rands/float(m)
    gauss_rands = np.zeros(N)
    for i in range(N):
        gauss_rands[i] = (-2*math.log(rands[2*i]))**0.5*math.sin(2*math.pi*rands[2*i+1])
    return gauss_rands

def my_CheckRandomNumbers1D(rand_array,NB):
    """
    This function can be reused later for larger data set

    Input:
    rand_array: array of length N
    NB: number of bins per dimension (for 1D we need NB bins in total)

    Output:
    the chi-squared value of the rand_array, with NB evenly spaced bins in [0,1).
    """
    # complete this function
    bins = np.zeros(NB)
    width = 1/NB
    for i in range(len(rand_array)):
        bins[int(np.floor(rand_array[i]/width))] += 1
    chi2 = float(0)
    expected = len(rand_array)/NB
    for i in range(len(bins)):
        chi2 += (bins[i]-expected)**2/expected
    return chi2


def my_CheckRandomNumbers2D(rand_array, NB):
    """
    This function can be reused later for larger data set

    Input:
    rand_array: array of size N-by-2, so (rand_array[0][0], rand_array[0][1]) is the first 2D point
    NB: number of bins per dimension (for 2D we need NB*NB bins in total)

    Output:
    the chi-squared value of the rand_array, with NB*NB evenly spaced bins in [0,1)x[0,1).
    """
    bins = np.zeros((NB, NB))
    width = 1/NB
    length = np.shape(rand_array)[0]
    for i in range(length):
        idx_x = np.floor(rand_array[i][0]/width)
        idx_y = np.floor(rand_array[i][1]/width)
        bins[int(idx_x)][int(idx_y)] += 1
    chi2 = float(0)
    expected = length/NB/NB
    for i in range(len(bins)):
        for j in range(len(bins)):
            chi2 += (bins[i][j]-expected)**2/expected
    return chi2


def my_CheckRandomNumbers3D(rand_array,NB):
    """
    This function can be reused later for larger data set

    Input:
    rand_array: array of size N-by-3, so (rand_array[0][0], rand_array[0][1], rand_array[0][2]) is the first 3D point
    NB: number of bins per dimension (for 3D we need NB*NB*NB bins in total)

    Output:
    the chi-squared value of the rand_array, with NB*NB*NB evenly spaced bins in [0,1)x[0,1)x[0,1).
    """
    # complete this function
    bins = np.zeros((NB, NB, NB))
    width = 1/NB
    length = np.shape(rand_array)[0]
    for i in range(length):
        idx_x = np.floor(rand_array[i][0]/width)
        idx_y = np.floor(rand_array[i][1]/width)
        idx_z = np.floor(rand_array[i][2]/width)
        bins[int(idx_x)][int(idx_y)][int(idx_z)] += 1
    chi2 = float(0)
    expected = length/NB/NB/NB
    for i in range(len(bins)):
        for j in range(len(bins)):
            for k in range(len(bins)):
                chi2 += (bins[i][j][k] - expected)**2/expected
    return chi2


def my_ComputeIntegral(func, L, N):
    """
    Input:
    func: a well defined function that decays fast when |x| goes to infinity
    L: we will approximate [-infinity, infinity] with [-L, L]
    N: we will discretize the [-L,L] into N equally long segments

    Output:
    the integral using rectangle rule
    """
    ret = float(0)
    width = 2*L/N
    endpoints = np.zeros(N+1)
    for i in range(N+1):
        endpoints[i] = (width*i)-L
    for i in range(len(endpoints)-1):
        ret += func((endpoints[i] + endpoints[i+1])/2)
    ret = ret * width
    return ret

def histogram_values(start, stop, nbins):
    width = (stop-start)/nbins
    histogram = np.zeros(nbins)
    for i in range(nbins):
        histogram[i] = (width*i)+start+width/2
    return histogram


m = 4294967296
a = 69069
c = 1
x0 = 0


#rands = np.loadtxt("C:/Users/Justin Baker/Downloads/3D_chi_squared_data_set_3.txt")
#print(my_CheckRandomNumbers3D(rands, 10))

#print(my_LCG(16,3,1,2, 20))
#gauss_values = my_GaussianRNG(m, a, c, x0, 10000)
#histogram = np.histogram(gauss_values, 50, (-5,5))
#x = histogram_values(-5, 5, 50)
#plt.plot(x, histogram[0])
#plt.show(block = True)
#histogram = np.asarray(histogram[0])
#histogram = histogram*50/2/5/10000
#np.savetxt("C:/Users/Justin Baker/Desktop/Q2x.txt", x)
#np.savetxt("C:/Users/Justin Baker/Desktop/Q2histogram.txt", histogram)

def twoD_randoms(m, a, c, x0, N):
    rands = np.zeros((N, 2))
    lin_rands = my_LCG(m, a, c, x0, 2*N)
    for i in range(len(rands)):
        rands[i][0] = lin_rands[2*i]
        rands[i][1] = lin_rands[2*i+1]
    return rands

def threeD_randoms(m, a, c, x0, N):
    rands = np.zeros((N, 3))
    lin_rands = my_LCG(m, a, c, x0, 3*N)
    for i in range(len(rands)):
        rands[i][0] = lin_rands[3*i]
        rands[i][1] = lin_rands[3*i+1]
        rands[i][2] = lin_rands[3*i+2]
    return rands

#values = my_LCG(m, a, c, x0, 10000)
#values = values/m
#values = threeD_randoms(m, a, c, x0, 10000)
#values = values/m
#print(values)
"""
values = np.zeros((10000, 3))
for i in range(len(values)):
    values[i][0] = random.random()
    values[i][1] = random.random()
    values[i][2] = random.random()
print(my_CheckRandomNumbers3D(values, 10))


rands = np.loadtxt("C:/Users/Justin Baker/Downloads/2D_chi_squared_data_set_3.txt")
rands = rands[:1000]
bins = np.zeros(100)
for i in range(len(rands)):
    idx_x = np.floor(rands[i][0]/.1)
    idx_y = np.floor(rands[i][1]/.1)
    idx = idx_x+10*idx_y
    bins[int(idx)] += 1
np.savetxt("C:/Users/Justin Baker/Desktop/Q3set3.txt", bins)
"""

def func(x):
    return math.exp(-(x**2)/2)/(1+x**2)


def my_ComputeIntegral2(func, alpha, N):
    """
    Input:
    func: a well defined function that decays fast when |x| goes to infinity
    alpha: variance of the normal distribution to be sampled
    N: length of Gaussian random numbers

    Output:
    a two-element list or numpy array, with the first element being the estimate of the integral,
    and the second being the the estimate of the variance
    """
    # -------------------------------------------------------------------
    # We will be using m=2^32, a=69069, c=1, and x0=0 for my_GaussianRNG
    # Multiply the stream of Gaussian random numbers by np.sqrt(alpha) to make their variance equal to alpha
    gausian_arrays = np.sqrt(alpha) * np.array(my_GaussianRNG(2 ** 32, 69069, 1, 0, N))

    # -------------------------------------------------------------------
    # define p(x, alpha)
    def p(x, alpha):
        return np.exp(-x * x / (2.0 * alpha)) / np.sqrt(2.0 * np.pi * alpha)

    # -------------------------------------------------------------------
    # complete this function
    values = np.zeros(N)
    for i in range(N):
        values[i] = func(gausian_arrays[i]) / p(gausian_arrays[i], alpha)
    return [np.average(values), np.std(values) ** 2]


def g_trace(func, alpha, N):
    gausian_arrays = np.sqrt(alpha) * np.array(my_GaussianRNG(2 ** 32, 69069, 1, 0, N))

    # -------------------------------------------------------------------
    # define p(x, alpha)
    def p(x, alpha):
        return np.exp(-x * x / (2.0 * alpha)) / np.sqrt(2.0 * np.pi * alpha)

    # -------------------------------------------------------------------
    # complete this function
    values = np.zeros(N)
    trace = np.zeros(N)
    for i in range(N):
        values[i] = func(gausian_arrays[i]) / p(gausian_arrays[i], alpha)
        trace[i] = np.average(values[:i+1])
    return trace

"""
alphas = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2]
variances = np.zeros(len(alphas))
for i in range(len(alphas)):
    ans = my_ComputeIntegral2(func, alphas[i], 40000)
    variances[i] = ans[1]
    print(ans[0])
"""
trace = g_trace(func, 0.1, 10000)
np.savetxt("C:/Users/Justin Baker/Desktop/Q12_0.8.txt", trace)