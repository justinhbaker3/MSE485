import json
import numpy as np
import Functions as fun

def open_data(filename):
    with open(filename,'r') as f:
        R = json.load(f)
    return np.array(R)

def load(file_name,timesize,atomsize):
	 ##timesize = 12 for this data set
	array = np.zeros((timesize,atomsize,3))
	fi = open(file_name,"r")
	for line in fi:
		line_str = line.split()
		if len(line_str) <=1 :
			continue
		if line_str[0] =="#":
			if line_str[1] =="time_index":
				tidx = int(line_str[-1])
			continue
		aidx = int(line_str[0])-1
		array[tidx][aidx][0] = float(line_str[1])
		array[tidx][aidx][1] = float(line_str[2])
		array[tidx][aidx][2] = float(line_str[3])
	fi.close()
	return array


def my_pair_correlation(distances, N, nbins, dr, L):
    """
    Parameters:
        distances : 1d array of pair distances
        N : number of atoms
        nbins : number of bins for the histogram
        dr : bin size for histogram
        L : length of cubic cell

    Returns:
        g : 1d array of length nbins (contains counts for each bin)
    """
    V = L ** 3
    rho = N / V
    g = np.zeros(nbins)
    for distance in distances:
        bin = int(np.floor(distance / dr))
        if bin < nbins:
            g[bin] += 1
        if (distance / dr) == nbins:
            g[nbins - 1] += 1
    for i in range(len(g)):
        v_sphere = 4 / 3 * np.pi * ((i + 1) * dr) ** 3
        v_inner = 4 / 3 * np.pi * (i * dr) ** 3
        n = (v_sphere - v_inner)
        n1 = N * (N - 1) / 2 / L ** 3
        g[i] = g[i] / n / n1
    return g


def get_distances(positions, L):
    num_distances = sum(range(1, len(positions)))
    distances = np.zeros(num_distances)
    idx = 0
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
           distances[idx] = fun.my_distance(fun.my_displacement(positions[i], positions[j], L))
           idx += 1
    return distances

def my_legal_kvecs(maxn, L):
    """
    Parameters:
      maxn : the maximum value for n_x, n_y, or n_z; maxn+1 is number of k-points along each axis
      L : length of (cubic) cell

    Returns:
      kvecs : mx3 array of k-vectors (m = (maxn+1)^3)
    """
    kvecs = np.zeros(((maxn + 1) ** 3, 3))
    idx = 0
    for i in range(maxn + 1):
        for j in range(maxn + 1):
            for k in range(maxn + 1):
                kvecs[idx][0] = i
                kvecs[idx][1] = j
                kvecs[idx][2] = k
                idx += 1
    kvecs = kvecs * 2 * np.pi / L

    return kvecs

def my_rhok(kvecs, rvecs):
    """
    Parameters:
        kvecs : mx3 array of k-vectors (or a single length-3 k-vector)
        rvecs : Nx3 array of position vectors

    Returns:
        rho : length-m vector (or scalar) of fourier transformed density
    """
    p = np.zeros(len(kvecs), dtype=np.complex_)
    for i in range(len(kvecs)):
        sum = 0 + 0J
        for k in range(len(rvecs)):
            sum += np.exp(-1J * np.dot(kvecs[i], rvecs[k]))
        p[i] = sum

    return p


def my_Sk(kvecs, rvecs):
    """
    Parameters:
        kvecs : mx3 array of k-vectors
        rvecs : Nx3 array of position vectors

    Returns:
        sk : length-m array, structure factor at each k-vector
    """
    return my_rhok(kvecs, rvecs) * my_rhok(kvecs * -1, rvecs) / len(rvecs)

def calc_Sk(R_all, maxn, L, start_time):
    steps = np.shape(R_all)[0]
    natoms = np.shape(R_all)[1]
    kvecs = my_legal_kvecs(maxn, L)
    sk = np.zeros(len(kvecs))
    for i in range(start_time, steps):
        sk += np.real(my_Sk(kvecs, R_all[i]))
    sk = sk/steps
    return sk

def myVelocityVelocityCorrelation(V, t):
    """
    Parameters:
      V: the array contianing particle velocities at different time.
         For example V[t][iat][i] gives the i compoent of the velcoity
         of iat particle at time t
      t: is the time index at which the velocity velocity correlation is
         calculated
    Returns:
      c: the vv correlation calculated at time index t.
    """
    sum = 0
    natoms = np.shape(V)[1]
    for i in range(natoms):
        sum += np.dot(V[0][i], V[t][i])

    return sum / natoms

def calc_vvc(V_all):
    steps = np.shape(V_all)[0]
    vvc = np.zeros(steps)
    for i in range(steps):
        vvc[i] = myVelocityVelocityCorrelation(V_all, i)
    return vvc

def myDiffusionConstant(vvCorrelation):
    """
    Parameters:
        vvCorrelation:  it is an array containing velocity-velocity Correlation at each time step.
    Returns:
        DiffusionConst: the diffusion constant calculated from the vv correlation.
    """
    h = 0.032  # NOTE Do not change this value. This is the time step for time integration.

    return np.sum(vvCorrelation) / 3 * h

def calc_momentum(V, M):
    steps = np.shape(V)[0]
    natoms = np.shape(V)[1]
    momentums = np.zeros(steps)
    for i in range(steps):
        for j in range(natoms):
            momentums[i] += (V[i][j][0]**2 + V[i][j][1]**2 + V[i][j][2]**2)**0.5
    momentums = momentums * M
    return momentums

def calc_temp(V, M, N):
    return fun.my_KineticEnergy(V, M)*2/3/N

def calc_temp_all(V_all, M, N):
    steps = np.shape(V_all)[0]
    temps = np.zeros(steps)
    for i in range(steps):
        temps[i] = calc_temp(V_all[i], M, N)
    return temps

#L = 4.2323167
#nbins = 21
#dr = 0.1

#data = load("C:/Users/Justin Baker/Downloads/velocities.txt", 12, 64)
def calc_pair_corr(R_all, L, N, nbins, dr, start_time):
    steps = np.shape(R_all)[0]
    natoms = np.shape(R_all)[1]
    corr = np.zeros(nbins)
    for i in range(start_time, steps):
        distances = get_distances(R_all[i], L)
        curr_corr = my_pair_correlation(distances, natoms, nbins, dr, L)
        corr += curr_corr
    corr = corr/steps
    return corr

"""
steps = np.shape(data)[0]
natoms = np.shape(data)[1]
correlation = np.zeros(nbins)

for i in range(steps):
    distances = get_distances(data[i], L)
    curr_corr = my_pair_correlation(distances, natoms, nbins, dr, L)
    correlation += curr_corr

correlation = correlation/steps
print(correlation)

distances = get_distances(positions, L)
"""

#print(myVelocityVelocityCorrelation(data, 10))