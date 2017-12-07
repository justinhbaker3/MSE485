import numpy as np
import matplotlib.pyplot as plt
import sets
from multiprocessing import Queue
import time

def my_SpinFlip2(lattice, x, y, bJ):
    """
    Input:
    lattice: numpy array of size N-by-N, where N is the length of the side.
             Each entry of lattice is either +1 (spin up) or -1 (spin down).
             Assuming periodic boundary conditions.
    x,y: integers, coordinates of the site at which a spin flip is proposed
    bJ: float, inverse temperature times Ising coupling strength

    Output:
    probability of the spin being flipped, and the flip's change to total magnetization.
    """
    dM = 0
    if lattice[x][y] == -1:
        dM = 2.0
    else:
        dM = -2.0
    lattice[x][y] *= -1
    neighbors = my_NeighborList(len(lattice), x, y)
    Eup = 0
    Edn = 0
    for nx, ny in neighbors:
        Eup += 1*lattice[nx][ny]
        Edn += -1*lattice[nx][ny]
    p_up = np.exp(-Eup*bJ)
    p_dn = np.exp(-Edn*bJ)
    acc_prob = 0
    if dM == -2:
        acc_prob = p_up/(p_dn+p_up)
    else:
        acc_prob = p_dn/(p_up+p_dn)
    return (acc_prob, dM)


def my_SpinFlip1(lattice, x, y, bJ):
    """
    Input:
    lattice: numpy array of size N-by-N, where N is the length of the side.
             Each entry of lattice is either +1 (spin up) or -1 (spin down).
             Assuming periodic boundary conditions.
    x,y: integers, coordinates of the site at which a spin flip is proposed
    bJ: float, inverse temperature times Ising coupling strength

    Output:
    probability of accepting the move, and the move's change to total magnetization.
    """
    dM = float(0)
    if lattice[x][y] == -1:
        lattice[x][y] = 1
        dM = 2.0
    else:
        lattice[x][y] = -1
        dM = -2.0
    neighbors = my_NeighborList(len(lattice), x, y)
    dE = 0
    for xn, yn in neighbors:
        dE += 2 * lattice[x][y] * lattice[xn][yn]
    acc_prob = np.exp(dE * bJ)
    if acc_prob > 1:
        acc_prob = 1
    return (acc_prob, dM)


def my_ComputeTotalMagnetization(lattice):
    """
    Input:
    lattice: numpy array of size N-by-N, where N is the length of the side.
             Each entry of lattice is either +1 (spin up) or -1 (spin down).

    Output:
    Total Magnetization.
    """
    mtotal = 0
    for i in range(len(lattice)):
        for j in range(len(lattice)):
            mtotal += lattice[i][j]
    return mtotal


def my_NeighborList(N, x, y):
    """
    Input:
    N: side legnth of the square lattice
    x,y: coordinates of a site

    Output:
    [[x_left,y_left], [x_top,y_top], [x_right,y_right], [x_bottom, y_bottom]]
    """
    x_left = x - 1
    if x_left < 0:
        x_left = N - 1
    y_top = y - 1
    if y_top < 0:
        y_top = N - 1
    x_right = x + 1
    if x_right >= N:
        x_right = 0
    y_bottom = y + 1
    if y_bottom >= N:
        y_bottom = 0
    return [[x_left, y], [x, y_bottom], [x_right, y], [x, y_top]]

def cost(a, t, mean, stdev):
    sum = 0
    for i in range(0, len(a)-t-1):
        sum += (a[i] - mean)*(a[t+i] - mean)
    return sum/stdev/stdev/(len(a)-t)

def my_actime(a):
    mean = np.mean(a)
    stdev = np.std(a)
    sum = 0
    t = 0
    c = 0
    while True:
        t += 1
        c = cost(a, t, mean, stdev)
        if c <= 0:
            break
        sum += c
    return 1 + 2*sum


def Met_sim(sweeps, bJ):
    spins = np.ones((sps,sps))
    M = my_ComputeTotalMagnetization(spins)
    M_list = np.zeros(sweeps)
    for sweep in range(sweeps):
        for i in range(sps*sps):
            x = np.random.randint(0, sps)
            y = np.random.randint(0, sps)
            acc_prob = my_SpinFlip1(spins, x, y, bJ)
            if acc_prob[0] > np.random.random():
                M += acc_prob[1]
            else:
                spins[x][y] *= -1
            M_list[sweep] = M
    return M_list

def heat_bath_sim(sweeps, bJ):
    spins = np.ones((sps,sps))
    M = my_ComputeTotalMagnetization(spins)
    M_list = np.zeros(sweeps)
    for sweep in range(sweeps):
        for i in range(sps*sps):
            x = np.random.randint(0, sps)
            y = np.random.randint(0, sps)
            acc_prob = my_SpinFlip2(spins, x, y, bJ)
            if acc_prob[0] > np.random.random():
                M += acc_prob[1]
            else:
                spins[x][y] *= -1
            M_list[sweep] = M
    return M_list

def cluster_step(bJ, spins):
    cluster = set()
    q = Queue()
    x_init = np.random.randint(0, sps)
    y_init = np.random.randint(0, sps)
    initial_spin = spins[x_init][y_init]
    cluster.add((x_init,y_init))
    q.put((x_init,y_init))
    while not q.empty():
        site = q.get()
        neighbors = my_NeighborList(sps, site[0], site[1])
        for x, y in neighbors:
            if (x,y) not in cluster:
                if spins[x][y] == spins[site[0]][site[1]]:
                    if (1-np.exp(-2*bJ)) > np.random.random():
                        cluster.add((x, y))
                        q.put((x,y))
    for x, y in cluster:
        spins[x][y] *= -1
    #print(len(cluster))
    time.sleep(0.005)
    return len(cluster)*-2*initial_spin

def cluster_sim(sweeps, bJ):
    spins = np.ones((sps, sps))
    M = my_ComputeTotalMagnetization(spins)
    M_list = np.zeros(sweeps)
    for sweep in range(sweeps):
        M += cluster_step(bJ, spins)
        M_list[sweep] = M
    return M_list

def get_stats(M_list):
    M2_list = (M_list/sps**2)**2
    act = my_actime(M_list)
    return [M2_list, act]

sps = 20
J = 1.0
bJ = 0.3

M_list = cluster_sim(1000, bJ)
#print(M_list)
stats = get_stats(M_list)
print(np.mean(stats[0]))
print(stats[1])
print(np.std(stats[0]))
#np.savetxt("C:/Users/Justin Baker/Desktop/4_0.44096.txt", stats[0])
plt.plot(stats[0])
plt.show(block=True)
