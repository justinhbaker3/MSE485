import numpy as np
import Functions as f
import Simulation as sim
import Analyzing_MD as an
import Statistics as stats
import random



def my_potential_energy_i(i, R, L):
    """
    Parameters:
      i : int, atom index
      R : Nx3 array of atom positions
      L : length of cubic periodic cell
    Returns:
      PE : float, potential energy sum over pairs ij (j!=i)
    """
    pe = 0
    for j in range(len(R)):
        if (j == i):
            continue
        disp = f.my_displacement(R[i], R[j], L)
        r = f.my_distance(disp)
        pe += 4 * (1 / r ** 12 - 1 / r ** 6)
    return pe

def my_update_position(i, R, L, beta, eta):
    """
    Propose a move to the position of particle i by updating its coordinates in R.
    Return the old position to reset in case the move is rejected.
    Compute the energy change and acceptance probability for the move.
    Parameters:
      i : int, particle index
      R : Nx3 array of particle positions
      L : float, length of cubic cell
      beta : float, inverse temperature
      eta : len-3 array for random gaussian proposed move
    Returns:
      oldPos : len-3 array, value of R[i] before move
      dE : float, energy change of proposed move
      acc_prob : float, probability of accepting move
    """
    R_old = R.copy()
    R[i] += eta
    dE = my_potential_energy_i(i, R, L) - my_potential_energy_i(i, R_old, L)
    acc_prob = np.exp(-beta*dE)
    return [R_old[i], dE, acc_prob]


def my_mc_sweep(R, L, beta, eta, acc_check):
    """
    Parameters:
      R : Nx3 array of atoms positions
      L : float, length of periodic cell
      beta : float, inverse temperature
      eta : Nx3 array of random gaussian proposed moves
      acc_check : len-N array of numbers in [0,1) for acceptance decision
    Returns:
      n_accepted : int, number of attempted moves that were accepted
      deltaE : float, total change in potential energy over the sweep
    """
    deltaE = 0
    n_accepted = 0
    for i in range(len(R)):
        update = my_update_position(i, R, L, beta, eta[i])
        if update[2] > acc_check[i]:
            n_accepted += 1
            deltaE += update[1]
        else:
            R[i] = update[0]
    return [n_accepted, deltaE]


def my_mc_sweep_force_bias(R, L, beta, eta, acc_check, sigma, M, dt):
    """
    Parameters:
      R : Nx3 array of atoms positions
      L : float, length of periodic cell
      beta : float, inverse temperature
      eta : Nx3 array of random gaussian proposed moves
      acc_check : len-N array of numbers in [0,1) for acceptance decision
      sigma : std dev used to propose Gaussian moves eta
      M : float, the mass of each atom
      dt : the time step for computing force step size
    Returns:
      n_accepted : int, number of attempted moves that were accepted
      deltaE : float, total change in potential energy over the sweep
    """

    deltaE = 0
    n_accepted = 0
    for i in range(len(R)):
        update = my_update_force_bias(i, R, L, beta, eta[i], sigma, M, dt)
        if update[2] > acc_check[i]:
            n_accepted += 1
            deltaE += update[1]
        else:
            R[i] = update[0]
    return [n_accepted, deltaE]

def my_update_force_bias(i, R, L, beta, eta, sigma, M, dt):
        """
        Propose a force biased move to the position of particle i by updating its coordinates in R.
        Return the old position to reset in case the move is rejected.
        Compute the energy change and acceptance probability for the move.
        Parameters:
          i : int, particle index
          R : Nx3 array of particle positions
          L : float, length of cubic cell
          beta : float, inverse temperature
          eta : len-3 array for random gaussian proposed move
          sigma : the standard deviation used to propose move eta
          M : float, mass of each atom
          dt : the time step for computing force step size
        Returns:
          oldPos : len-3 array, value of R[i] before move
          dE : float, energy change of proposed move
          acc_prob : float, probability of accepting move
        """
        oldPos = R[i].copy()
        oldE = my_potential_energy_i(i, R, L)
        force = f.my_InternalForce(i, R, L)
        x_adj = force / 2 / M * dt ** 2
        R[i] += eta + x_adj

        dE = my_potential_energy_i(i, R, L) - oldE
        force_new = f.my_InternalForce(i, R, L)
        x_adj_new = force_new / 2 / M * dt ** 2
        eta_d = f.my_distance(eta)
        new_d = f.my_distance(eta + x_adj + x_adj_new)
        tfor = np.exp(-(eta_d ** 2) / 2 / sigma ** 2)
        trev = np.exp(-(new_d ** 2) / 2 / sigma ** 2)
        acc_prob = np.exp(-beta * dE) * trev / tfor
        return [oldPos, dE, acc_prob]


def generate_eta(N, sigma):
    eta = sigma * np.random.randn(N, 3)
    return eta

def generate_acc_check(N):
    acc_check = np.random.rand(N)
    return acc_check


L = 4.0
rho = 1.0
N = 64
M = 48.0
T0 = 2
beta = 0.5
sigma = 0.06
dt = 0.032

def run_passes(N, L, beta, sigma, passes):
    R = sim.InitPosition(N, L)
    V = sim.InitVelocity(N, T0, M)
    pot_E = np.zeros(passes)
    R_all = np.zeros((passes, N, 3))
    V_all = np.zeros((passes, N, 3))
    acc_ratio = 0
    for i in range(passes):
        R_all[i] = R.copy()
        V_all[i] = V.copy()
        pot_E[i] = f.my_PotentialEnergy(R, L)
        eta = generate_eta(N, sigma)
        acc_check = generate_acc_check(N)
        acc_ratio += my_mc_sweep(R, L, beta, eta, acc_check)[0]
    print(acc_ratio/passes/N)
    return [R_all, V_all, pot_E]

def run_passes_force(N, L, beta, sigma, passes, M, dt):
    R = sim.InitPosition(N, L)
    V = sim.InitVelocity(N, T0, M)
    pot_E = np.zeros(passes)
    R_all = np.zeros((passes, N, 3))
    V_all = np.zeros((passes, N, 3))
    acc_ratio = 0
    for i in range(passes):
        R_all[i] = R.copy()
        V_all[i] = V.copy()
        pot_E[i] = f.my_PotentialEnergy(R, L)
        eta = generate_eta(N, sigma)
        acc_check = generate_acc_check(N)
        acc_ratio += my_mc_sweep_force_bias(R, L, beta, eta, acc_check, sigma, M, dt)[0]
    print(acc_ratio / passes / N)
    return [R_all, V_all, pot_E]


#R = sim.InitPosition(N, L)
#V = sim.InitVelocity(N, T0, L)
#update = my_mc_sweep(R, L, beta, generate_eta(N, sigma), generate_acc_check(N))
#print(float(update[0])/N)


data = run_passes(N, L, beta, sigma, 2000)
R_all = data[0]
V_all = data[1]
E = data[2]
np.savetxt("C:/Users/Justin Baker/Desktop/7potE.txt", E)
#pair_corr = an.calc_pair_corr(R_all, L, N, 40, 0.1, 0)
#np.savetxt("C:/Users/Justin Baker/Desktop/7pairCorr.txt", pair_corr)

#kvecs = an.my_legal_kvecs(5, L)
#sk = an.calc_Sk(R_all, 5, L, 250)
#kmags = [np.linalg.norm(kvec) for kvec in kvecs]
#unique_kmags = np.unique(kmags)
#unique_sk = np.zeros(len(unique_kmags))
#for iukmag in range(len(unique_kmags)):
#    kmag = unique_kmags[iukmag]
#    idx2avg = np.where(kmags==kmag)
#    unique_sk[iukmag] = np.mean(sk[idx2avg])

#np.savetxt("C:/Users/Justin Baker/Desktop/7sk.txt", unique_sk)
#np.savetxt("C:/Users/Justin Baker/Desktop/7kmags.txt", unique_kmags)
print(stats.my_stderr(E))