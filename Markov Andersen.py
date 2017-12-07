import Simulation as sim
import numpy as np

p_choose_particle = 0.01

def MA_step(R, V, T, M):
    proposed_velocity = np.random.