import numpy as np

p_choose_particle = 0.8

def MA_step(V, T, M):
    sigma = (T/M)**0.5
    count = 0

    for i in range(0, len(V)):
        if p_choose_particle > np.random.ranf():    #choose particles with probability p_choose_particle
            proposed_velocity = sigma*np.random.randn(3)    #choose propoesed velocity on the gaussian distribution

            acc_prob = 1 - np.absolute(proposed_velocity-V[i])/6/sigma        #accecptance probability inverse of percent change
            acc_check = np.random.ranf(3)

            if acc_prob[0] > acc_check[0]:
                count += 1
                V[i][0] = proposed_velocity[0]          #update V[i]x
            if acc_prob[1] > acc_check[1]:
                count += 1
                V[i][1] = proposed_velocity[1]          #update V[i]y
            if acc_prob[2] > acc_check[2]:
                count += 1
                V[i][2] = proposed_velocity[2]          #update V[i]z

    return count

