import numpy as np
import Functions as func

def my_timestep(R, V, L, n):
    for i in range(n):
        for i in range(len(V)):
            force = func.my_InternalForce(i, R, L)
            acc = force/M
            V[i] = V[i] + acc*h

        for i in range(len(R)):
            R[i] = R[i] + V[i]*h

        print(R)
        print(V)
    return R

initial_R_txt = open("C:/Users/Justin Baker/Downloads/data2.10_initial_R.txt", "r")
initial_V_txt = open("C:/Users/Justin Baker/Downloads/data2.10_initial_V.txt", "r")

initial_R_list = []
initial_V_list = []
for line in initial_R_txt.readlines():
    xyz = line.split(" ")
    initial_R_list.append([xyz[0], xyz[1], xyz[2]])

for line in initial_V_txt.readlines():
    xyz = line.split(" ")
    initial_V_list.append([xyz[0], xyz[1], xyz[2]])

initial_R = np.asarray(initial_R_list, float)
initial_V = np.asarray(initial_V_list, float)

L = 4.2323167
M = 48
h = 0.01
N = 64

R = initial_R
V = initial_V

my_timestep(R, V, L, 1000)

ke = func.my_KineticEnergy(V, M)
pe = func.my_PotentialEnergy(R, L)
E = func.my_ComputeEnergy(R, V, L, M)
print(ke)
print(pe)
print(E)