import numpy, math, random
import Functions as func
import Analyzing_MD as analysis

def InitPosition(N, L):
    R_initial = numpy.zeros((N, 3)) + 0.0
    Ncube = 1
    while(N > Ncube*Ncube*Ncube):
        Ncube += 1
    rs = float(L)/Ncube
    roffset = float(L)/Ncube
    added = 0
    for x in range(0, Ncube):
        for y in range(0, Ncube):
            for z in range(0, Ncube):
                if added < N:
                    R_initial[added, 0] = rs*x - roffset
                    R_initial[added, 1] = rs*y - roffset
                    R_initial[added, 2] = rs*z - roffset
                    added += 1
    return R_initial

def InitVelocity(N, T0, mass=1.):
    dimensions = 3
    V_initial = numpy.zeros((N, 3)) + 0.0
    random.seed(1)
    netP = numpy.zeros((3 ,)) + 0.
    netE = 0.0
    for n in range(0, N):
        for x in range(0, dimensions):
            newP = random.random()-0.5
            netP[x] += newP
            netE += newP*newP
            V_initial[n, x] = newP
    netP *= 1.0/N
    vscale = math.sqrt(3*N*T0/(mass*netE))
    for n in range(0, N):
        for x in range(0, dimensions):
            V_initial[n, x] = (V_initial[n, x] - netP[x])*vscale
    return V_initial

def VerletNextR(R, V, A, h):
    newR = R + V*h + 0.5*A*h*h
    return newR
def VerletNextV(V, A, Anext, h):
    newV = V + 0.5*(A + Anext)*h
    return newV

def TaylorNextR(R, V, A, h):
    newR = R + V*h + 0.5*A*h*h
    return newR
def TaylorNextV(V, A, h):
    newV = V + 0.5*A*h
    return newV


def simulate(steps, h, thermostat):
    R = InitPosition(N, L)
    V = InitVelocity(N, T0, M)
    A = numpy.zeros((N, 3))

    nR = numpy.zeros((N,3))
    nV = numpy.zeros((N,3))

    E = numpy.zeros(steps)
    R_all = numpy.zeros((steps, N, 3))
    V_all = numpy.zeros((steps, N, 3))

    for t in range(0, steps):

        for i in range(0, len(R)):
            F = func.my_InternalForce(i, R, L)
            A[i] = F/M
            nR[i] = VerletNextR(R[i], V[i], A[i], h)
            func.PutInBox(nR[i], L)
        for i in range(0, len(R)):
            nF = func.my_InternalForce(i, nR, L)
            nA = nF/M
            nV[i] = VerletNextV(V[i], A[i], nA, h)

        if thermostat:
            prob = 0.01
            sigma = (T/M)**0.5
            for i in range(len(R)):
                if (numpy.random.ranf() < prob):
                    #print("collision")
                    nV[i] = sigma * numpy.random.randn(3)

        R = nR.copy()
        V = nV.copy()
        V_all[t] = V.copy()
        R_all[t] = R.copy()
        E[t] = func.my_ComputeEnergy(R, V, L, M)


    return [E, R_all, V_all]
"""
output = simulate(800, h, 1)
E = output[0]
V_all = output[2]
R_all = output[1]
#momentum = analysis.calc_momentum(V_all, M)
#vvc = analysis.calc_vvc(V_all)
#print(analysis.myDiffusionConstant(vvc))
kvecs = analysis.my_legal_kvecs(5, L)
sk = analysis.calc_Sk(R_all, 5, L, 250)
kmags = [numpy.linalg.norm(kvec) for kvec in kvecs]
unique_kmags = numpy.unique(kmags)
unique_sk = numpy.zeros(len(unique_kmags))
for iukmag in range(len(unique_kmags)):
    kmag = unique_kmags[iukmag]
    idx2avg = numpy.where(kmags==kmag)
    unique_sk[iukmag] = numpy.mean(sk[idx2avg])

pair_corr = analysis.calc_pair_corr(R_all, L, N, 100, L/100, 250)
#temps = analysis.calc_temp_all(V_all, M, N)
numpy.savetxt("C:/Users/Justin Baker/Desktop/7kmags0.5.txt", unique_kmags)
numpy.savetxt("C:/Users/Justin Baker/Desktop/7sk0.5.txt", unique_sk)
numpy.savetxt("C:/Users/Justin Baker/Desktop/7pc0.5.txt", pair_corr)
"""