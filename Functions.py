import numpy


def my_ComputeEnergy(R, V, L, M):
    """
    Input:
    R: Positions of a list of atoms, size nx3, where n is the number of atoms
    V: Velocities of a list of atoms, size nx3, where n is the number of atoms
    L: Box length
    M: Mass

    Output:
    Float point number
    """
    ke = my_KineticEnergy(V, M)
    pe = my_PotentialEnergy(R, L)
    return ke + pe


def my_KineticEnergy(V, M):
    """
    Input:
    V: Velocities of a list of atoms, size nx3, where n is the number of atoms
    M: Mass

    Output:
    Float point number
    """
    ke = 0
    for i in range(len(V)):
        ke += M / 2 * (V[i][0] ** 2 + V[i][1] ** 2 + V[i][2] ** 2)
    return ke


def my_PotentialEnergy(R, L):
    """
    Input:
    R: Positions of a list of atoms, size nx3, where n is the number of atoms
    L: Box length

    Output:
    Float point number
    """
    pe = 0
    for i in range(len(R)):
        for j in range(i + 1, len(R)):
            disp = my_displacement(R[i], R[j], L)
            r = my_distance(disp)
            pe += 4 * (1 / r ** 12 - 1 / r ** 6)
    return pe


def my_InternalForce(i, R, L):
    """
    Input:
    i: index of the atom to calculate the total Lennard-Jones force on
    R: list of positions of all atoms, n-by-3 numpy array, n is the number of particle
    L: length of periodic box

    Output:
    Length-3 list or numpy array
    """
    sumForce = [0]*3

    for n in range(len(R)):
        if n == i:
            continue
        disp = my_displacement(R[i], R[n], L)
        r = my_distance(disp)
        force = (48 / (r ** 14) - 24 / (r ** 8))
        fvector = [k * force for k in disp]
        sumForce[0] += fvector[0]
        sumForce[1] += fvector[1]
        sumForce[2] += fvector[2]
    return numpy.asarray(sumForce)


def my_displacement(Ri, Rj, L):
    """
    Input:
    Two length-3 lists or numpy arrays Ri and Rj.
    Find the displacement (Ri-Rj) using minimum image convention.
    L is the box length.

    Output:
    A length-3 list or numpy array
    """

    dx = Ri[0] - Rj[0]
    while dx > L / 2:
        dx = dx - L
    while dx <= -L / 2:
        dx = dx + L

    dy = Ri[1] - Rj[1]
    while dy > L / 2:
        dy = dy - L
    while dy <= -L / 2:
        dy = dy + L

    dz = Ri[2] - Rj[2]
    while dz > L / 2:
        dz = dz - L
    while dz <= -L / 2:
        dz = dz + L

    return [dx, dy, dz]


def my_distance(dR):
    """
    Input:
    dR: length-3 displacement vector, already following minimum image convention.

    Output:
    Float point number, length of dR
    """
    return (dR[0] ** 2 + dR[1] ** 2 + dR[2] ** 2) ** 0.5

"""
i = 1
R = [[1, 2, 3], [2, 2, 3]]
L = 5
"""
def PutInBox(pos, L):
    pos[0] += L/2
    pos[1] += L/2
    pos[2] += L/2
    pos[0] = pos[0] - numpy.floor(pos[0]/L)*L
    pos[1] = pos[1] - numpy.floor(pos[1]/L)*L
    pos[2] = pos[2] - numpy.floor(pos[2]/L)*L
    pos[0] -= L/2
    pos[1] -= L/2
    pos[2] -= L/2

    return