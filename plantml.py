import numpy as np


def plantml(ex: np.array, ey: np.array, s: float):
    """
    Computes the integral of the form-functions over a 3-node triangle element
        Me = int(s*N^T*N)dA

    Inputs:
        ex:     element x-coordinates
        ey:     element y-coordinates
        s:      constant scalar, e.g. density*thickness

    Outputs:
        Me:     integrated element matrix
    """
    if not ex.shape == (3,) or not ey.shape == (3,):
        raise Exception("Incorrect shape of ex or ey: {0}, {1} but should be (3,)".format(ex.shape, ey.shape))
    
    # Compute element area
    Cmat = np.vstack((np.ones((3, )), ex, ey))
    A = np.linalg.det(Cmat)/2

    # Set up quadrature
    g1 = [0.5, 0.0, 0.5]
    g2 = [0.5, 0.5, 0.0]
    g3 = [0.0, 0.5, 0.5]
    w = (1/3)

    # Perform numerical integration
    Me = np.zeros((3, 3))
    for i in range(0, 3):
        Me += w*np.array([
            [g1[i]**2,      g1[i]*g2[i],    g1[i]*g3[i]],
            [g2[i]*g1[i],   g2[i]**2,       g2[i]*g3[i]],
            [g3[i]*g1[i],   g3[i]*g2[i],    g3[i]**2]])

    Me *= A*s
    return Me