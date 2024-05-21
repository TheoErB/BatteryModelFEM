import matplotlib.pyplot as plt
import calfem.vis_mpl as cfv
import numpy as np
import calfem.core as cfc

import create_mesh as mesh

# Constant material properties 
Emod        = 5 * 10**9         # Young's modulus
ny          = 0.36              # Poisson's ration
alpha_exp   = 60 * 10**(-6)     # Thermal expansion coefficient
density     = 540               # Density
cp          = 3600              # Specific heat capacity
k_tc        = 80                # Thermal conductivity
t           = 1600 * 10**(-3)   # Battery thickness

alpha_n     = 40                # Convection coefficient surrounding
T_inf       = 293               # Surrounding temperature
alpha_c     = 120               # Convection coefficient for cooling tubes
T_in        = 277               # Temperature, cold cooling tubes
T_out       = 285               # Temperature, hot cooling tubes
T_zero      = 293               # Initial temperature

# Homogenous diffusion matrix
D = np.array([[k_tc, 0], [0, k_tc]]) 


# Find stationary temperature distribution
def find_stationary_temp_distribution():
    coord, edof, dofs, bdofs, element_markers, boundaryElements = mesh.generate_mesh()

    nen = np.shape(edof)[1]                     # Number of nodes per element
    nDofs = np.size(dofs)                       # Number of frihetsgrader
    et = np.array([t])                          # Element thickness

    # Get node coordinates for each element
    Ex, Ey = cfc.coordxtr(edof, coord, dofs)
    nel = len(Ex)                               # Number of elements

    # Empty matrices for stiffness matrix and force vector
    K = np.zeros((nDofs, nDofs))
    F = np.zeros([nDofs, 1])
    fl = np.zeros([nDofs, 1])

    # Getting element stiffness matrix and assembling

    for i in range(nel):
        ex = np.array(Ex[i, :])
        ey = np.array(Ey[i, :])


        Ke = cfc.flw2te(ex, ey, et, D)
        cfc.assem(edof[i,:], K, Ke)
        
        

    # Generating K_c and F_c matrices

    f_c = np.zeros([nDofs, 1])
    K_c = np.zeros([nDofs, nDofs])

    # Lists holding constant values used to generate f_c and K_c
    T_list =        [alpha_c * T_out , alpha_c * T_in , 0, alpha_n * T_inf ]
    alpha_list =    [alpha_c         , alpha_c        , 0, alpha_n         ]



    for j in range(4):
        for elements in boundaryElements[j]:
            el1 = elements['node-number-list'][0]-1
            el2 = elements['node-number-list'][1]-1
            
            if j != 2:
                # Calculations length between two boundary nodes
                L = np.sqrt( (coord[el1][0] - coord[el2][0])**2 + (coord[el1][1] - coord[el2][1])**2 )

                # Assembles f_c for each boundary line
                f_c[el1] = f_c[el1] + T_list[j] * t * L/2
                f_c[el2] = f_c[el2] + T_list[j] * t * L/2

                # Assembles K_c for each boundary line
                K_c[el1][el1] = K_c[el1][el1] + alpha_list[j] * t * L/3
                K_c[el1][el2] = K_c[el1][el2] + alpha_list[j] * t * L/6
                K_c[el2][el1] = K_c[el2][el1] + alpha_list[j] * t * L/6
                K_c[el2][el2] = K_c[el2][el2] + alpha_list[j] * t * L/3


    # Creating the full F and K vector used for the solution
    
    F = fl + f_c 
    K = K + K_c

    # Initiation boundry conditions
    bc, bc_value = np.array([], 'i'), np.array([], 'f')

    # Solving the stationary temperature distribution
    a, r = cfc.solveq(K, F, bc, bc_value)

    return a, coord, edof

if __name__ == "__main__":
        
    a, coord, edof = find_stationary_temp_distribution()
    ed = cfc.extract_ed(edof,a)

    # Plots the stationary temperature distribution within the battery
    cfv.figure(fig_size=(5,5))
    cfv.draw_nodal_values_shaded(a, coord, edof, dofs_per_node=1, el_type=2, title="Stationary temperature distribution")
    cfv.colorbar()
    plt.show()