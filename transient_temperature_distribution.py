import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.utils as cfu
import matplotlib as mpl
import matplotlib.pyplot as plt
import calfem.vis_mpl as cfv
import numpy as np
import calfem.core as cfc
import math as m
mpl.use('TkAgg')

import create_mesh as mesh
import plantml

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
D = np.array([[k_tc, 0], 
              [0, k_tc]]) 

# Time-step properties
nbr_of_steps = 120
nbr_of_plotpoints = 6

# Battery-charging functions

def f1(t):
    return 100* np.exp(-144*((600-t)/3600)**2) * 1000

def f2(t):
    return 88.42 * 1000 if t < 600 else 0


# Finding transient temperature distribution
def find_transient_temp_distribution( main = False ):

    # Generate mesh
    coord, edof, dofs, bdofs, element_markers, boundaryElements = mesh.generate_mesh()

    # Mesh parameters
    nen = np.shape(edof)[1]                     # Number of nodes per element
    nDofs = np.size(dofs)                       # Number of frihetsgrader
    et = np.array([t])                          # Element thickness

    # Get node coordinates for each element
    Ex, Ey = cfc.coordxtr(edof, coord, dofs)
    nel = np.shape(edof)[0]                     # Number of elements

    # Parameters for the timestep-method
    totalTime = 3600
    steps = nbr_of_steps
    delta_t = totalTime/steps


    # ----- ASSEMNBLING ----- #

    # Empty matrices for stiffness matrix and C-matrix
    K = np.mat(np.zeros((nDofs, nDofs)))
    C = np.mat(np.zeros((nDofs, nDofs)))

    # Empty vectors for force vectors
    F = np.zeros([nDofs, 1])
    fl = np.zeros([nDofs, 1])

    # Getting element stiffness matrix, f-load vector and C-matrix and assembling
    for i in range(nel):
        ex = np.array(Ex[i, :])
        ey = np.array(Ey[i, :])

        Ke, fle = cfc.flw2te(ex, ey, et, D, 1)
        cfc.assem(edof[i,:], K, Ke, fl, fle)

        Ce = plantml.plantml(ex, ey, density*t*cp)
        cfc.assem(edof[i,:], C, Ce)
        
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
    
    F = F + f_c 
    K = K + K_c

    # ----- TIME-STEPPING ----- #

    # Vectors to store all timesteps and their respective temperature distribution
    an_1 = np.ones([nDofs, 1])*T_zero           # Initial temperature during Q_1 charging 
    an_2 = np.ones([nDofs, 1])*T_zero           # Initial temperature during Q_2 charging 
    aTimef1 = np.ones([nDofs, 1])*T_zero        # All temperatures during Q_1 charging   
    aTimef2 = np.ones([nDofs, 1])*T_zero        # All temperatures during Q_2 charging
    time = [0]                                  # Time vector
    tn = 0                                      # Time passed at start

    # Vectors to store maxlimal and minimal temperatures within the battery for each timestep
    maxTempf1= [T_zero]
    minTempf1= [T_zero]

    maxTempf2= [T_zero]
    minTempf2= [T_zero]

    # Vectors to store the maximun deviation from the original temperature for each timestep
    maxDevf1 = [0]
    maxDevf2 = [0]

    CKinv = np.linalg.inv( C + delta_t*K )

    # Variables used to find distribution used in c)
    max_q1 = 0
    max_q2 = 0
    max_a_1 = 0
    max_a_2 = 0

    for i in range(1,steps+1):
        print(i)

        # Gets the new temperature after one time step
        tn = tn + delta_t
        time = np.append(time, tn)

        an_1 = CKinv * ( (f1(tn)*fl + F) * delta_t + (C * an_1))
        an_2 = CKinv * ( (f2(tn)*fl + F) * delta_t + (C * an_2))
        
        # Get max and min temp, as well as the current maximun deviation for Q1
        max_n = np.max(an_1)
        maxTempf1.append( max_n)
        min_n = np.min(an_1)
        minTempf1.append( min_n)

        maxDev_n = np.maximum( np.abs(max_n - T_zero), np.abs(T_zero - min_n) )
        maxDevf1.append( maxDev_n)

        if max_n > max_q1:
            max_q1 = max_n
            max_a_1 = an_1

        # Get max and min temp, as well as the current maximun deviation for Q2
        max_n = np.max(an_2)
        maxTempf2.append( max_n)
        min_n = np.min(an_2)
        minTempf2.append( min_n)

        maxDev_n = np.maximum( np.abs(max_n - T_zero), np.abs(T_zero - min_n) )
        maxDevf2.append( maxDev_n)

        # Stores temperature distribution when deviation is maximal
        if max_n > max_q2:
            max_q2 = max_n
            max_a_2 = an_2

        # Stores the current temperature distribution
        aTimef1 = np.hstack((aTimef1, an_1))
        aTimef2 = np.hstack((aTimef2, an_2))

    if main:    
        return aTimef1, aTimef2, maxTempf1, maxTempf2, minTempf1, minTempf2, maxDevf1, maxDevf2, time, coord, edof
    else: 
        return max_a_1, max_a_2, coord, dofs, edof, bdofs, nel, Ex, Ey
    


if __name__ == "__main__":
    aTimef1, aTimef2, maxTempf1, maxTempf2, minTempf1, minTempf2, maxDevf1, maxDevf2, time, coord, edof = find_transient_temp_distribution( main = True )
    
    t_spots = [600, 1200, 1800, 2400, 3000, 3600 ]
    for i in range(nbr_of_plotpoints):


        cfv.figure(fig_size=(5,5))
        cfv.draw_nodal_values_shaded(aTimef1[:,20*i + 19], coord, edof, title="Temperature for Q_1 during t: " + str( t_spots[i] ) + " s")
        cfv.colorbar()

    plt.figure()
    plt.plot(time, maxTempf1, linestyle='-', color='red', label='Maximum temperature')
    plt.plot(time, minTempf1, linestyle='-', color='blue', label='Minimum temperature')
    plt.xlabel('Time - s')
    plt.ylabel('Temperature - Kelvin')
    plt.title('Temperature over time - F1')
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(time, maxDevf1, linestyle='-', color='orange', label='Maximum deviation')
    plt.xlabel('Time - s')
    plt.ylabel('Temp - Kelvin')
    plt.title('Maximum Deviation over time - F1')
    plt.grid(True)

    for i in range(nbr_of_plotpoints):

        cfv.figure(fig_size=(5,5))
        cfv.draw_nodal_values_shaded(aTimef2[:,20*i + 19], coord, edof, title="Temperature for Q_2 during t: " + str( t_spots[i] ) + " s")
        cfv.colorbar()

    plt.figure()
    plt.plot(time, maxTempf2, linestyle='-', color='red', label='Maximum temperature')
    plt.plot(time, minTempf2, linestyle='-', color='blue', label='Minimum temperature')
    plt.xlabel('Time - s')
    plt.ylabel('Temperature - Kelvin')
    plt.title('Temperature over time - F2')
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(time, maxDevf2, linestyle='-', color='orange', label='Maximum deviation')
    plt.xlabel('Time - s')
    plt.ylabel('Temp - Kelvin')
    plt.title('Maximum Deviation over time - F2')
    plt.grid(True)

    plt.show()
