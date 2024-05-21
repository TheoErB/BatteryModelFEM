
import calfem.utils as cfu
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('TkAgg')
import calfem.vis_mpl as cfv
import numpy as np
import calfem.core as cfc

import transient_temperature_distribution as transtemp

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

# Boundry marker data 
MARKER_red=0
MARKER_blue=1
MARKER_bottom=2
MARKER_top=3
MARKER_qn0=4

def compute_internal_strain( max_an, coord, dofs, edof, bdofs, nel, Ex, Ey ):
    
    # Initiate new degrees of freedom, due to all nodes now having 2 dof
    new_dofs = np.array([1, 2])
    new_ndofs = 2

    # Using old dof, create a new ndof 
    for i in range(1, len(dofs)):
        new_stack = np.array([2*i+1, 2*i+2])
        new_dofs = np.vstack( (new_dofs, new_stack) )
        new_ndofs += 2


    element_temps = []
    new_edof = ""

    # For each element
    for element in edof:
        
        # Calculate the average temperature deviation from T_zero 
        avg_delta_temp = np.abs( ( max_an[element[0]-1] + max_an[element[1]-1] + max_an[element[2]-1] ) / 3 - T_zero )
        element_temps.append(avg_delta_temp)

        new_el_dof = []

        # Create a new edof matrix using previus edof and all new nodes dof
        for e in element:

            new_el_dof.append( new_dofs[e-1][0] )
            new_el_dof.append( new_dofs[e-1][1] )

        new_el_dof = np.array(new_el_dof)
        
        if type(new_edof) == str:
            new_edof = new_el_dof
        
        else:
            new_edof = np.vstack( ( new_edof, new_el_dof ) )

    # ----- ASSEMBLING ----- #

    # Initiate the constituive matrix D
    D = Emod / ( (1 + ny) * (1 - 2*ny) ) * np.array([   [ 1 - ny, ny    , 0                 ],
                                                        [ ny    , 1 - ny, 0                 ],
                                                        [ 0     , 0     , 0.5 * (1 - 2*ny)  ]])
    
    # Initiate the K-matrix and F_0 vector
    K = np.zeros([new_ndofs, new_ndofs])
    F_0 = np.zeros([new_ndofs, 1])

    # Assembling K and F_0
    for i in range(0, nel):
        ex = np.array(Ex[i, :])
        ey = np.array(Ey[i, :])

        # Calculating element stress matrix for each element
        e_0 = (1 + ny) * alpha_exp * float(element_temps[i]) * np.array([[1], [1], [0]])
        e_0 = np.matmul(D, e_0)

        Ke = cfc.plante(ex, ey, [2, t], D)
        fe = cfc.plantf(ex, ey, [2, t], np.transpose(e_0))

        cfc.assem(new_edof[i, :], K, Ke, F_0, fe)


    new_bdofs = {}

    # Creating the new bdof matrix according to all boundary nodes new dof
    for i in bdofs.keys():
        new_bdofs_temp = []

        for j in bdofs[i]:

            new_bdofs_temp.append( new_dofs[j-1][0] )
            new_bdofs_temp.append( new_dofs[j-1][1] )

        new_bdofs[i] = new_bdofs_temp

    # Marking boundary conditions, bottom boundary is fixed and the cut created in the battery only moved in y-direction
    bc, bc_value = np.array([], 'i'), np.array([], 'f')
    bc, bc_value = cfu.applybc(new_bdofs, bc, bc_value, MARKER_bottom, 0, 0)
    bc, bc_value = cfu.applybc(new_bdofs, bc, bc_value, MARKER_qn0, 0, 1)

    # Solve for the nodal displacement and extract the displacement
    a, r = cfc.solveq(K, F_0, bc, bc_value)
    ed = cfc.extract_ed(new_edof,a)
    
    stress = []

    # Using the element displacement, calculate all components of each element stress matrix
    for i in range(nel):
        ex = np.array(Ex[i, :])
        ey = np.array(Ey[i, :])

        el_stress , el_strain = cfc.plants(ex, ey, [2, t], D, ed[i,:])
        
        sig_xx = float( el_stress[0][0] )
        sig_yy = float( el_stress[0][1] )
        tau_xy = float( el_stress[0][2] )

        # Including the temperature dependent part of respective stress
        sig_zz = float( ny * (sig_xx + sig_yy) - alpha_exp * Emod * element_temps[i] )

        sig_xx = sig_xx - (alpha_exp * Emod * float(element_temps[i])) / (1 - 2*ny) 
        sig_yy = sig_yy - (alpha_exp * Emod * float(element_temps[i])) / (1 - 2*ny) 

        # Calculating von Mises
        von_mises = np.sqrt( sig_xx**2 + sig_yy**2 + sig_zz**2 - sig_xx*sig_yy - sig_xx*sig_zz - sig_zz*sig_yy + 3 * tau_xy**2)

        stress.append(von_mises)

    print(np.max(stress))

    cfv.figure(fig_size=(5,5))
    cfv.draw_displacements(np.multiply(100, a), coord, new_edof, 2, 2 )
    
    
    cfv.figure(fig_size=(5,5))
    cfv.draw_element_values(stress, coord, new_edof, 2, 2, a, draw_elements=False)
    cfv.colorbar()
    plt.show()

if __name__ == "__main__":
    max_a_1, max_a_2, coord, dofs, edof, bdofs, nel, Ex, Ey = transtemp.find_transient_temp_distribution()
    compute_internal_strain( max_a_1, coord, dofs, edof, bdofs, nel, Ex, Ey )
    compute_internal_strain( max_a_2, coord, dofs, edof, bdofs, nel, Ex, Ey )