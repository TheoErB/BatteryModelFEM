import calfem.geometry as cfg
import calfem.mesh as cfm
import matplotlib as mpl
import matplotlib.pyplot as plt
import calfem.vis_mpl as cfv

mpl.use('TkAgg')

# Mesh Data
element_size    = 0.015
el_type         = 2
dofs_pn         = 1

el_sizef = element_size
mesh_dir = "./"

#####      Geomety      #####
# Boundary markers
MARKER_red=0
MARKER_blue=1
MARKER_bottom=2
MARKER_top=3
MARKER_qn0=4

def generate_mesh(show_mesh=False):

    # Mesh measurements
    L_top       = 400 * 10**(-3)
    L_bottom    = 300 * 10**(-3)
    height      = 200 * 10**(-3)
    radii       = 25 * 10**(-3)
    circleDelta = 87.5 * 10**(-3)

    # initialize mesh
    g = cfg.geometry()

    # Add boundry points
    g.point([0, 0], 0)
    g.point([L_bottom, 0], 1)
    g.point([L_top, height], 2)
    g.point([0, height], 3)
    g.point([0, height/2+radii], 4)

    # Half circle
    g.point([0, height/2], 5)
    g.point([0, height/2-radii], 6)
    g.point([radii, height/2], 7)

    # Circle 1
    g.point([circleDelta, height/2], 8)
    g.point([circleDelta+radii, height/2], 9)
    g.point([circleDelta, height/2+radii], 10)
    g.point([circleDelta-radii, height/2], 11)
    g.point([circleDelta, height/2-radii], 12)

    # Circle 2
    g.point([circleDelta*2, height/2], 13)
    g.point([circleDelta*2+radii, height/2], 14)
    g.point([circleDelta*2, height/2+radii], 15)
    g.point([circleDelta*2-radii, height/2], 16)
    g.point([circleDelta*2, height/2-radii], 17)

    # Circle 3
    g.point([circleDelta*3, height/2], 18)
    g.point([circleDelta*3+radii, height/2], 19)
    g.point([circleDelta*3, height/2+radii], 20)
    g.point([circleDelta*3-radii, height/2], 21)
    g.point([circleDelta*3, height/2-radii], 22)

    # Define lines / circle segments

    g.spline([0, 1], 0, marker=MARKER_bottom)
    g.spline([1, 2], 1, marker=MARKER_bottom)
    g.spline([2, 3], 2, marker=MARKER_top)
    g.spline([3, 4], 3, marker=MARKER_qn0)

    g.circle([7, 5, 4], 4, marker=MARKER_red)
    g.circle([6, 5, 7], 5, marker=MARKER_red)

    g.spline([6, 0], 6, marker=MARKER_qn0)

    # Circle 1
    g.circle([9, 8, 10], 7, marker=MARKER_blue)
    g.circle([10, 8, 11], 8, marker=MARKER_blue)
    g.circle([11, 8, 12], 9, marker=MARKER_blue)
    g.circle([12, 8, 9], 10, marker=MARKER_blue)

    # Circle 2
    g.circle([14, 13, 15], 11, marker=MARKER_red)
    g.circle([15, 13, 16], 12, marker=MARKER_red)
    g.circle([16, 13, 17], 13, marker=MARKER_red)
    g.circle([17, 13, 14], 14, marker=MARKER_red)

    # Circle 3
    g.circle([19, 18, 20], 15, marker=MARKER_blue)
    g.circle([20, 18, 21], 16, marker=MARKER_blue)
    g.circle([21, 18, 22], 17, marker=MARKER_blue)
    g.circle([22, 18, 19], 18, marker=MARKER_blue)

    # Define surface
    g.surface([0, 1, 2, 3, 4, 5, 6], [[7, 8, 9, 10], [11, 12, 13, 14], [15, 16, 17, 18]])

    # Generate mesh
    mesh = cfm.GmshMeshGenerator(g, mesh_dir=mesh_dir, return_boundary_elements=True)
    mesh.el_size_factor = el_sizef
    mesh.el_type = el_type
    mesh.dofs_per_node = dofs_pn
    coord, edof, dofs, bdofs, element_markers, boundaryElements = mesh.create()

    if show_mesh:
        fig, ax = plt.subplots()
        cfv.draw_geometry(
            g,
            label_curves=True,
            title="Geometry: Projekt:)"
        )
        cfv.figure(fig_size=(5,5))
        cfv.draw_mesh(coords=coord, edof=edof, dofs_per_node=mesh.dofs_per_node, el_type=mesh.el_type, filled=True)
        plt.show()

    return coord, edof, dofs, bdofs, element_markers, boundaryElements

if __name__ == "__main__":
    generate_mesh(show_mesh=True)