import openmc
import numpy as np

rpp = openmc.model.surface_composite.RectangularParallelepiped
rcc = openmc.model.surface_composite.RightCircularCylinder
# Generalized circular cylinder/cyliinder from points
def gcc(p0, p1, r):
    # Vector from point 0 to point 1, normal to end planes
    nv = (p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2])
    plane1 = openmc.Quadric(0, 0, 0, 0, 0, 0, nv[0], nv[1], nv[2],
                            (nv[0]*p0[0] + nv[1]*p0[1] + nv[2]*p0[2]))
    plane2 = openmc.Quadric(0, 0, 0, 0, 0, 0, nv[0], nv[1], nv[2],
                            (nv[0]*p1[0] + nv[1]*p1[1] + nv[2]*p1[2]))
    cyl = openmc.model.cylinder_from_points(p0, p1, r)
    # Now to find the sign of the space in between the end planes
    # evaluate the equations at the mid point between the ends
    mid_pt = ((p1[0]+p0[0])/2, (p1[1]+p0[1])/2, (p1[2]+p0[2])/2)
    plane1_pt = nv[0]*mid_pt[0] + nv[1]*mid_pt[1] + nv[2]*mid_pt[2] + (nv[0]*p0[0] + nv[1]*p0[1] + nv[2]*p0[2])
    if plane1_pt < 0:
        return (-plane1 & +plane2 & -cyl)
    else:
        return (+plane1 & -plane2 & -cyl)

# First create a dictionary containing all regions
r = {}

# Magnet coil inner radius in cm
coil_radius = 75
# Magnet coil z-position from the center of machine to center of coil in cm
coil_z = 240
# Number of coil windings in Z
coil_nz = 16
# Number of coil windings in r
coil_nr = 16
# Conductor thickness in Z (cm)
coil_dz = 2.4
# Conductor thickness in r (cm)
coil_dr = 1.9
# Cryostat clearance in cm
# This is the distance from coil pack to cryostat in all directions
cryostat_dist = 5
# Cryostat thickness in cm
cryostat_thickness = 1.5

coil_zmin = coil_z-(coil_dz*coil_nz/2)
cryo_zmin = coil_zmin - cryostat_dist - cryostat_thickness
coil_zmax = coil_z+(coil_dz*coil_nz/2)
cryo_zmax = coil_zmax + cryostat_dist + cryostat_thickness

##########################Universe Cell############################

r[1] = -rpp(-199, 199, -239, 239, -199, 199)
p_vacx1 = openmc.XPlane(-400, boundary_type='vacuum')
p_vacx2 = openmc.XPlane(400, boundary_type='vacuum')
p_vacy1 = openmc.YPlane(-400, boundary_type='vacuum')
p_vacy2 = openmc.YPlane(400, boundary_type='vacuum')
p_vacz1 = openmc.ZPlane(-0.001, boundary_type='reflective')
p_vacz2 = openmc.ZPlane(400, boundary_type='vacuum')
r[1000] = +p_vacx1 & -p_vacx2 & +p_vacy1 & -p_vacy2 & +p_vacz1 & -p_vacz2
#r[1000] = -rpp(-400, 240, -240, 240, -200, 200, boundary_type='vacuum')

############################Shield##################################
# The conical itself
r[1001] = +openmc.model.surface_composite.ZConeOneSided(0, 0, 220, 0.3, False)
r[1001] &= +openmc.ZPlane(0)
r[1001] &= -openmc.ZPlane(175)
r[1001] &= +openmc.ZCylinder(0, 0, 25.4)
r[1001] &= -openmc.model.surface_composite.ZConeOneSided(0, 0, 0, 120/289, True)
r[1002] = -openmc.model.surface_composite.ZConeOneSided(0, 0, 0, 120/289, True)
r[1002] &= +openmc.ZPlane(175)
r[1002] &= -openmc.ZPlane(cryo_zmax)
r[1002] &= +openmc.ZCylinder(0, 0, 25.4)

# Coil itself
r[2001] = -openmc.ZCylinder(0, 0, coil_radius) 
r[2001] &= +openmc.ZPlane(coil_zmin) & -openmc.ZPlane(coil_zmax)
r[2002] = -openmc.ZCylinder(0, 0, coil_radius + coil_dr*coil_nr) 
r[2002] &= +openmc.ZPlane(coil_zmin) & -openmc.ZPlane(coil_zmax)

# Cryostat big doughnut
r[4001] = -openmc.ZCylinder(0, 0, coil_radius+coil_dr*coil_nr+cryostat_thickness+cryostat_dist)
r[4001] &= +openmc.ZPlane(cryo_zmin-cryostat_thickness) & -openmc.ZPlane(cryo_zmax+cryostat_thickness)
r[4001] &= +openmc.ZCylinder(0, 0, coil_radius-cryostat_thickness-cryostat_dist)

# Cryostat small doughnut
r[4002] = -openmc.ZCylinder(0, 0, coil_radius+coil_dr*coil_nr+cryostat_dist)
r[4002] &= +openmc.ZPlane(cryo_zmin) & -openmc.ZPlane(cryo_zmax)
r[4002] &= +openmc.ZCylinder(0, 0, coil_radius-cryostat_dist)

