import openmc
import numpy as np
from openmc.surface import ZCylinder

# Some alias for geometry building blocks
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
#########################Geometry parameters######################
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

# Throat inner radius in cm
throat_IR = 25.4
# Cryostat clearance in cm
# This is the distance from coil pack to cryostat in all directions
cryostat_dist = 5
# Cryostat thickness in cm
cryostat_thickness = 1.5
# First wall thickness in cm
fw_thickness = 0.1
# First wall support structure thickness in cm
fw_support_thickness = 5
# First wall cylinder end location in cm
fw_end = 130
# First wall inner radius in cm
fw_radius = 80
# Breeder blanket outer radius in cm
breeder_OR = 175
# Angle between axis and the expanding portion of shield in degree
expand_angle = 17.5
# Virtex of the cone for the expanding portion of shield in cm
expand_virtex = 210
# Virtex of the outer cone of the shield
shield_virtex = -30
# Angle between axis and the outer cone of the shield
shield_angle = 30

coil_zmin = coil_z-(coil_dz*coil_nz/2)
cryo_zmin = coil_zmin - cryostat_dist - cryostat_thickness
coil_zmax = coil_z+(coil_dz*coil_nz/2)
cryo_zmax = coil_zmax + cryostat_dist + cryostat_thickness

expand_tan = np.tan(expand_angle*np.pi/180)
expand_sin = np.sin(expand_angle*np.pi/180)
shield_tan = np.tan(shield_angle*np.pi/180)
##########################Universe Cell############################

# IMPORTANT: use p_vacz1 instead of ZPlane(0)
r[1] = -rpp(-199, 199, -239, 239, -199, 199)
#p_vacx1 = openmc.XPlane(-400, boundary_type='vacuum')
#p_vacx2 = openmc.XPlane(400, boundary_type='vacuum')
#p_vacy1 = openmc.YPlane(-400, boundary_type='vacuum')
#p_vacy2 = openmc.YPlane(400, boundary_type='vacuum')
p_vacz1 = openmc.ZPlane(0, boundary_type='reflective')
p_vacz2 = openmc.ZPlane(400, boundary_type='vacuum')
#r[1000] = +p_vacx1 & -p_vacx2 & +p_vacy1 & -p_vacy2 & +p_vacz1 & -p_vacz2
r[1000] = -openmc.ZCylinder(0, 0, 800, boundary_type='vacuum') & +p_vacz1 & -p_vacz2

############################Shield##################################
# The conical shield itself
r[1001] = +openmc.model.surface_composite.ZConeOneSided(0, 0, expand_virtex, expand_tan, False)
r[1001] &= +openmc.ZPlane(fw_end)
r[1001] &= -openmc.ZPlane(cryo_zmax)
r[1001] &= +openmc.ZCylinder(0, 0, throat_IR)
r[1001] &= -openmc.model.surface_composite.ZConeOneSided(0, 0, shield_virtex, shield_tan, True)
r[1001] &= -openmc.ZCylinder(0, 0, breeder_OR)

# Extended shield on the back of the magnets
r[1002] = -openmc.ZCylinder(coil_radius + coil_dr*coil_nr + 20)
r[1002] &= +openmc.ZPlane(cryo_zmax)
r[1002] &= -openmc.ZPlane(cryo_zmax+10)
r[1002] &= +openmc.ZCylinder(0, 0, throat_IR)

# vacuum inside the throat
r[1901] = -openmc.ZCylinder(0, 0, throat_IR)
r[1901] &= +p_vacz1
r[1901] &= -openmc.ZPlane(cryo_zmax + 15+3.81)

# vacuum inside the expanding portion
r[1902] = -openmc.model.surface_composite.ZConeOneSided(0, 0, expand_virtex, expand_tan, False)
r[1902] &= +p_vacz1
r[1902] &= +openmc.ZCylinder(0, 0, throat_IR)
r[1902] &= -openmc.ZCylinder(0, 0, fw_radius)

# Coil itself
r[2001] = -openmc.ZCylinder(0, 0, coil_radius)
r[2001] &= +openmc.ZPlane(coil_zmin) & -openmc.ZPlane(coil_zmax)
r[2002] = -openmc.ZCylinder(0, 0, coil_radius + coil_dr*coil_nr)
r[2002] &= +openmc.ZPlane(coil_zmin) & -openmc.ZPlane(coil_zmax)

# Cryostat big doughnut
r[4001] = -openmc.ZCylinder(0, 0, coil_radius+coil_dr*coil_nr+cryostat_thickness+cryostat_dist)
r[4001] &= +openmc.ZPlane(cryo_zmin) & -openmc.ZPlane(cryo_zmax)
r[4001] &= +openmc.ZCylinder(0, 0, coil_radius-cryostat_thickness-cryostat_dist)

# Cryostat small doughnut
r[4002] = -openmc.ZCylinder(0, 0, coil_radius+coil_dr*coil_nr+cryostat_dist)
r[4002] &= +openmc.ZPlane(cryo_zmin+cryostat_thickness) & -openmc.ZPlane(cryo_zmax-cryostat_thickness)
r[4002] &= +openmc.ZCylinder(0, 0, coil_radius-cryostat_dist)

# Close shield
r[3001] = -openmc.ZCylinder(0, 0, coil_radius+coil_dr*coil_nr+cryostat_thickness+cryostat_dist+5)
r[3001] &= +openmc.ZPlane(cryo_zmin-5) & -openmc.ZPlane(cryo_zmax)
r[3001] &= +openmc.ZCylinder(0, 0, coil_radius-cryostat_thickness-cryostat_dist-5)

# First wall cylinder
r[5001] = -openmc.ZCylinder(0, 0, fw_radius+fw_thickness)
r[5001] &= +openmc.ZCylinder(0, 0, fw_radius)
r[5001] &= +p_vacz1
r[5001] &= -openmc.model.surface_composite.ZConeOneSided(0, 0, expand_virtex, expand_tan, False)

# First wall throat cylinder
r[5002] = -openmc.ZCylinder(0, 0, throat_IR+fw_thickness)
r[5002] &= +openmc.ZCylinder(0, 0, throat_IR)
r[5002] &= +openmc.model.surface_composite.ZConeOneSided(0, 0, expand_virtex, expand_tan, False)
r[5002] &= -openmc.ZPlane(cryo_zmax)

# First wall exanding
r[5003] = +openmc.model.surface_composite.ZConeOneSided(0, 0, expand_virtex, expand_tan, False)
r[5003] &= -openmc.model.surface_composite.ZConeOneSided(0, 0, expand_virtex+fw_thickness/expand_sin, expand_tan, False)
r[5003] &= -openmc.ZCylinder(0, 0, fw_radius+fw_thickness)
r[5003] &= +openmc.ZCylinder(0, 0, throat_IR)

# First wall cylinder support
r[5101] = -openmc.ZCylinder(0, 0, fw_radius+fw_thickness+fw_support_thickness)
r[5101] &= +openmc.ZCylinder(0, 0, fw_radius+fw_thickness)
r[5101] &= +p_vacz1
r[5101] &= -openmc.model.surface_composite.ZConeOneSided(0, 0, expand_virtex+fw_thickness/expand_sin, expand_tan, False)

# First wall throat cylinder support
r[5102] = -openmc.ZCylinder(0, 0, throat_IR+fw_thickness+fw_support_thickness)
r[5102] &= +openmc.ZCylinder(0, 0, throat_IR+fw_thickness)
r[5102] &= +openmc.model.surface_composite.ZConeOneSided(0, 0, expand_virtex+fw_thickness/expand_sin, expand_tan, False)
r[5102] &= -openmc.ZPlane(cryo_zmax)

# First wall exanding support
r[5103] = +openmc.model.surface_composite.ZConeOneSided(0, 0, expand_virtex+fw_thickness/expand_sin, expand_tan, False)
r[5103] &= -openmc.model.surface_composite.ZConeOneSided(0, 0, expand_virtex+(fw_thickness+fw_support_thickness)/expand_sin, expand_tan, False)
r[5103] &= -openmc.ZCylinder(0, 0, fw_radius+fw_thickness+fw_support_thickness)
r[5103] &= +openmc.ZCylinder(0, 0, throat_IR+fw_thickness)
r[5103] &= +p_vacz1

# Breeder cylinder
r[6000] = -openmc.ZCylinder(0, 0, breeder_OR)
r[6000] &= +openmc.ZCylinder(0, 0, fw_radius+fw_thickness+fw_support_thickness)
r[6000] &= +p_vacz1
r[6000] &= -openmc.ZPlane(fw_end)

# Breeder cone
r[6001] = -openmc.ZCylinder(0, 0, breeder_OR)
r[6001] &= +openmc.model.surface_composite.ZConeOneSided(0, 0, shield_virtex, shield_tan, True)
r[6001] &= +openmc.model.surface_composite.ZConeOneSided(0, 0, expand_virtex+(fw_thickness+fw_support_thickness)/expand_sin, expand_tan, False)
r[6001] &= -openmc.ZPlane(cryo_zmax)
r[6001] &= +p_vacz1

# Breeder small cone
r[6002] = +openmc.model.surface_composite.ZConeOneSided(0, 0, expand_virtex+(fw_thickness+fw_support_thickness)/expand_sin, expand_tan, False)
r[6002] &= -openmc.ZPlane(fw_end)
r[6002] &= -openmc.model.surface_composite.ZConeOneSided(0, 0, shield_virtex, shield_tan, True)

# Breeder reflector
r[6100] = +openmc.ZCylinder(0, 0, breeder_OR)
r[6100] &= -openmc.ZCylinder(0, 0, breeder_OR+10)
r[6100] &= +p_vacz1
r[6100] &= -openmc.ZPlane(cryo_zmin)

# Vacuum around the breeder
r[6901] = +openmc.ZCylinder(0, 0, breeder_OR)
r[6901] &= +p_vacz1
r[6901] &= -openmc.ZPlane(cryo_zmax)

# End Tank plate
r[7001] = +openmc.ZPlane(cryo_zmax + 15)
r[7001] &= -openmc.ZPlane(cryo_zmax + 15 + 3.81)
r[7001] &= -openmc.ZCylinder(0, 0, 101.6)
r[7001] &= +openmc.ZCylinder(0, 0, throat_IR)

# End Tank wall
r[7002] = +openmc.ZPlane(cryo_zmax + 15+3.81)
r[7002] &= -p_vacz2
r[7002] &= -openmc.ZCylinder(0, 0, 101.6)
r[7002] &= +openmc.ZCylinder(0, 0, 100.97)

# End Tank vacuum
r[7901] = -openmc.ZCylinder(0, 0, 100.97)
r[7901] &= +openmc.ZPlane(cryo_zmax+15+3.81)
r[7901] &= -p_vacz2

# Vacuum around end tank
r[7902] = +openmc.ZCylinder(0, 0, 101.6)
r[7902] &= +openmc.ZPlane(cryo_zmax+15)
r[7902] &= -p_vacz2

# Vacuum behind magnets
r[7903] = +openmc.ZCylinder(0, 0, throat_IR)
r[7903] &= +openmc.ZPlane(cryo_zmax)
r[7903] &= -openmc.ZPlane(cryo_zmax+15)