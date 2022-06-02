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
throat_IR = 28
# Cryostat clearance in cm
# This is the distance from coil pack to cryostat in all directions
cryostat_dist = 3
# Cryostat thickness in cm
cryostat_thickness = 1
# First wall thickness in cm
fw_thickness = 0.1
# First wall support structure thickness in cm
fw_support_thickness = 2
# Shield end location in cm
sh_end = 132.5
# First wall inner radius in cm
fw_radius = 77.5
# Breeder cylinder outer radius in cm
breeder_OR = 175
# Breeder blanket radial extension thickness
breeder_extension = 75
# Angle between axis and the expanding portion of shield in degree
expand_angle = 7.5
# Virtex of the cone for the expanding portion of shield in cm
expand_virtex = 250
# Virtex of the outer cone of the shield
shield_virtex = -30
# Angle between axis and the outer cone of the shield
shield_angle = 20
# End expander angle
end_angle = 8
# End expander breeder radius
end_radius = 450
# End expander breeder thickness
end_thickness = 120
# Expander tank radius
expand_tank_radius = 150

# Thickness for the testing region
test_thickness = 5

coil_zmin = coil_z-(coil_dz*coil_nz/2)
cryo_zmin = coil_zmin - cryostat_dist - cryostat_thickness
coil_zmax = coil_z+(coil_dz*coil_nz/2)
cryo_zmax = coil_zmax + cryostat_dist + cryostat_thickness

expand_tan = np.tan(expand_angle*np.pi/180)
expand_sin = np.sin(expand_angle*np.pi/180)
shield_tan = np.tan(shield_angle*np.pi/180)
chamfer_tan = np.tan((shield_angle+30)*np.pi/180)
end_tan = np.tan(end_angle*np.pi/180)
##########################Universe Cell############################

#r[1] = -rpp(-199, 199, -239, 239, -199, 199)
#p_vacx1 = openmc.XPlane(-400, boundary_type='vacuum')
#p_vacx2 = openmc.XPlane(400, boundary_type='vacuum')
#p_vacy1 = openmc.YPlane(-400, boundary_type='vacuum')
#p_vacy2 = openmc.YPlane(400, boundary_type='vacuum')
p_vacz1 = openmc.ZPlane(0, boundary_type='reflective')
p_vacz2 = openmc.ZPlane(1000, boundary_type='vacuum')
#r[1000] = +p_vacx1 & -p_vacx2 & +p_vacy1 & -p_vacy2 & +p_vacz1 & -p_vacz2
r[1000] = -openmc.ZCylinder(0, 0, 800, boundary_type='vacuum') & +p_vacz1 & -p_vacz2

# conical surfaces
expand_cone_outer = openmc.model.surface_composite.ZConeOneSided(0, 0, expand_virtex+(fw_thickness+fw_support_thickness)/expand_sin/2, expand_tan, False)
expand_cone_inner = openmc.model.surface_composite.ZConeOneSided(0, 0, expand_virtex+fw_thickness/expand_sin/2, expand_tan, False)
expand_cone_throat = openmc.model.surface_composite.ZConeOneSided(0, 0, expand_virtex, expand_tan, False)

chamfer_cone = openmc.model.surface_composite.ZConeOneSided(0, 0, shield_virtex+90, chamfer_tan, True)
shield_cone = openmc.model.surface_composite.ZConeOneSided(0, 0, shield_virtex, shield_tan, True)

end_cone = openmc.model.surface_composite.ZConeOneSided(0, 0, 75, end_tan, True)
# first wall surfaces
throat_outer_cyl = openmc.ZCylinder(0, 0, throat_IR+fw_thickness+fw_support_thickness)
throat_fw_cylinder = openmc.ZCylinder(0, 0, throat_IR+fw_thickness)
throat_inner_cyl = openmc.ZCylinder(0, 0, throat_IR)

chamber_outer_cyl = openmc.ZCylinder(0, 0, fw_radius+fw_thickness+fw_support_thickness)
chamber_fw_cyl = openmc.ZCylinder(0, 0, fw_radius+fw_thickness)
chamber_inner_cyl = openmc.ZCylinder(0, 0, fw_radius)

test_outer_cyl = openmc.ZCylinder(0, 0, fw_radius+fw_thickness+fw_support_thickness+test_thickness)
test_throat_outer_cyl = openmc.ZCylinder(0, 0, throat_IR+fw_thickness+fw_support_thickness+test_thickness)
test_expand_cone_outer = openmc.model.surface_composite.ZConeOneSided(0, 0, 
                         expand_virtex+(fw_thickness+fw_support_thickness+test_thickness)/expand_sin/2,
                         expand_tan, False)

# shield planes
cryo_max_zplane = openmc.ZPlane(cryo_zmax)
shield_max_zplane = openmc.ZPlane(cryo_zmax+15)
shield_min_zplane = openmc.ZPlane(sh_end)

# breeder and end tank cylinders
tank_outer_cyl = openmc.ZCylinder(0, 0, expand_tank_radius+0.7)
breeder_inner_cyl = openmc.ZCylinder(0, 0, breeder_OR+breeder_extension)
reflector_outer_cyl = openmc.ZCylinder(0, 0, breeder_OR+breeder_extension+10)

# end tank faces
end_inner_cyl = openmc.ZCylinder(0, 0, expand_tank_radius)
end_ring_sph = openmc.Sphere(0, 0, 75, end_radius+end_thickness)
end_breeder_sph = openmc.Sphere(0, 0, 75, end_radius)
end_breeder_cyl = openmc.ZCylinder(0, 0, expand_tank_radius+0.7+end_thickness)

############################Shield##################################
# The conical shield itself
r[1001] = +expand_cone_outer
r[1001] &= -chamfer_cone
r[1001] &= +shield_min_zplane
r[1001] &= -cryo_max_zplane
r[1001] &= +throat_outer_cyl
r[1001] &= -shield_cone
r[1001] &= -tank_outer_cyl

# Extended shield on the back of the magnets
r[1002] = -tank_outer_cyl
r[1002] &= +cryo_max_zplane
r[1002] &= -shield_max_zplane
r[1002] &= +throat_outer_cyl

# vacuum inside the throat
r[1901] = -throat_inner_cyl
r[1901] &= +p_vacz1
r[1901] &= -openmc.ZPlane(cryo_zmax + 15+3.81)

# vacuum inside the expanding portion
r[1902] = -expand_cone_throat
r[1902] &= +p_vacz1
r[1902] &= +throat_inner_cyl
r[1902] &= -chamber_inner_cyl

# Coil itself
r[2001] = -openmc.ZCylinder(0, 0, coil_radius)
r[2001] &= +openmc.ZPlane(coil_zmin) & -openmc.ZPlane(coil_zmax)

r[2002] = -openmc.ZCylinder(0, 0, coil_radius + coil_dr*coil_nr)
r[2002] &= +openmc.ZPlane(coil_zmin) & -openmc.ZPlane(coil_zmax)

# Cryostat big doughnut
r[4001] = -openmc.ZCylinder(0, 0, coil_radius+coil_dr*coil_nr+cryostat_thickness+cryostat_dist)
r[4001] &= +openmc.ZPlane(cryo_zmin) & -cryo_max_zplane
r[4001] &= +openmc.ZCylinder(0, 0, coil_radius-cryostat_thickness-cryostat_dist)

# Cryostat small doughnut
r[4002] = -openmc.ZCylinder(0, 0, coil_radius+coil_dr*coil_nr+cryostat_dist)
r[4002] &= +openmc.ZPlane(cryo_zmin+cryostat_thickness) & -openmc.ZPlane(cryo_zmax-cryostat_thickness)
r[4002] &= +openmc.ZCylinder(0, 0, coil_radius-cryostat_dist)

# Close shield
r[3001] = -openmc.ZCylinder(0, 0, coil_radius+coil_dr*coil_nr+cryostat_thickness+cryostat_dist+5)
r[3001] &= +openmc.ZPlane(cryo_zmin-5) & -cryo_max_zplane
r[3001] &= +openmc.ZCylinder(0, 0, coil_radius-cryostat_thickness-cryostat_dist-5)

# First wall cylinder
r[5001] = -chamber_fw_cyl
r[5001] &= +chamber_inner_cyl
r[5001] &= +p_vacz1
r[5001] &= -expand_cone_throat

# First wall throat cylinder
r[5002] = -throat_fw_cylinder
r[5002] &= +throat_inner_cyl
r[5002] &= +expand_cone_throat
r[5002] &= -shield_max_zplane

# First wall exanding
r[5003] = +expand_cone_throat
r[5003] &= -expand_cone_inner
r[5003] &= -chamber_fw_cyl
r[5003] &= +throat_inner_cyl

# First wall cylinder support
r[5101] = -chamber_outer_cyl
r[5101] &= +chamber_fw_cyl
r[5101] &= +p_vacz1
r[5101] &= -expand_cone_inner

# First wall throat cylinder support
r[5102] = -throat_outer_cyl
r[5102] &= +throat_fw_cylinder
r[5102] &= +expand_cone_inner
r[5102] &= -shield_max_zplane

# First wall exanding support
r[5103] = +expand_cone_inner
r[5103] &= -expand_cone_outer
r[5103] &= -chamber_outer_cyl
r[5103] &= +throat_outer_cyl
r[5103] &= +p_vacz1

# Material testing cylinder region
r[5201] = -test_outer_cyl
r[5201] &= +chamber_outer_cyl
r[5201] &= +p_vacz1
r[5201] &= -expand_cone_outer

# Material testing throat region
r[5202] = -test_outer_cyl
r[5202] &= +throat_outer_cyl
r[5202] &= +expand_cone_inner
r[5202] &= -shield_max_zplane

# Material testing expander region
r[5203] = +expand_cone_outer
r[5203] &= -test_expand_cone_outer
r[5203] &= -test_outer_cyl
r[5203] &= +throat_outer_cyl
r[5203] &= -shield_min_zplane

# Breeder cylinder
r[6000] = -tank_outer_cyl
r[6000] &= +chamber_outer_cyl
r[6000] &= +p_vacz1
r[6000] &= -shield_min_zplane

# Breeder cone
r[6001] = -tank_outer_cyl
r[6001] &= +shield_cone
r[6001] &= +expand_cone_outer
r[6001] &= -cryo_max_zplane
r[6001] &= +shield_min_zplane

# Breeder small cone
r[6002] = +expand_cone_outer
r[6002] &= -shield_min_zplane
r[6002] &= -chamber_outer_cyl

# Breeder Extension
r[6003] = +p_vacz1
r[6003] &= -breeder_inner_cyl
r[6003] &= +openmc.ZCylinder(0, 0, breeder_OR)
r[6003] &= -cryo_max_zplane

# Breeder behind end cell bias rings
r[6004] = -end_ring_sph
r[6004] &= +end_breeder_sph
r[6004] &= -end_cone
r[6004] &= -end_inner_cyl

# Breeder wedge to fill chamfer
r[6005] = +shield_min_zplane
r[6005] &= +chamfer_cone
r[6005] &= -shield_cone

# Breeder reflector
r[6100] = +breeder_inner_cyl
r[6100] &= -reflector_outer_cyl
r[6100] &= +p_vacz1
r[6100] &= -cryo_max_zplane

# Breeder reflector for extension
r[6101] = +end_breeder_cyl
r[6101] &= -reflector_outer_cyl
r[6101] &= +cryo_max_zplane
r[6101] &= -openmc.ZPlane(cryo_zmax+10)

# Breeder reflector behind the end cell breeders
r[6102] = +end_ring_sph
r[6102] &= -openmc.Sphere(0, 0, 75, end_radius+end_thickness+10)
r[6102] &= -end_cone
r[6102] &= -end_inner_cyl

# Breeder to the side of the expanding tank
r[6201] = -breeder_inner_cyl
r[6201] &= +tank_outer_cyl
r[6201] &= +p_vacz1
r[6201] &= -openmc.ZPlane(end_radius+end_thickness+50)

# Breeder between the end tank and the main breeder
r[6202] = -shield_max_zplane
r[6202] &= +cryo_max_zplane
r[6202] &= -end_breeder_cyl
r[6202] &= +throat_fw_cylinder
r[6202] &= +shield_cone

# Breeder reflector around the side of the expanding tank
r[6203] = -reflector_outer_cyl
r[6203] &= -openmc.ZPlane(end_radius+end_thickness+50)
r[6203] &= +cryo_max_zplane
r[6203] &= +breeder_inner_cyl

# Breeder reflector around the end of the expanding tank
r[6204] = -reflector_outer_cyl
r[6204] &= +openmc.ZPlane(end_radius+end_thickness+50)
r[6204] &= -openmc.ZPlane(end_radius+end_thickness+60)
r[6204] &= +tank_outer_cyl

# Vacuum around the breeder
r[6901] = +reflector_outer_cyl
r[6901] &= +p_vacz1
r[6901] &= -p_vacz2
r[6901] &= -openmc.ZCylinder(0, 0, 800, boundary_type='vacuum')

# End Tank plate
r[7001] = +shield_max_zplane
r[7001] &= -openmc.ZPlane(cryo_zmax+15+3.81)
r[7001] &= -tank_outer_cyl
r[7001] &= +throat_inner_cyl

# End Tank wall
r[7002] = +openmc.ZPlane(cryo_zmax+15+3.81)
r[7002] &= -p_vacz2
r[7002] &= -tank_outer_cyl
r[7002] &= +end_inner_cyl

# End cell converter/bias rings
r[7101] = -openmc.Sphere(0, 0, 75, end_radius-25)
r[7101] &= +openmc.Sphere(0, 0, 75, end_radius-50)
r[7101] &= -end_cone
r[7101] &= -end_inner_cyl

# End Tank vacuum
r[7901] = -end_inner_cyl
r[7901] &= +openmc.ZPlane(cryo_zmax+15+3.81)
r[7901] &= -end_breeder_sph

# End Tank vacuum beyond blanket
r[7902] = -p_vacz2
r[7902] &= -end_inner_cyl
r[7902] &= +openmc.Sphere(0, 0, 75, end_radius+end_thickness+10)

# Vacuum around end tank
r[7903] = -p_vacz2
r[7903] &= +tank_outer_cyl
r[7903] &= -reflector_outer_cyl
r[7903] &= +openmc.ZPlane(end_radius+end_thickness+60)