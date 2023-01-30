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
coil_z = 450
# Number of coil windings in Z
coil_nz = 1
# Number of coil windings in r
coil_nr = 1
# Conductor thickness in Z (cm)
coil_dz = 106
# Conductor thickness in r (cm)
coil_dr = 40

# Divertor coil inner radius in cm
divertor_radius = 200
# Divertor coil z-position (to center of coil)
divertor_z = 225
# Divertor coil windings in z
divertor_nz = 1
# Divertor coil windings in r
divertor_nr = 1
# Divertor conductor thickness in Z (cm)
divertor_dz = 21
# Divertor conductor thickness in R (cm)
divertor_dr = 16

# Midplane coil inner radius in cm
midplane_radius = 200.01
# Midplane coil z-position (to beginning of coil)
midplane_z = 75 - 21/2
# Midplane coil windings in z (on one side)
midplane_nz = 1
# Midplane coil windings in r
midplane_nr = 1
# Midplane conductor thickness in Z (cm)
midplane_dz = 21
# Midplane conductor thickness in R (cm)
midplane_dr = 16

# Throat inner radius in cm
throat_IR = 21
# Cryostat clearance in cm
# This is the distance from coil pack to cryostat in all directions
cryostat_dist = 3
# Cryostat thickness in cm
cryostat_thickness = 15.6
# Close shield thickness in cm
close_shield_thickness = 2.54
# First wall thickness in cm
fw_thickness = 0.254
# First wall support structure thickness in cm
fw_support_thickness = 2.54
# Shield end location (min z) in cm
sh_end = 340
# First wall inner radius in cm
fw_radius = 40
# Breeder cylinder outer radius in cm
breeder_OR = 160
# Breeder blanket radial extension thickness
breeder_extension = 150
# Angle between axis and the expanding portion of shield in degree
expand_angle = 8.5
# Virtex of the cone for the expanding portion of shield in cm
expand_virtex = 425
# Virtex of the outer cone of the shield
shield_virtex = 125
# Angle between axis and the outer cone of the shield
shield_outer_angle = 20
# Angle between axis and the inner cone of the shield
shield_inner_angle = 65
# End expander angle
end_angle = 8
# End expander breeder radius
end_radius = 850
# End expander breeder thickness
end_thickness = 120
# Expander tank radius
expand_tank_radius = 165

# Reflector Thickness
reflector_thickness = 25.4

# Thickness for the testing region
test_thickness = 5.08

coil_zmin = coil_z-(coil_dz*coil_nz/2)
cryo_zmin = coil_zmin - cryostat_dist - cryostat_thickness
coil_zmax = coil_z+(coil_dz*coil_nz/2)
cryo_zmax = coil_zmax + cryostat_dist + cryostat_thickness

midplane_coil_zmin = midplane_z
midplane_cryo_zmin = midplane_coil_zmin - cryostat_dist - cryostat_thickness
midplane_coil_zmax = midplane_z+(midplane_dz*midplane_nz)
midplane_cryo_zmax = midplane_coil_zmax + cryostat_dist + cryostat_thickness

divertor_zmin = divertor_z-(divertor_dz*divertor_nz/2)
divertor_zmax = divertor_z+(divertor_dz*divertor_nz/2)

divertor_coil_zmin = divertor_zmin
divertor_cryo_zmin = divertor_coil_zmin - cryostat_dist - cryostat_thickness
divertor_coil_zmax = divertor_zmax
divertor_cryo_zmax = divertor_coil_zmax + cryostat_dist + cryostat_thickness

expand_tan = np.tan(expand_angle*np.pi/180)
expand_sin = np.sin(expand_angle*np.pi/180)
shield_tan = np.tan(shield_outer_angle*np.pi/180)
chamfer_tan = np.tan((shield_inner_angle)*np.pi/180)
end_tan = np.tan(end_angle*np.pi/180)
##########################Universe Cell############################

#r[1] = -rpp(-199, 199, -239, 239, -199, 199)
#p_vacx1 = openmc.XPlane(-400, boundary_type='vacuum')
#p_vacx2 = openmc.XPlane(400, boundary_type='vacuum')
#p_vacy1 = openmc.YPlane(-400, boundary_type='vacuum')
#p_vacy2 = openmc.YPlane(400, boundary_type='vacuum')
p_vacz1 = openmc.ZPlane(0, boundary_type='reflective')
p_vacz2 = openmc.ZPlane(1500, boundary_type='vacuum')
p_vaccyl = openmc.ZCylinder(0, 0, 800, boundary_type='vacuum')
#r[1000] = +p_vacx1 & -p_vacx2 & +p_vacy1 & -p_vacy2 & +p_vacz1 & -p_vacz2
r[1000] = -p_vaccyl & +p_vacz1 & -p_vacz2

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
end_max_zplane = openmc.ZPlane(cryo_zmax+15+3.81)
shield_min_zplane = openmc.ZPlane(sh_end)

# breeder and end tank cylinders
breeder_inner_cyl = openmc.ZCylinder(0, 0, breeder_OR+breeder_extension)
reflector_outer_cyl = openmc.ZCylinder(0, 0, breeder_OR+breeder_extension+reflector_thickness)

# end tank faces
tank_inner_cyl = openmc.ZCylinder(0, 0, expand_tank_radius)
tank_outer_cyl = openmc.ZCylinder(0, 0, expand_tank_radius+0.7)
end_ring_sph = openmc.Sphere(0, 0, 75, end_radius+end_thickness)
end_breeder_sph = openmc.Sphere(0, 0, 75, end_radius)
end_reflector_sph = openmc.Sphere(0, 0, 75, end_radius+end_thickness+10)
end_breeder_cyl = openmc.ZCylinder(0, 0, expand_tank_radius+0.7+end_thickness)
end_reflector_zmin = openmc.ZPlane(end_radius+end_thickness+50)
end_reflector_zmax = openmc.ZPlane(end_radius+end_thickness+60)

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
r[1002] &= +shield_min_zplane

# vacuum inside the throat
r[1901] = -throat_inner_cyl
r[1901] &= +p_vacz1
r[1901] &= -openmc.ZPlane(cryo_zmax + 15+3.81)

# vacuum inside the expanding portion
r[1902] = -expand_cone_throat
r[1902] &= +p_vacz1
r[1902] &= +throat_inner_cyl
r[1902] &= -chamber_inner_cyl

# Mirror coil (inner and outer)
r[2001] = -openmc.ZCylinder(0, 0, coil_radius)
r[2001] &= +openmc.ZPlane(coil_zmin) & -openmc.ZPlane(coil_zmax)

r[2002] = -openmc.ZCylinder(0, 0, coil_radius + coil_dr*coil_nr)
r[2002] &= +openmc.ZPlane(coil_zmin) & -openmc.ZPlane(coil_zmax)

# Divertor coil (inner and outer)
r[2101] = -openmc.ZCylinder(0, 0, divertor_radius)
r[2101] &= +openmc.ZPlane(divertor_zmin) & -openmc.ZPlane(divertor_zmax)

r[2102] = -openmc.ZCylinder(0, 0, divertor_radius + divertor_dr*divertor_nr)
r[2102] &= +openmc.ZPlane(divertor_zmin) & -openmc.ZPlane(divertor_zmax)

# Central coil (inner and outer)
r[2201] = -openmc.ZCylinder(0, 0, midplane_radius)
#r[2201] &= +p_vacz1 & -openmc.ZPlane(midplane_z+midplane_dz*midplane_nz)
r[2201] &= +openmc.ZPlane(midplane_z) & -openmc.ZPlane(midplane_z+midplane_dz*midplane_nz)

r[2202] = -openmc.ZCylinder(0, 0, midplane_radius + midplane_dr*midplane_nr)
#r[2202] &= +p_vacz1 & -openmc.ZPlane(midplane_z+midplane_dz*midplane_nz)
r[2202] &= +openmc.ZPlane(midplane_z) & -openmc.ZPlane(midplane_z+midplane_dz*midplane_nz)

# Close shield
r[3001] = -openmc.ZCylinder(0, 0, coil_radius+coil_dr*coil_nr+cryostat_thickness+cryostat_dist+close_shield_thickness)
r[3001] &= +openmc.ZPlane(cryo_zmin-close_shield_thickness) & -cryo_max_zplane
r[3001] &= +openmc.ZCylinder(0, 0, coil_radius-cryostat_thickness-cryostat_dist-close_shield_thickness)

# Cryostat big doughnut
r[4001] = -openmc.ZCylinder(0, 0, coil_radius+coil_dr*coil_nr+cryostat_thickness+cryostat_dist)
r[4001] &= +openmc.ZPlane(cryo_zmin) & -cryo_max_zplane
r[4001] &= +openmc.ZCylinder(0, 0, coil_radius-cryostat_thickness-cryostat_dist)

# Cryostat small doughnut
r[4002] = -openmc.ZCylinder(0, 0, coil_radius+coil_dr*coil_nr+cryostat_dist)
r[4002] &= +openmc.ZPlane(cryo_zmin+cryostat_thickness) & -openmc.ZPlane(cryo_zmax-cryostat_thickness)
r[4002] &= +openmc.ZCylinder(0, 0, coil_radius-cryostat_dist)

# Central coil cryostat big doughnut
r[4101] = -openmc.ZCylinder(0, 0, midplane_radius+midplane_dr*midplane_nr+cryostat_thickness+cryostat_dist)
r[4101] &= +openmc.ZPlane(midplane_cryo_zmin) & -openmc.ZPlane(midplane_cryo_zmax)
r[4101] &= +openmc.ZCylinder(0, 0, midplane_radius-cryostat_thickness-cryostat_dist)

# Cryostat small doughnut
r[4102] = -openmc.ZCylinder(0, 0, midplane_radius+midplane_dr*midplane_nr+cryostat_dist)
r[4102] &= +openmc.ZPlane(midplane_cryo_zmin+cryostat_thickness) & -openmc.ZPlane(midplane_cryo_zmax-cryostat_thickness)
r[4102] &= +openmc.ZCylinder(0, 0, midplane_radius-cryostat_dist)

# Divertor coil cryostat big doughnut
r[4201] = -openmc.ZCylinder(0, 0, divertor_radius+divertor_dr*divertor_nr+cryostat_thickness+cryostat_dist)
r[4201] &= +openmc.ZPlane(divertor_cryo_zmin) & -openmc.ZPlane(divertor_cryo_zmax)
r[4201] &= +openmc.ZCylinder(0, 0, divertor_radius-cryostat_thickness-cryostat_dist)

# Divertor small doughnut
r[4202] = -openmc.ZCylinder(0, 0, divertor_radius+divertor_dr*divertor_nr+cryostat_dist)
r[4202] &= +openmc.ZPlane(divertor_cryo_zmin+cryostat_thickness) & -openmc.ZPlane(divertor_cryo_zmax-cryostat_thickness)
r[4202] &= +openmc.ZCylinder(0, 0, divertor_radius-cryostat_dist)

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
r[5003] &= +throat_fw_cylinder

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
r[6004] &= -tank_inner_cyl

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
r[6102] &= -end_reflector_sph
r[6102] &= -end_cone
r[6102] &= -tank_inner_cyl

# Breeder to the side of the expanding tank
r[6201] = -breeder_inner_cyl
r[6201] &= +tank_outer_cyl
r[6201] &= +p_vacz1
r[6201] &= -end_reflector_zmin

# Breeder between the end tank and the main breeder
r[6202] = -shield_max_zplane
r[6202] &= +cryo_max_zplane
r[6202] &= -end_breeder_cyl
r[6202] &= +throat_fw_cylinder
r[6202] &= +shield_cone

# Breeder reflector around the side of the expanding tank
r[6203] = -reflector_outer_cyl
r[6203] &= -end_reflector_zmin
r[6203] &= +cryo_max_zplane
r[6203] &= +breeder_inner_cyl

# Breeder reflector around the end of the expanding tank
r[6204] = -reflector_outer_cyl
r[6204] &= +end_reflector_zmin
r[6204] &= -end_reflector_zmax
r[6204] &= +tank_outer_cyl

# Vacuum around the breeder
r[6901] = +reflector_outer_cyl
r[6901] &= +p_vacz1
r[6901] &= -p_vacz2
r[6901] &= -p_vaccyl

# End Tank plate
r[7001] = +shield_max_zplane
r[7001] &= -end_max_zplane
r[7001] &= -tank_outer_cyl
r[7001] &= +throat_inner_cyl

# End Tank wall
r[7002] = +end_max_zplane
r[7002] &= -end_reflector_sph
r[7002] &= -tank_outer_cyl
r[7002] &= +tank_inner_cyl

# End cell converter/bias rings
r[7101] = -openmc.Sphere(0, 0, 75, end_radius-25)
r[7101] &= +openmc.Sphere(0, 0, 75, end_radius-50)
r[7101] &= -end_cone
r[7101] &= -tank_inner_cyl

# End Tank vacuum
r[7901] = -tank_inner_cyl
r[7901] &= +end_max_zplane
r[7901] &= -end_breeder_sph

# End Tank vacuum beyond blanket
r[7902] = -p_vacz2
r[7902] &= -tank_outer_cyl
r[7902] &= +end_reflector_sph

# Vacuum around end tank
r[7903] = -p_vacz2
r[7903] &= +tank_outer_cyl
r[7903] &= -reflector_outer_cyl
r[7903] &= +end_reflector_zmax