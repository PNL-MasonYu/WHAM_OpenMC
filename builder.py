# %%
import openmc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from material import materials
from boolean import c, root_cells, root
from source import vns_sources, worst_source
import plotting as p

working_directory = "./test_run"

#######################Simulation Settings###########################
settings = openmc.Settings()

settings.run_mode = 'fixed source'
#settings.source = vns_sources
# Alternatively use the worst-case scenario source
settings.source = worst_source

settings.particles = int(100000)
settings.batches = 300
settings.output = {'tallies': False}
#settings.max_lost_particles = int(settings.particles / 2e4)
#settings.verbosity = 7
#settings.seed = 53713
#settings.trace = (1, 1, 52687)
#settings.survival_bias = True
settings.photon_transport = True
settings.export_to_xml(working_directory)
settings.export_to_xml("./")

# Set OPENMC_CROSS_SECTIONS environment variable to the path to cross_sections.xml, or
# modify this on different machines to point to the correct cross_sections.xml file
# ENDF/B-VIII.0 cross sections on local machine
#materials.cross_sections = "/mnt/d/endfb80_hdf5/cross_sections.xml"
# ENDF/B-VIII.0 cross sections on cluster
#materials.cross_sections = "/home/myu233/nuclear_data/endfb80_hdf5/cross_sections.xml"

materials.export_to_xml("./")

###########################Tally Definition#############################

tallies_file = openmc.Tallies()

log_energy_filter = openmc.EnergyFilter(np.logspace(-4, 7, 1000))
energy_filter = openmc.EnergyFilter([0., 0.5, 1.0e6, 20.0e6])
all_cell_id = [c for c in c.keys()]
all_cell_filter = openmc.CellFilter(all_cell_id)
# Full mesh tally covering the whole device for both thermal and fast flux
mesh = openmc.RegularMesh(mesh_id=1)
mesh.dimension = [275, 275, 160]
mesh.lower_left = [-275, -275, 0]
mesh.width = [2, 2, 4]
full_mesh_filter = openmc.MeshFilter(mesh)

# Mesh tally covering a quarter of the device, focused on the breeder
breeder_mesh = openmc.RegularMesh(mesh_id=2)
breeder_mesh.dimension = [125, 125, 120]
breeder_mesh.lower_left = [-200, -200, 0]
breeder_mesh.width = [4, 4, 4]
breeder_mesh_filter = openmc.MeshFilter(breeder_mesh)

coil_filter = openmc.CellFilter([c[2001].id])
# Mesh surface tally for neutron current
mesh_surface = openmc.MeshSurfaceFilter(mesh)
total_current = openmc.Tally(1,name='total neutron current')
total_current.filters = [mesh_surface]
total_current.scores = ['current']
tallies_file.append(total_current)

fast_current = openmc.Tally(2,name='fast neutron current')
fast_current.filters = [mesh_surface, openmc.EnergyFilter([1e5, 20e6])]
fast_current.scores = ['current']
tallies_file.append(fast_current)

photon_flux = openmc.Tally(3,name='photon flux')
photon_flux.filters = [full_mesh_filter, openmc.ParticleFilter('photon')]
photon_flux.scores = ['flux']
tallies_file.append(photon_flux)

epithermal_flux = openmc.Tally(4,name='epithermal flux')
epithermal_flux.filters = [full_mesh_filter, openmc.EnergyFilter([0.5, 1.0e5])]
epithermal_flux.scores = ['flux']
tallies_file.append(epithermal_flux)

fast_flux = openmc.Tally(5,name='fast flux')
fast_flux.filters = [full_mesh_filter, openmc.EnergyFilter([1.0e5 ,20.0e6])]
fast_flux.scores = ['flux']
tallies_file.append(fast_flux)

strict_fast_flux = openmc.Tally(19,name='strict fast flux')
strict_fast_flux.filters = [full_mesh_filter, openmc.EnergyFilter([1.0e6 ,20.0e6])]
strict_fast_flux.scores = ['flux']
tallies_file.append(strict_fast_flux)

multiplying_flux = openmc.Tally(18,name='Pb multiplying flux')
multiplying_flux.filters = [full_mesh_filter, openmc.EnergyFilter([5e6 ,20.0e6])]
multiplying_flux.scores = ['flux']
tallies_file.append(multiplying_flux)

heat_load = openmc.Tally(6,name='neutron heat load')
heat_load.filters = [full_mesh_filter]
heat_load.scores = ['heating']
tallies_file.append(heat_load)

local_heat_load = openmc.Tally(7,name='neutron local heat load')
local_heat_load.filters = [full_mesh_filter]
local_heat_load.scores = ['heating-local']
#tallies_file.append(local_heat_load)

damage_energy = openmc.Tally(8,name='neutron damage_energy')
damage_energy.filters = [full_mesh_filter]
damage_energy.scores = ['damage-energy']
tallies_file.append(damage_energy)

radiative_capture = openmc.Tally(9,name='neutron radiative capture')
radiative_capture.filters = [full_mesh_filter]
radiative_capture.scores = ['(n,gamma)']
#tallies_file.append(radiative_capture)

absorption = openmc.Tally(10,name='neutron absorption')
absorption.filters = [full_mesh_filter]
absorption.scores = ['absorption']
tallies_file.append(absorption)

avg_coil_flux = openmc.Tally(11,name='Average neutron flux')
avg_coil_flux.filters = [coil_filter, log_energy_filter]
avg_coil_flux.scores = ['flux']
tallies_file.append(avg_coil_flux)

breeder_reaction = openmc.Tally(12,name='Breeder Li-6(n,alpha)T reaction')
breeder_reaction.filters = [openmc.CellFilter(c[6000].id)]
breeder_reaction.scores = ['(n,a)', '(n,Xt)']
tallies_file.append(breeder_reaction)

multiplier_reaction = openmc.Tally(13,name='Breeder Pb(n,2n) reaction')
multiplier_reaction.filters = [openmc.CellFilter(c[6000].id)]
multiplier_reaction.scores = ['(n,2n)', '(n,3n)']
tallies_file.append(multiplier_reaction)

breeder_mesh = openmc.Tally(14,name='Breeder mesh')
breeder_mesh.filters = [full_mesh_filter]
breeder_mesh.scores = ['(n,Xt)']
tallies_file.append(breeder_mesh)

multiplier_mesh = openmc.Tally(15,name='Multiplier mesh')
multiplier_mesh.filters = [full_mesh_filter]
multiplier_mesh.scores = ['(n,2n)', '(n,3n)']
tallies_file.append(multiplier_mesh)

helium_mesh = openmc.Tally(16, name='helium production mesh')
helium_mesh.filters = [full_mesh_filter]
helium_mesh.scores = ['He4-production']
tallies_file.append(helium_mesh)

inverse_velocity_mesh = openmc.Tally(17, name='inverse-velocity mesh')
inverse_velocity_mesh.filters = [full_mesh_filter, openmc.ParticleFilter('neutron')]
inverse_velocity_mesh.scores = ['inverse-velocity']
#tallies_file.append(inverse_velocity_mesh)

all_spectrum = openmc.Tally(20, name="neutron spectrum all cell")
all_spectrum.filters = [all_cell_filter, openmc.ParticleFilter('neutron'), log_energy_filter]
all_spectrum.scores = ['flux']
tallies_file.append(all_spectrum)

central_dpa = openmc.Tally(21, name="damage energy for test region around central cylinder")
central_dpa.filters = [openmc.CellFilter([c[5201].id])]
central_dpa.scores = ['damage-energy']
tallies_file.append(central_dpa)
"""
expanding_dpa = openmc.Tally(22, name="damage energy for test region around expanding chamfer")
expanding_dpa.filters = [openmc.CellFilter([c[5203].id])]
expanding_dpa.scores = ['damage-energy']
tallies_file.append(expanding_dpa)
"""
all_gamma_spectrum = openmc.Tally(23, name="neutron spectrum all cell")
all_gamma_spectrum.filters = [all_cell_filter, openmc.ParticleFilter('photon'), log_energy_filter]
all_gamma_spectrum.scores = ['flux']
tallies_file.append(all_gamma_spectrum)

tallies_file.export_to_xml("./")

geometry = openmc.Geometry(root)
geometry.export_to_xml(working_directory)
geometry.export_to_xml('./')

chamber_geometry_plot = p.slice_plot(basis='yz', 
                                   origin=(0, 0, 250), 
                                   width=(500, 500), 
                                   cwd='./slice')
chamber_geometry_plot.export_to_xml("./")

# Plot geometry
#openmc.plot_geometry(openmc_exec='/software/myu233/openmc/build/bin/openmc')
#openmc.plot_geometry()
#r_bounds, z_bounds = a.plasma_boundary()
#p.plot_geometry(r_bounds, z_bounds)
# Plot geometry in line
#openmc.plot_inline(chamber_geometry_plot)
# Run locally
openmc.run(threads=16, openmc_exec="/usr/local/bin/openmc", geometry_debug=False)
# %%