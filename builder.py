# %%
import openmc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


from material import materials
from boolean import c, root_cells, root
from source import vns_sources
import plotting as p

working_directory = "./test_run"

#######################Simulation Settings###########################
settings = openmc.Settings()

settings.run_mode = 'fixed source'
settings.source = vns_sources
# Alternatively use the worst-case scenario source
"""
worst_source = openmc.Source()
z_uniform = openmc.stats.Uniform(85, 90)
r_uniform = openmc.stats.Uniform(0, 10)
phi_uniform = openmc.stats.Uniform(0, 2*np.pi)
worst_source.space = openmc.stats.CylindricalIndependent(r_uniform, phi_uniform, z_uniform)
worst_source.energy = openmc.stats.Normal(14.1e6, 0.04e6)
worst_source.angle = openmc.stats.Isotropic()
settings.source = worst_source
"""
settings.particles = int(5e6)
settings.batches = 100
settings.output = {'tallies': False}
#settings.max_lost_particles = int(settings.particles / 2e4)
settings.verbosity = 7
#settings.seed = 53713
settings.survival_bias = True
settings.photon_transport = True
settings.export_to_xml(working_directory)
settings.export_to_xml("./")

# ENDF/B-VIII.0 cross sections
# Likely will have to modify this on different machines to point to the correct cross_sections.xml file
materials.cross_sections = "/mnt/d/endfb80_hdf5/cross_sections.xml"

materials.export_to_xml("./")

###########################Tally Definition#############################

tallies_file = openmc.Tallies()

log_energy_filter = openmc.EnergyFilter(np.logspace(-4, 7, 1000))
energy_filter = openmc.EnergyFilter([0., 0.5, 1.0e6, 20.0e6])
# Full mesh tally covering the whole irradiator for both thermal and fast flux
mesh = openmc.RegularMesh(mesh_id=1)
mesh.dimension = [200, 200, 200]
mesh.lower_left = [-200, -200, 0]
mesh.width = [2, 2, 2]
full_mesh_filter = openmc.MeshFilter(mesh)

coil_filter = openmc.CellFilter([c[2001].id])

# Mesh surface tally for neutron current
mesh_surface = openmc.MeshSurfaceFilter(mesh)
total_current = openmc.Tally(name='total neutron current')
total_current.filters = [mesh_surface]
total_current.scores = ['current']
tallies_file.append(total_current)

fast_current = openmc.Tally(name='fast neutron current')
fast_current.filters = [mesh_surface, openmc.EnergyFilter([1e6, 20e6])]
fast_current.scores = ['current']
tallies_file.append(fast_current)

thermal_flux = openmc.Tally(name='thermal flux')
thermal_flux.filters = [full_mesh_filter, openmc.EnergyFilter([0., 0.5])]
thermal_flux.scores = ['flux']
#tallies_file.append(thermal_flux)

epithermal_flux = openmc.Tally(name='epithermal flux')
epithermal_flux.filters = [full_mesh_filter, openmc.EnergyFilter([0.5, 1.0e5])]
epithermal_flux.scores = ['flux']
tallies_file.append(epithermal_flux)

fast_flux = openmc.Tally(name='fast flux')
fast_flux.filters = [full_mesh_filter, openmc.EnergyFilter([1.0e5 ,20.0e6])]
fast_flux.scores = ['flux']
tallies_file.append(fast_flux)

heat_load = openmc.Tally(name='neutron heat load')
heat_load.filters = [full_mesh_filter]
heat_load.scores = ['heating']
tallies_file.append(heat_load)

local_heat_load = openmc.Tally(name='neutron local heat load')
local_heat_load.filters = [full_mesh_filter]
local_heat_load.scores = ['heating-local']
tallies_file.append(local_heat_load)

damage_energy = openmc.Tally(name='neutron damage_energy')
damage_energy.filters = [full_mesh_filter]
damage_energy.scores = ['damage-energy']
tallies_file.append(damage_energy)

radiative_capture = openmc.Tally(name='neutron radiative capture')
radiative_capture.filters = [full_mesh_filter]
radiative_capture.scores = ['(n,gamma)']
tallies_file.append(radiative_capture)

absorption = openmc.Tally(name='neutron absorption')
absorption.filters = [full_mesh_filter]
absorption.scores = ['absorption']
tallies_file.append(absorption)

avg_coil_flux = openmc.Tally(name='Average neutron flux')
avg_coil_flux.filters = [coil_filter, log_energy_filter]
avg_coil_flux.scores = ['flux']
tallies_file.append(avg_coil_flux)

tallies_file.export_to_xml("./")

geometry = openmc.Geometry(root)
geometry.export_to_xml(working_directory)
geometry.export_to_xml('./')

chamber_geometry_plot = p.slice_plot(basis='yz', 
                                   origin=(0, 0, 200), 
                                   width=(400, 400), 
                                   cwd='./slice')
chamber_geometry_plot.export_to_xml("./")
openmc.plot_inline(chamber_geometry_plot)
#openmc.run(threads=16, openmc_exec="/usr/local/bin/openmc")
# %%