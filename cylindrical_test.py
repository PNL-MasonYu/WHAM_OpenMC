# %%
import openmc, openmc.lib
import numpy as np

import material as m
import plotting as p
import os, shutil
import openmc_weight_window_generator


class Model:
    def __init__(self, thickness, material, working_directory = "/mnt/d/WHAM_OpenMC/cylindrical_test", ww_bias = None) -> None:
        # list containing the thicknesses and material definitions of each layer of the shield
        # First layer thickness is the inner radius
        self.thickness = thickness
        self.material = material
        self.cwd = working_directory
        self.ww_bias = ww_bias
        # clear the cwd first
        try:
            shutil.rmtree(self.cwd)
        except OSError as error:
            print(error)
        try:
            os.mkdir(working_directory)
            print("created " + working_directory)
        except OSError as error:
            print(error)

    def build_model(self):
        #######################Simulation Settings############################
        settings = openmc.Settings()

        settings.run_mode = 'fixed source'
        beam_source = openmc.Source()
        beam_source.particle = 'neutron'
        
        beam_source.energy = openmc.stats.Normal(14.1e6, 0.05e6)
        #beam_source.space = openmc.stats.Point((0,0,0))
        uniform_r = openmc.stats.PowerLaw(0, 25, 2)
        uniform_phi = openmc.stats.Uniform(0, 2*np.pi)
        discrete_z = openmc.stats.Discrete(0, 10)
        discrete_mu = openmc.stats.Discrete(1, 1)
        beam_source.angle = openmc.stats.Isotropic()
        #beam_source.angle = openmc.stats.PolarAzimuthal(mu=discrete_mu, phi=uniform_phi)
        beam_source.space = openmc.stats.CylindricalIndependent(uniform_r, uniform_phi, discrete_z)
        settings.source = beam_source

        settings.particles = int(1e5)
        settings.batches = 10
        settings.output = {'tallies': False}
        #settings.max_lost_particles = int(settings.particles / 2e4)
        #settings.verbosity = 7
        settings.seed = 53713
        #settings.track = [(1, 1, 95007)]
        #settings.trace = (1, 1, 95007)
        #settings.survival_bias = True
        #settings.cutoff = {"weight": 0.3,  # value needs to be between 0 and 1
        #                   "weight_avg": 0.9,  # value needs to be between 0 and 1
        #                   }
        #settings.cutoff = {'energy_photon' : 1e3}
        settings.photon_transport = True
        
        

        # Set OPENMC_CROSS_SECTIONS environment variable to the path to cross_sections.xml, or
        # modify this on different machines to point to the correct cross_sections.xml file
        # ENDF/B-VIII.0 cross sections on local machine
        #materials.cross_sections = "/mnt/g/endfb-viii.0-hdf5/cross_sections.xml"
        # ENDF/B-VIII.0 cross sections on cluster
        #materials.cross_sections = "/home/myu233/nuclear_data/endfb80_hdf5/cross_sections.xml"

        m.materials.export_to_xml(self.cwd)

        ###########################Geometry parameters##########################
        rcc = openmc.model.surface_composite.RightCircularCylinder
        # First create a dictionary containing all regions
        r = {}
        p1 = openmc.ZPlane(0, boundary_type='reflective')
        p2 = openmc.ZPlane(10, boundary_type='reflective')
        cylinders = [openmc.ZCylinder(0, 0, self.thickness[0])]
        r[1000] = -p2 & +p1 & -cylinders[0]
        t = self.thickness[0]
        for layer_n in range(1, len(self.thickness)):
            cylinders.append(openmc.ZCylinder(0, 0, t))
            t += self.thickness[layer_n]
            r[1000 + layer_n] = -p2 & +p1 & -cylinders[layer_n] & +cylinders[layer_n-1]

        outer_bound = openmc.Sphere(0,0,0, 500)
        outer_bound.boundary_type = 'vacuum'
        r[0] = -outer_bound & (+p1 & -p2 & +cylinders[-1]) | (+p2 & -outer_bound) | (-p1 & -outer_bound)
        
        all_cells = [openmc.Cell(region=r[0], fill=m.vacuum)]
        for layer_n in range(len(self.thickness)):
            all_cells.append(openmc.Cell(region=r[1000 + layer_n], fill=self.material[layer_n]))

        root_universe = openmc.Universe(cells=all_cells)
        geometry = openmc.Geometry()
        geometry.root_universe = root_universe
        geometry.export_to_xml(self.cwd)
        ###########################Tally Definition#############################

        tallies_file = openmc.Tallies()

        log_energy_filter = openmc.EnergyFilter(np.logspace(-4, 7, 1000))
        energy_filter = openmc.EnergyFilter([0., 0.5, 1.0e6, 20.0e6])
        all_cells_filters = []
        for cell in all_cells:
            all_cells_filters.append(openmc.CellFilter(cell))

        # Full mesh tally covering the whole device for diagnosing problems
        mesh = openmc.RegularMesh()
        mesh.dimension = [30, 30, 10]
        mesh.lower_left = [-60, -60, 0]
        mesh.width = [4, 4, 1]
        full_mesh_filter = openmc.MeshFilter(mesh)

        cyl_mesh = openmc.CylindricalMesh()
        cyl_mesh.origin = [0,0,0]
        cyl_mesh.r_grid = np.linspace(0, 200, 201)
        cyl_mesh.z_grid = [0, 10]
        cyl_mesh_filter = openmc.MeshFilter(cyl_mesh)

        photon_flux = openmc.Tally(name='photon flux')
        photon_flux.filters = [full_mesh_filter, openmc.ParticleFilter('photon')]
        photon_flux.scores = ['flux']
        tallies_file.append(photon_flux)

        total_flux = openmc.Tally(tally_id=2, name='total neutron flux')
        total_flux.filters = [full_mesh_filter, openmc.ParticleFilter('neutron')]
        total_flux.scores = ['flux']
        tallies_file.append(total_flux)

        thermal_flux = openmc.Tally(name='thermal flux')
        thermal_flux.filters = [cyl_mesh_filter, openmc.EnergyFilter([1e-6, 0.5]), openmc.ParticleFilter('neutron')]
        thermal_flux.scores = ['flux']
        tallies_file.append(thermal_flux)

        epithermal_flux = openmc.Tally(name='epithermal flux')
        epithermal_flux.filters = [cyl_mesh_filter, openmc.EnergyFilter([0.5, 1.0e5]), openmc.ParticleFilter('neutron')]
        epithermal_flux.scores = ['flux']
        tallies_file.append(epithermal_flux)

        fast_flux = openmc.Tally(name='fast flux')
        fast_flux.filters = [cyl_mesh_filter, openmc.EnergyFilter([1.0e5 ,20.0e6]), openmc.ParticleFilter('neutron')]
        fast_flux.scores = ['flux']
        tallies_file.append(fast_flux)

        heat_load = openmc.Tally(name='neutron heat load')
        heat_load.filters = [cyl_mesh_filter]
        heat_load.scores = ['heating', 'heating-local']
        tallies_file.append(heat_load)

        gas_production = openmc.Tally(name='helium, hydrogen production, (n,alpha)')
        gas_production.filters = [cyl_mesh_filter]
        gas_production.scores = ['(n,a)', 'He4-production', 'H1-production']
        tallies_file.append(gas_production)

        gas_production_nuclides = openmc.Tally(name='helium, hydrogen production, (n,alpha), for nuclides')
        gas_production_nuclides.filters = [cyl_mesh_filter]
        gas_production_nuclides.scores = ['(n,a)', 'He4-production', 'H1-production']
        gas_production_nuclides.nuclides = ['B10', 'B11', 'W182', 'W183', 'W184', 'W186', 'C12']
        tallies_file.append(gas_production_nuclides)

        damage_energy = openmc.Tally(name='neutron damage_energy')
        damage_energy.filters = [cyl_mesh_filter]
        damage_energy.scores = ['damage-energy']
        tallies_file.append(damage_energy)

        shield_reactions = openmc.Tally(name='reactions in shield')
        shield_reactions.filters = [cyl_mesh_filter]
        shield_reactions.scores = ['(n,gamma)', 'elastic', '(n,2n)', 'absorption', '(n,nc)', '(n,3n)', '(n,a)', 'photoelectric', 'H1-production', 'He4-production']
        tallies_file.append(shield_reactions)

        elastic_scatter = openmc.Tally(name='elastic scattering')
        elastic_scatter.filters = [cyl_mesh_filter, openmc.ParticleFilter('neutron')]
        elastic_scatter.scores = ['elastic']
        tallies_file.append(elastic_scatter)

        target_spectrum = openmc.Tally(name="neutron spectrum target")
        target_spectrum.filters = all_cells_filters + [openmc.ParticleFilter('neutron'), log_energy_filter]
        target_spectrum.scores = ['flux']
        tallies_file.append(target_spectrum)

        target_flux = openmc.Tally(name="neutron flux target")
        target_flux.filters = all_cells_filters + [openmc.ParticleFilter('neutron'), energy_filter]
        target_flux.scores = ['flux']
        tallies_file.append(target_flux)

        target_photon_spectrum = openmc.Tally(name="photon spectrum target")
        target_photon_spectrum.filters = all_cells_filters + [openmc.ParticleFilter('photon'), log_energy_filter]
        target_photon_spectrum.scores = ['flux']
        tallies_file.append(target_photon_spectrum)

        target_photon_flux = openmc.Tally(name="photon flux target")
        target_photon_flux.filters = all_cells_filters + [openmc.ParticleFilter('photon'), energy_filter]
        target_photon_flux.scores = ['flux']
        tallies_file.append(target_photon_flux)

        tallies_file.export_to_xml(self.cwd)

        chamber_geometry_plot = p.slice_plot(basis='yz', 
                                   origin=(0, 0, 0), 
                                   width=(500, 500), 
                                   pixels=(1500, 1500),
                                   cwd='./background')
        chamber_geometry_plot.export_to_xml(self.cwd)
        ###########################Weight Window Biasing#############################
        if not self.ww_bias == None:
            if os.path.exists(self.ww_bias):
                sp_file = openmc.StatePoint(self.ww_bias)
                weight_windows = sp_file.generate_wws(tally=total_flux, rel_err_tol=0.7)
                settings.weight_windows = weight_windows
                settings.max_splits = int(10000)
            else:
                # Run OpenMC once to generate a statepoint for that thickness if the WW statepoint for that thickness doesn't exist
                settings.export_to_xml(self.cwd)
                self.run_model()
                shutil.copy(self.cwd + "/statepoint.10.h5", self.ww_bias)
                # clear the output statepoint
                try:
                    shutil.rmtree(self.cwd + "/statepoint.10.h5")
                    shutil.rmtree(self.cwd + "/settings.xml")
                except OSError as error:
                    print(error)
                sp_file = openmc.StatePoint(self.ww_bias)
                weight_windows = sp_file.generate_wws(tally=total_flux, rel_err_tol=0.7)
                settings.weight_windows = weight_windows
                settings.max_splits = int(10000)
                settings.export_to_xml(self.cwd)
                # Run again to get the biased result
                self.run_model()

        settings.export_to_xml(self.cwd)
        #settings.export_to_xml("./")

    def plot_geometry(self):
        openmc.plot_geometry(cwd=self.cwd, path_input=self.cwd)


    def run_model(self, plot_only=False):
        self.plot_geometry()
        if not plot_only:
            openmc.run(threads=16, openmc_exec="/usr/local/bin/openmc", geometry_debug=False, cwd=self.cwd)
            print("finished " + self.cwd)


test_model = Model([50, 10, 50, 10, 10], [m.vacuum, m.he_cooled_rafm, m.LiPb_breeder, m.tungsten])
test_model.build_model()
test_model.run_model()