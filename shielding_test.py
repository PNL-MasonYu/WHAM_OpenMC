# %%
import openmc, openmc.lib
import numpy as np

import material as m
import plotting as p
import os, shutil
import openmc_weight_window_generator


class Model:
    def __init__(self, thickness, material, working_directory = "./shielding", ww_bias = None) -> None:
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
        #######################Simulation Settings############################
        settings = openmc.Settings()

        settings.run_mode = 'fixed source'
        beam_source = openmc.Source()
        beam_source.particle = 'neutron'
        
        beam_source.energy = openmc.stats.Normal(14.1e6, 0.05e6)
        #beam_source.space = openmc.stats.Point((0,0,0))
        uniform_r = openmc.stats.PowerLaw(0, 5, 2)
        uniform_phi = openmc.stats.Uniform(0, 2*np.pi)
        discrete_z = openmc.stats.Discrete(0, 1)
        discrete_mu = openmc.stats.Discrete(1, 1)
        beam_source.angle = openmc.stats.Monodirectional([0,0,1])
        #beam_source.angle = openmc.stats.PolarAzimuthal(mu=discrete_mu, phi=uniform_phi)
        beam_source.space = openmc.stats.CylindricalIndependent(uniform_r, uniform_phi, discrete_z)
        settings.source = beam_source

        settings.particles = int(1e5)
        settings.batches = 10
        settings.output = {'tallies': False}
        #settings.max_lost_particles = int(settings.particles / 2e4)
        #settings.verbosity = 7
        #settings.seed = 53713
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

        m.materials.export_to_xml(working_directory)

        ###########################Geometry parameters##########################
        rcc = openmc.model.surface_composite.RightCircularCylinder
        # First create a dictionary containing all regions
        r = {}
        r[1000] = -rcc((0,0,10), height=thickness, radius=50, axis='z')
        r[2000] = -rcc((0,0,10.01+thickness), height=10, radius=50, axis='z')
        outer_bound = openmc.Sphere(0,0,0, 500)
        outer_bound.boundary_type = 'vacuum'
        r[0] = -outer_bound
        r[0] &= ~r[1000] 
        r[0] &= ~r[2000]

        shield_cell = openmc.Cell(cell_id=1, region=r[1000], fill=material)
        target_cell = openmc.Cell(cell_id=2, region=r[2000], fill=m.vacuum)
        universe_cell = openmc.Cell(region=r[0], fill=m.vacuum)

        root_universe = openmc.Universe(cells=(shield_cell, target_cell, universe_cell))
        geometry = openmc.Geometry()
        geometry.root_universe = root_universe
        geometry.export_to_xml(working_directory)
        ###########################Tally Definition#############################

        tallies_file = openmc.Tallies()

        log_energy_filter = openmc.EnergyFilter(np.logspace(-4, 7, 1000))
        energy_filter = openmc.EnergyFilter([0., 0.5, 1.0e6, 20.0e6])
        target_cell_filter = openmc.CellFilter(target_cell)
        shield_cell_filter = openmc.CellFilter(shield_cell)

        # Full mesh tally covering the whole device for both thermal and fast flux
        mesh = openmc.RegularMesh()
        mesh.dimension = [30, 30, 150]
        mesh.lower_left = [-60, -60, 0]
        mesh.width = [4, 4, 1]
        full_mesh_filter = openmc.MeshFilter(mesh)

        photon_flux = openmc.Tally(name='photon flux')
        photon_flux.filters = [full_mesh_filter, openmc.ParticleFilter('photon')]
        photon_flux.scores = ['flux']
        tallies_file.append(photon_flux)

        total_flux = openmc.Tally(tally_id=2, name='total neutron flux')
        total_flux.filters = [full_mesh_filter, openmc.ParticleFilter('neutron')]
        total_flux.scores = ['flux']
        tallies_file.append(total_flux)

        thermal_flux = openmc.Tally(name='thermal flux')
        thermal_flux.filters = [full_mesh_filter, openmc.EnergyFilter([1e-6, 0.5]), openmc.ParticleFilter('neutron')]
        thermal_flux.scores = ['flux']
        tallies_file.append(thermal_flux)

        epithermal_flux = openmc.Tally(name='epithermal flux')
        epithermal_flux.filters = [full_mesh_filter, openmc.EnergyFilter([0.5, 1.0e5]), openmc.ParticleFilter('neutron')]
        epithermal_flux.scores = ['flux']
        tallies_file.append(epithermal_flux)

        fast_flux = openmc.Tally(name='fast flux')
        fast_flux.filters = [full_mesh_filter, openmc.EnergyFilter([1.0e5 ,20.0e6]), openmc.ParticleFilter('neutron')]
        fast_flux.scores = ['flux']
        tallies_file.append(fast_flux)

        heat_load = openmc.Tally(name='neutron heat load')
        heat_load.filters = [full_mesh_filter]
        heat_load.scores = ['heating', 'heating-local']
        tallies_file.append(heat_load)

        gas_production = openmc.Tally(name='helium, hydrogen production, (n,alpha)')
        gas_production.filters = [full_mesh_filter]
        gas_production.scores = ['(n,a)', 'He4-production', 'H1-production']
        tallies_file.append(gas_production)

        gas_production_nuclides = openmc.Tally(name='helium, hydrogen production, (n,alpha)')
        gas_production_nuclides.filters = [full_mesh_filter]
        gas_production_nuclides.scores = ['(n,a)', 'He4-production', 'H1-production']
        gas_production_nuclides.nuclides = ['B10', 'B11', 'W182', 'W183', 'W184', 'W186', 'C12']
        tallies_file.append(gas_production_nuclides)

        damage_energy = openmc.Tally(name='neutron damage_energy')
        damage_energy.filters = [full_mesh_filter]
        damage_energy.scores = ['damage-energy']
        tallies_file.append(damage_energy)

        rad_capture = openmc.Tally(name='radiative capture')
        rad_capture.filters = [full_mesh_filter]
        rad_capture.scores = ['(n,gamma)']
        tallies_file.append(rad_capture)

        shield_reactions = openmc.Tally(name='reactions in shield')
        shield_reactions.filters = [shield_cell_filter]
        shield_reactions.scores = ['(n,gamma)', 'elastic', '(n,2n)', 'absorption', '(n,nc)', '(n,3n)', '(n,a)', 'photoelectric', 'H1-production', 'He4-production']
        tallies_file.append(shield_reactions)

        elastic_scatter = openmc.Tally(name='elastic scattering')
        elastic_scatter.filters = [full_mesh_filter]
        elastic_scatter.scores = ['elastic']
        tallies_file.append(elastic_scatter)

        target_spectrum = openmc.Tally(name="neutron spectrum target")
        target_spectrum.filters = [target_cell_filter, openmc.ParticleFilter('neutron'), log_energy_filter]
        target_spectrum.scores = ['flux']
        tallies_file.append(target_spectrum)

        target_flux = openmc.Tally(name="neutron flux target")
        target_flux.filters = [target_cell_filter, openmc.ParticleFilter('neutron'), energy_filter]
        target_flux.scores = ['flux']
        tallies_file.append(target_flux)

        target_photon_spectrum = openmc.Tally(name="photon spectrum target")
        target_photon_spectrum.filters = [target_cell_filter, openmc.ParticleFilter('photon'), log_energy_filter]
        target_photon_spectrum.scores = ['flux']
        tallies_file.append(target_photon_spectrum)

        target_photon_flux = openmc.Tally(name="photon flux target")
        target_photon_flux.filters = [target_cell_filter, openmc.ParticleFilter('photon'), energy_filter]
        target_photon_flux.scores = ['flux']
        tallies_file.append(target_photon_flux)

        tallies_file.export_to_xml(working_directory)

        chamber_geometry_plot = p.slice_plot(basis='yz', 
                                   origin=(0, 0, 75), 
                                   width=(120, 150), 
                                   pixels=(1200, 1500),
                                   cwd='./background')
        chamber_geometry_plot.export_to_xml(working_directory)
        ###########################Weight Window Biasing#############################
        if not self.ww_bias == None:
            if os.path.exists(self.ww_bias):
                sp_file = openmc.StatePoint(self.ww_bias)
                weight_windows = sp_file.generate_wws(tally=total_flux, rel_err_tol=0.7)
                settings.weight_windows = weight_windows
                settings.max_splits = int(10000)
            else:
                # Run OpenMC once to generate a statepoint for that thickness if the WW statepoint for that thickness doesn't exist
                settings.export_to_xml(working_directory)
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
                settings.export_to_xml(working_directory)
                # Run again to get the biased result
                self.run_model()

        settings.export_to_xml(working_directory)
        #settings.export_to_xml("./")

    def plot_geometry(self):
        openmc.plot_geometry(cwd=self.cwd)

    def run_model(self, plot_only=False):
        self.plot_geometry()
        if not plot_only:
            openmc.run(threads=16, openmc_exec="/usr/local/bin/openmc", geometry_debug=False, cwd=self.cwd)
            print("finished " + self.cwd)
        
   
def run_sweeps():
    for t in np.arange(90, 5, step=-5):
        
        ww_bias_dir = "shielding_plots/t-" + str(int(t)) + "_statepoint.10.h5"
        """
        w_model = Model(thickness=t, material=m.tungsten, working_directory="./shielding/W/" + str(int(t)), ww_bias=ww_bias_dir)
        w_model.run_model(plot_only=False)

        
        wb_model = Model(thickness=t, material=m.tungsten_boride, working_directory="./shielding/WB/" + str(int(t)), ww_bias=ww_bias_dir)
        wb_model.run_model(plot_only=False)

        wb2_model = Model(thickness=t, material=m.WB2, working_directory="./shielding/WB2/" + str(int(t)), ww_bias=ww_bias_dir)
        wb2_model.run_model(plot_only=False)

        wc_model = Model(thickness=t, material=m.tungsten_carbide, working_directory="./shielding/WC/" + str(int(t)), ww_bias=ww_bias_dir)
        wc_model.run_model(plot_only=False)

        TiH2_model = Model(thickness=t, material=m.TiH2, working_directory="./shielding/TiH2/" + str(int(t)), ww_bias=ww_bias_dir)
        TiH2_model.run_model(plot_only=False)

        MgO_HfH2_model = Model(thickness=t, material=m.MgO_HfH2, working_directory="./shielding/MgO_HfH2/" + str(int(t)), ww_bias=ww_bias_dir)
        MgO_HfH2_model.run_model(plot_only=False)

        Fe_HfH2_WB2_model = Model(thickness=t, material=m.Fe_HfH2_WB2, working_directory="./shielding/Fe_HfH2_WB2/" + str(int(t)), ww_bias=ww_bias_dir)
        Fe_HfH2_WB2_model.run_model(plot_only=False)

        Ti_HfH2_model = Model(thickness=t, material=m.Ti_HfH2, working_directory="./shielding/Ti_HfH2/" + str(int(t)), ww_bias=ww_bias_dir)
        Ti_HfH2_model.run_model(plot_only=False)
        
        stainless_304 = Model(thickness=t, material=m.stainless, working_directory="./shielding/stainless_304/" + str(int(t)), ww_bias=None)
        stainless_304.run_model(plot_only=False)

        H2O = Model(thickness=t, material=m.water, working_directory="./shielding/H2O/" + str(int(t)), ww_bias=None)
        H2O.run_model(plot_only=False)

        HfH2 = Model(thickness=t, material=m.HfH2, working_directory="./shielding/HfH2/" + str(int(t)), ww_bias=ww_bias_dir)
        HfH2.run_model(plot_only=False)
        
        B4C = Model(thickness=t, material=m.B4C, working_directory="./shielding/B4C/" + str(int(t)), ww_bias=None)
        B4C.run_model(plot_only=False)
        
        ZrH2 = Model(thickness=t, material=m.zirconium_hydride, working_directory="./shielding/ZrH2/" + str(int(t)), ww_bias=ww_bias_dir)
        ZrH2.run_model(plot_only=False)
        
        pb_model = Model(thickness=t, material=m.lead, working_directory="./shielding/Pb/" + str(int(t)), ww_bias=ww_bias_dir)
        pb_model.run_model(plot_only=False)

        li_model = Model(thickness=t, material=m.lithium, working_directory="./shielding/Li/" + str(int(t)), ww_bias=ww_bias_dir)
        li_model.run_model(plot_only=False)

        PbLi_model = Model(thickness=t, material=m.LiPb_breeder, working_directory="./shielding/PbLi_natural/" + str(int(t)), ww_bias=ww_bias_dir)
        PbLi_model.run_model(plot_only=False)
        """

        W2B5_model = Model(thickness=t, material=m.w2b5, working_directory="./shielding/W2B5/" + str(int(t)), ww_bias=ww_bias_dir)
        W2B5_model.run_model(plot_only=False)

        crispy_model = Model(thickness=t, material=m.crispy, working_directory="./shielding/B4C_epoxy_mix/" + str(int(t)), ww_bias=ww_bias_dir)
        crispy_model.run_model(plot_only=False)

        
run_sweeps()
"""
ww_bias_dir = "shielding_plots/t-" + str(int(90)) + "_statepoint.10.h5"
ZrH2 = Model(thickness=90, material=m.zirconium_hydride, working_directory="./shielding/ZrH2/" + str(int(90)), ww_bias=ww_bias_dir)
ZrH2.run_model(plot_only=False)
w_model = Model(thickness=90, material=m.tungsten, working_directory="./shielding/W/" + str(int(90)), ww_bias=ww_bias_dir)
w_model.run_model(plot_only=False)
"""
# %%