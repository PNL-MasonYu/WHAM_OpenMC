# %%
from matplotlib.pyplot import flag
import openmc
import numpy as np
from cql3d_analysis_toolbox import ion_dens, axial_neutron_flux


ndwarmz, ndfz, ndtotz, solrz, solzz = ion_dens("WHAM_VNS_gen3_large_50keV_NBI_HFS_2p9Tres_2x2MWrf_5keV.nc", False)
flux_neutron_f, z_fus = axial_neutron_flux("WHAM_VNS_gen3_large_50keV_NBI_HFS_2p9Tres_2x2MWrf_5keV.nc", False)
#print(cql3d.fusion_rx_rate("WHAM_VNS_gen3_large_50keV_NBI_HFS_2p9Tres_2x2MWrf_5keV.nc",True))
#print(flux_neutron_f)
#print(solrz)
# %%
vns_sources = []
source_distribution = []
coordinates_distribution = []
midpt_index = int(len(flux_neutron_f) / 2)
strength_sum = 0
for n in range(ndfz.shape[1]):
    strength_sum += flux_neutron_f[midpt_index+n]

for n in range(ndfz.shape[1]):
    fusion_source = openmc.Source()
    fusion_source.angle = openmc.stats.Isotropic()
    #fusion_source.energy = openmc.stats.Normal(2.45e6, 0.04e6)
    fusion_source.energy = openmc.stats.Normal(14.1e6, 0.05e6)
    r_tabular = openmc.stats.Tabular(solrz[:, n]*100, ndfz[:, n])
    radial_distribution = ndfz[:, n] / np.sum(ndfz[:, n])
    z_thickness = solzz[0, 1]*100 - solzz[0, 0]*100
    z_uniform = openmc.stats.Uniform(z_thickness*n, z_thickness*(n+1))
    phi_uniform = openmc.stats.Uniform(0, 2*np.pi)
    fusion_source.space = openmc.stats.CylindricalIndependent(r_tabular, phi_uniform ,z_uniform)
    # Normalization so the total source strength remains 1 when integrated across all space
    fusion_source.strength = flux_neutron_f[midpt_index+n] / strength_sum

    radial_distribution = np.multiply(radial_distribution, fusion_source.strength)
    source_distribution.append(radial_distribution)

    vns_sources.append(fusion_source)
# %%

worst_source = openmc.Source()
z_uniform = openmc.stats.Uniform(85, 90)
r_uniform = openmc.stats.Uniform(0, 10)
phi_uniform = openmc.stats.Uniform(0, 2*np.pi)
worst_source.space = openmc.stats.CylindricalIndependent(r_uniform, phi_uniform, z_uniform)
worst_source.energy = openmc.stats.Normal(14.1e6, 0.04e6)
worst_source.angle = openmc.stats.Isotropic()

"""
Export the spatial distribution of the source in a csv file
"""
def export_source(path, vns_source):
    source_array = np.array(vns_source)
    #source_array.tofile(path, ",", "%s")
    #solrz.tofile("./data_files/WHAM VNS source radial position.csv", ",", "%s")
    #solzz.tofile("./data_files/WHAM VNS source axial position.csv", ",", "%s")
    np.savetxt(path, source_array, delimiter=",")
    np.savetxt("./data_files/WHAM VNS source radial position.csv", solrz, delimiter=",")
    np.savetxt("./data_files/WHAM VNS source axial position.csv", solzz, delimiter=",")
#export_source("./data_files/WHAM VNS source 4e17.csv", source_distribution)
# %%
