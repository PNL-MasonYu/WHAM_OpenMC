
import openmc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cql3d_analysis_toolbox as cql3d
 
import plotting as p

def plasma_boundary(cql3d_file="WHAM_VNS_gen3_large_50keV_NBI_HFS_2p9Tres_2x2MWrf_5keV.nc"):
    """
    returns the outer plasma boundary in the outermost r-z grid of the plasma
    """
    ndwarmz, ndfz, ndtotz, solrz, solzz = cql3d.ion_dens("WHAM_VNS_gen3_large_50keV_NBI_HFS_2p9Tres_2x2MWrf_5keV.nc", False)
    r_bounds = solrz[-1] * 100
    z_bounds = solzz[-1] * 100
    return r_bounds, z_bounds

sp = openmc.StatePoint("statepoint.50.h5")
r_bounds, z_bounds = plasma_boundary()

thermalflux_tally = sp.get_tally(name='thermal flux')
epithermalflux_tally = sp.get_tally(name='epithermal flux')
fastflux_tally = sp.get_tally(name='fast flux')
heat_tally = sp.get_tally(name="neutron heat load")
thermal_flux = thermalflux_tally.get_slice(scores=['flux'])
epithermal_flux = epithermalflux_tally.get_slice(scores=['flux'])
fast_flux = fastflux_tally.get_slice(scores=['flux'])
heat = heat_tally.get_slice(scores=['heating'])

mesh_shape = (200, 200, 200)
thermal_flux.mean.shape = mesh_shape
epithermal_flux.mean.shape = mesh_shape
fast_flux.mean.shape = mesh_shape
heat.mean.shape = mesh_shape
normalized_heat = np.add(heat.mean, 1e-10)
extent=(-200, 200, 0, 400)

background_plot = p.slice_plot(basis="xz", origin=(0, 0, 200), cwd='./background',
                               width=(400, 400), color=p.material_color)
background_plot.export_to_xml()
openmc.plot_inline(background_plot)
background_image = plt.imread('./background.png')
# %%
fig1 = plt.figure(num=1, figsize=(10, 10))
plt.imshow(background_image, extent=extent)
im = plt.imshow(np.multiply(thermal_flux.mean[:, 100, :], 4.2e17/2), cmap='Spectral_r', origin='lower', 
                alpha=.9, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=0.1, vmax=1e6))
CS = plt.contour(np.multiply(thermal_flux.mean[:, 100, :], 4.2e17/2), np.logspace(-1, 6, 8), origin="lower",
                 extent=extent, cmap='flag', linewidths=0.5)
plt.colorbar(im, label='Thermal neutron flux $[n/cm^2-s]$', orientation='vertical',
             shrink=0.8, format='%0.0e')
plt.clabel(CS, fmt='%0.0e')
plt.title('DT Thermal Flux (<0.5 eV)\n 4.2e17 n/s Source Rate\n Tungsten shield')
plt.xlabel('y (cm)')
plt.ylabel('z (cm)')
fig1.savefig('./plots/DT strict thermal flux yz')
# %%
fig2 = plt.figure(num=2, figsize=(15, 15))
plt.imshow(background_image, extent=extent)
im = plt.imshow(np.multiply(epithermal_flux.mean[:, 100, :], 4.2e17/2), cmap='Spectral_r', origin='lower', 
                alpha=.85, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e10, vmax=1e16))
CS = plt.contour(np.multiply(epithermal_flux.mean[:, 100, :], 4.2e17/2), np.logspace(10, 16, 16), origin="lower",
                 extent=extent, cmap='flag', linewidths=0.5)
plt.title('DT Epithermal Flux (0.5 eV - 100 keV)\n 4.2e17 n/s Source Rate\n Tungsten shield')
plt.colorbar(im, label='Epithermal neutron flux $[n/cm^2-s]$', orientation='vertical',
             shrink=0.8, format='%0.0e')
plt.clabel(CS, fmt='%0.0e')
plt.xlabel('y (cm)')
plt.ylabel('z (cm)')
fig2.savefig('./plots/DT strict epithermal flux yz')

# %%
fig3 = plt.figure(num=3, figsize=(15, 15))
plt.plot(r_bounds, z_bounds, "g-")
plt.plot(-r_bounds, z_bounds, "g-")
plt.imshow(background_image, extent=extent)
im = plt.imshow(np.multiply(fast_flux.mean[:, 100, :], 4.2e17/2), cmap='Spectral_r', origin='lower', 
                alpha=.85, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e10, vmax=1e16))
CS = plt.contour(np.multiply(fast_flux.mean[:, 100, :], 4.2e17/2), np.logspace(10, 16, 12), origin="lower",
                 extent=extent, cmap='flag', linewidths=0.5)

plt.title('DT Fast Flux (>100 keV)\n 4.2e17 n/s Source Rate\n Tungsten shield')
plt.colorbar(im, label='Fast neutron flux $[n/cm^2-s]$', orientation='vertical',
             shrink=0.8, format='%0.0e')
plt.clabel(CS, fmt='%0.0e')
plt.xlabel('y (cm)')
plt.ylabel('z (cm)')
fig3.savefig('./plots/DT strict fast flux yz')
# %%
fig4 = plt.figure(num=4, figsize=(15, 15))
plt.imshow(background_image, extent=extent)
im = plt.imshow(np.multiply(normalized_heat[:, 100, :], 4.2e17/2*1.602e-19/8), cmap='Spectral_r', origin='lower', 
                alpha=0.85, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e-5, vmax=1e1))
CS = plt.contour(np.multiply(normalized_heat[:, 100, :], 4.2e17/2*1.602e-19/8), np.logspace(-3, 1, 10), origin="lower",
                 extent=extent, cmap='flag', linewidths=0.5)
plt.title('DT Neutron Nuclear Heating with Gamma\n 4.2e17 n/s Source Rate\n Tungsten shield')
plt.colorbar(im, label='Neutron local energy deposition $[W/cm^3]$', orientation='vertical',
             shrink=0.8, format='%0.0e')
plt.clabel(CS, fmt='%0.0e')
plt.xlabel('y (cm)')
plt.ylabel('z (cm)')
fig4.savefig('./plots/DT heating with photon yz')

# %%
