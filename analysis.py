# %%
import openmc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cql3d_analysis_toolbox as cql3d
import os
 
import plotting as p

def plasma_boundary(cql3d_file="WHAM_VNS_gen3_large_50keV_NBI_HFS_2p9Tres_2x2MWrf_5keV.nc"):
    """
    returns the outer plasma boundary in the outermost r-z grid of the plasma
    """
    ndwarmz, ndfz, ndtotz, solrz, solzz = cql3d.ion_dens("WHAM_VNS_gen3_large_50keV_NBI_HFS_2p9Tres_2x2MWrf_5keV.nc", False)
    r_bounds = solrz[-1] * 100
    z_bounds = solzz[-1] * 100
    return r_bounds, z_bounds

def aggregate_rectangular(result, r_bins=100, dim=1):
    """
    aggegate the cylindrical tally in the r direction
    The z-dimension should be specified in the dim variable
    The center of the tally will be the axis of revolution
    """
    center = result.shape[dim-1]/2
    z_length = result.shape[dim]
    aggregate = np.zeros((z_length, r_bins*2))
    for z in range(z_length):
        slice = result[z, :, :]
        R = []
        W = []
        for x in range(int(center*2)):
            for y in range(int(center*2)):
                r = np.sqrt((x-center)**2 + (y-center)**2)
                R.append(r)
                W.append(slice[x, y])
        hist = np.histogram(R, bins=r_bins, range=(0, int(center*2)), weights=W)[0]
        edges = np.histogram(R, bins=r_bins, range=(0, int(center*2)), weights=W)[1]
        for r in range(r_bins):
            aggregate[z, int(center-r)] = hist[r]
            aggregate[z, int(center+r)] = hist[r]
    return aggregate, edges

def surf_to_grid():
    """
    This converts the dataframe for the mesh surface tally into grids of U, V vectors at each cell
    Returns the U, V vectors on each point in the grid
    """
    

if not os.path.isdir('./plots'):
    os.makedirs('./plots')

sp = openmc.StatePoint("statepoint.50.h5")
r_bounds, z_bounds = plasma_boundary()

#
mesh_shape = (200, 200, 200)
extent=(-200, 200, 0, 400)
background_plot = p.slice_plot(basis="xz", origin=(0, 0, 200), cwd='./background',
                               width=(400, 400), color=p.material_color)
background_plot.export_to_xml()
#openmc.plot(background_plot)
background_image = plt.imread('./background.png')
# %%
def plot_thermal_flux():
    thermalflux_tally = sp.get_tally(name='thermal flux')
    thermal_flux = thermalflux_tally.get_slice(scores=['flux'])
    thermal_flux.mean.shape = mesh_shape
    fig1 = plt.figure(num=1, figsize=(10, 10))
    data = aggregate_rectangular(thermal_flux.mean)[0]
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, 4.2e17*2), cmap='Spectral_r', origin='lower', 
                    alpha=.7, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=0.1, vmax=1e6))
    CS = plt.contour(np.multiply(data, 4.2e17*2), np.logspace(-1, 6, 8), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.colorbar(im, label='Thermal neutron flux $[n/cm^2-s]$', orientation='vertical',
                shrink=0.8, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e')
    plt.title('DT Thermal Flux (<0.5 eV)\n 4.2e17 n/s Source Rate\n Tungsten shield')
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    fig1.savefig('./plots/DT strict thermal flux yz')
    return fig1

def plot_epithermal_flux():
    epithermalflux_tally = sp.get_tally(name='epithermal flux')
    epithermal_flux = epithermalflux_tally.get_slice(scores=['flux'])
    epithermal_flux.mean.shape = mesh_shape
    data = aggregate_rectangular(epithermal_flux.mean)[0]
    fig2 = plt.figure(num=2, figsize=(15, 15))
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, 4.2e17*2), cmap='Spectral_r', origin='lower', 
                    alpha=.7, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e10, vmax=1e16))
    CS = plt.contour(np.multiply(data, 4.2e17*2), np.logspace(10, 17, 16), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('DT Epithermal Flux (0.5 eV - 100 keV)\n 4.2e17 n/s Source Rate\n Tungsten shield')
    plt.colorbar(im, label='Epithermal neutron flux $[n/cm^2-s]$', orientation='vertical',
                shrink=0.8, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e')
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    fig2.savefig('./plots/DT strict epithermal flux yz')
    return fig2

def plot_fast_flux():
    fastflux_tally = sp.get_tally(name='fast flux')
    fast_flux = fastflux_tally.get_slice(scores=['flux'])
    fast_flux.mean.shape = mesh_shape
    fig3 = plt.figure(num=3, figsize=(15, 15))
    data = aggregate_rectangular(fast_flux.mean)[0]
    #print(aggregate_rectangular(fast_flux.mean)[1])
    #data = fast_flux.mean[:, 100, :]
    plt.plot(r_bounds, z_bounds, "g-")
    plt.plot(-r_bounds, z_bounds, "g-")
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, 4.2e17*2), cmap='Spectral_r', origin='lower', 
                    alpha=.7, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e10, vmax=1e16))
    CS = plt.contour(np.multiply(data, 4.2e17*2), np.logspace(10, 17, 12), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('DT Fast Flux (>100 keV)\n 4.2e17 n/s Source Rate\n Tungsten shield')
    plt.colorbar(im, label='Fast neutron flux $[n/cm^2-s]$', orientation='vertical',
                shrink=0.8, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e')
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    fig3.savefig('./plots/DT strict fast flux yz')
    return fig3

def plot_total_heat():
    heat_tally = sp.get_tally(name="neutron heat load")
    heat = heat_tally.get_slice(scores=['heating'])
    heat.mean.shape = mesh_shape
    normalized_heat = np.add(heat.mean, 1e-10)
    fig4 = plt.figure(num=4, figsize=(15, 15))
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(normalized_heat[:, 100, :], 4.2e17*2*1.602e-19/8), cmap='plasma', origin='lower', 
                    alpha=0.85, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e-5, vmax=1e1))
    CS = plt.contour(np.multiply(normalized_heat[:, 100, :], 4.2e17*2*1.602e-19/8), np.logspace(-3, 1, 10), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('DT Neutron Nuclear Heating with Gamma\n 4.2e17 n/s Source Rate\n Tungsten shield')
    plt.colorbar(im, label='Total Nuclear Heating $[W/cm^3]$', orientation='vertical',
                shrink=0.8, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e')
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    fig4.savefig('./plots/DT heating with photon yz')
    return fig4

def plot_local_heat():
    local_heat_tally = sp.get_tally(name="neutron local heat load")
    local_heat = local_heat_tally.get_slice(scores=['heating-local'])
    local_heat.mean.shape = mesh_shape
    normalized_local_heat = np.add(local_heat.mean, 1e-10)
    fig5 = plt.figure(num=5, figsize=(15, 15))
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(normalized_local_heat[:, 100, :], 4.2e17*2*1.602e-19/8), cmap='plasma', origin='lower', 
                    alpha=0.85, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e-5, vmax=1e1))
    CS = plt.contour(np.multiply(normalized_local_heat[:, 100, :], 4.2e17*2*1.602e-19/8), np.logspace(-3, 1, 10), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('DT Neutron Local Heating with Gamma\n 4.2e17 n/s Source Rate\n Tungsten shield')
    plt.colorbar(im, label='Neutron local energy deposition $[W/cm^3]$', orientation='vertical',
                shrink=0.8, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e')
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    fig5.savefig('./plots/DT local heating with photon yz')
    return fig5

def plot_radiative_capture():
    radiative_capture_tally = sp.get_tally(name="neutron radiative capture")
    radiative_capture = radiative_capture_tally.get_slice(scores=['(n,gamma)'])
    radiative_capture.mean.shape = mesh_shape
    radiative_capture = np.add(radiative_capture.mean, 1e-15)
    fig6 = plt.figure(num=6, figsize=(15, 15))
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(radiative_capture[:, 100, :], 4.2e17*2*1.602e-19/8), cmap='viridis', origin='lower', 
                    alpha=0.85, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e-12, vmax=1e-6))
    CS = plt.contour(np.multiply(radiative_capture[:, 100, :], 4.2e17*2*1.602e-19/8), np.logspace(-12, -6, 10), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('DT Neutron Radiative Capture\n 4.2e17 n/s Source Rate\n Tungsten shield')
    plt.colorbar(im, label='Neutron radiative capture $[#/cm^3-s]$', orientation='vertical',
                shrink=0.8, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e')
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    fig6.savefig('./plots/DT radiative capture yz')
    return fig6

def plot_absorption():
    absorption_tally = sp.get_tally(name="neutron absorption")
    absorption = absorption_tally.get_slice(scores=['absorption'])
    absorption.mean.shape = mesh_shape
    absorption = np.add(absorption.mean, 1e-15)
    fig7 = plt.figure(num=7, figsize=(15, 15))
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(absorption[:, 100, :], 4.2e17*2*1.602e-19/8), cmap='viridis', origin='lower', 
                    alpha=0.85, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e-12, vmax=1e-6))
    CS = plt.contour(np.multiply(absorption[:, 100, :], 4.2e17*2*1.602e-19/8), np.logspace(-12, -6, 10), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('DT Neutron Absorption\n 4.2e17 n/s Source Rate\n Tungsten shield')
    plt.colorbar(im, label='Neutron absorption $[#/cm^3-s]$', orientation='vertical',
                shrink=0.8, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e')
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    fig7.savefig('./plots/DT absorption yz')
    return fig7

def plot_coil_flux_energy():
    avg_coil_flux_tally = sp.get_tally(name='Average neutron flux')
    avg_coil_flux = avg_coil_flux_tally.get_slice(scores=['flux'])

    fig8 = plt.figure(num=8, figsize=(15, 15))
    flux = np.multiply(avg_coil_flux.mean, 4.2e17*2)[0]
    print(flux)
    plt.hist(["<0.5 eV", "0.5 eV - 100 keV", ">100 keV"],
            [flux[0], flux[1], flux[2]])
    return fig8

# %%
plot_fast_flux()
#plot_local_heat()
# %%
