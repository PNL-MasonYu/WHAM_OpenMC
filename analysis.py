# %%
import openmc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from cql3d_analysis_toolbox import ion_dens
import os
 
import plotting as p

def plasma_boundary(cql3d_file="WHAM_VNS_gen3_large_50keV_NBI_HFS_2p9Tres_2x2MWrf_5keV.nc"):
    """
    returns the outer plasma boundary in the outermost r-z grid of the plasma
    """
    ndwarmz, ndfz, ndtotz, solrz, solzz = ion_dens("WHAM_VNS_gen3_large_50keV_NBI_HFS_2p9Tres_2x2MWrf_5keV.nc", False)
    r_bounds = solrz[-1] * 100
    z_bounds = solzz[-1] * 100
    return r_bounds, z_bounds

def aggregate_rectangular(result, r_bins=100, dim=0):
    """
    aggegate the cylindrical tally in the r direction
    The z-dimension should be specified in the dim variable
    The center of the tally will be the axis of revolution
    """
    center = result.shape[dim-1]/2
    z_length = result.shape[dim]
    aggregate = np.zeros((z_length, r_bins*2+2))
    edges = np.linspace(0, int(center), num=r_bins)
    #print(z_length)
    for z in range(z_length):
        slice = result[z, :, :]
        #print(slice.shape)
        R = []
        W = {}
        for x in range(result.shape[dim-1]):
            for y in range(result.shape[dim-1]):
                r = np.sqrt((x-center)**2 + (y-center)**2)
                R.append(r)
                W[r] = slice[x, y]
        for i in range(r_bins-1):
            binned_r = [radius for radius in R if (radius <= edges[i+1] and radius >= edges[i])]
            binned_w = [W[radius] for radius in binned_r]
            aggregate[z, int(r_bins-i)] = np.mean(binned_w)
            aggregate[z, int(r_bins+i)] = np.mean(binned_w)
    return aggregate

def surf_to_grid():
    """
    This converts the dataframe for the mesh surface tally into grids of U, V vectors at each cell
    Returns the U, V vectors on each point in the grid
    """

def plot_background():
    background_plot = p.slice_plot(basis="xz", origin=(0, 0, 320), cwd='./background',
                                width=(550, 640), color=p.material_color)
    background_plot.export_to_xml()
    openmc.plot_geometry(background_plot)

if not os.path.isdir('./plots'):
    os.makedirs('./plots')

#plot_background()

sp = openmc.StatePoint("./statepoint.500-W2B5-PbLi.h5", autolink=False)
r_bounds, z_bounds = plasma_boundary()

#
mesh_shape = (160, 275, 275)
extent=(-275, 275, 0, 640)

background_image = plt.imread('./background.ppm')
# %%
def plot_thermal_flux():
    thermalflux_tally = sp.get_tally(name='thermal flux')
    thermal_flux = thermalflux_tally.get_slice(scores=['flux'])
    thermal_flux.mean.shape = mesh_shape
    fig1 = plt.figure(num=1, figsize=(15, 10))
    data = aggregate_rectangular(thermal_flux.mean)
    #data = thermal_flux.mean[:, :, 137]
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, 4.2e17), cmap='Spectral_r', origin='lower', 
                    alpha=.7, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=0.1, vmax=1e6))
    CS = plt.contour(np.multiply(data, 4.2e17), np.logspace(-1, 6, 8), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.colorbar(im, label='Thermal neutron flux $[n/cm^2-s]$', orientation='vertical',
                shrink=0.7, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e')
    plt.title('DT Thermal Flux (<0.5 eV)\n 4.2e17 n/s Source Rate\n tungsten boride W2B5 10% vo water shield')
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    fig1.savefig('./plots/DT strict thermal flux yz')
    return fig1

def plot_epithermal_flux():
    epithermalflux_tally = sp.get_tally(name='epithermal flux')
    epithermal_flux = epithermalflux_tally.get_slice(scores=['flux'])
    epithermal_flux.mean.shape = mesh_shape
    data = aggregate_rectangular(epithermal_flux.mean)
    #data = epithermal_flux.mean[:, :, 137]
    fig2 = plt.figure(num=2, figsize=(15, 10))
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, 4.2e17), cmap='Spectral_r', origin='lower', 
                    alpha=.7, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e10, vmax=1e16))
    CS = plt.contour(np.multiply(data, 4.2e17), np.logspace(10, 17, 16), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('DT Epithermal Flux (0.5 eV - 100 keV)\n 4.2e17 n/s Source Rate\n tungsten boride W2B5 10% vo water shield')
    plt.colorbar(im, label='Epithermal neutron flux $[n/cm^2-s]$', orientation='vertical',
                shrink=0.7, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e')
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    fig2.savefig('./plots/DT strict epithermal flux yz')
    return fig2

def plot_fast_flux():
    fastflux_tally = sp.get_tally(name='fast flux')
    fast_flux = fastflux_tally.get_slice(scores=['flux'])
    fast_flux.mean.shape = mesh_shape
    fig3 = plt.figure(num=3, figsize=(15, 10))
    data = aggregate_rectangular(fast_flux.mean)
    #data = fast_flux.mean[:, :, 137]
    plt.plot(r_bounds, z_bounds, "g-")
    plt.plot(-r_bounds, z_bounds, "g-")
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, 4.2e17), cmap='Spectral_r', origin='lower', 
                    alpha=.7, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e8, vmax=1e15))
    CS = plt.contour(np.multiply(data, 4.2e17), np.logspace(8, 15, 12), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('DT Fast Flux (>100 keV)\n 4.2e17 n/s Source Rate\n tungsten boride W2B5 10% vo water shield')
    plt.colorbar(im, label='Fast neutron flux $[n/cm^2-s]$', orientation='vertical',
                shrink=0.7, format='%0.0e')
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
    data = aggregate_rectangular(normalized_heat)
    #data = normalized_heat.mean[:, :, 137]
    fig4 = plt.figure(num=4, figsize=(15, 10))
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, 4.2e17*1.602e-19/8), cmap='plasma', origin='lower', 
                    alpha=0.85, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e-5, vmax=1e1))
    CS = plt.contour(np.multiply(data, 4.2e17*1.602e-19/8), np.logspace(-5, 1, 10), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('DT Neutron Nuclear Heating with Gamma\n 4.2e17 n/s Source Rate\n tungsten boride W2B5 10% vo water shield')
    plt.colorbar(im, label='Total Nuclear Heating $[W/cm^3]$', orientation='vertical',
                shrink=0.7, format='%0.0e')
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
    data = aggregate_rectangular(normalized_local_heat)
    #data = normalized_local_heat.mean[:, :, 137]
    fig5 = plt.figure(num=5, figsize=(15, 10))
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, 4.2e17*1.602e-19/8), cmap='plasma', origin='lower', 
                    alpha=0.85, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e-5, vmax=1e1))
    CS = plt.contour(np.multiply(data, 4.2e17*1.602e-19/8), np.logspace(-3, 1, 10), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('DT Neutron Local Heating with Gamma\n 4.2e17 n/s Source Rate\n tungsten boride W2B5 10% vo water shield')
    plt.colorbar(im, label='Neutron local energy deposition $[W/cm^3]$', orientation='vertical',
                shrink=0.7, format='%0.0e')
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
    data = aggregate_rectangular(radiative_capture.mean)
    #data = radiative_capture.mean[:, :, 137]
    fig6 = plt.figure(num=6, figsize=(15, 10))
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, 4.2e17/16), cmap='viridis', origin='lower', 
                    alpha=0.85, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e7, vmax=1e13))
    CS = plt.contour(np.multiply(data, 4.2e17/16), np.logspace(7, 13, 14), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('DT Neutron Radiative Capture\n 4.2e17 n/s Source Rate\n tungsten boride W2B5 10% vo water shield')
    plt.colorbar(im, label='Neutron radiative capture $[#/cm^3-s]$', orientation='vertical',
                shrink=0.7, format='%0.0e')
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
    data = aggregate_rectangular(absorption)
    #data = absorption.mean[:, :, 137]
    fig7 = plt.figure(num=7, figsize=(15, 10))
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, 4.2e17/16), cmap='viridis', origin='lower', 
                    alpha=0.85, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e14, vmax=1e19))
    CS = plt.contour(np.multiply(data, 4.2e17/16), np.logspace(14, 19, 14), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('DT Neutron Absorption\n 4.2e17 n/s Source Rate\n tungsten boride W2B5 10% vo water shield')
    plt.colorbar(im, label='Neutron absorption $[#/cm^3-s]$', orientation='vertical',
                shrink=0.7, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e')
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    fig7.savefig('./plots/DT absorption yz')
    return fig7

def plot_coil_flux_energy():
    avg_coil_flux_tally = sp.get_tally(name='Average neutron flux')
    avg_coil_flux = avg_coil_flux_tally.get_slice(scores=['flux'])
    fig8 = plt.figure(num=8, figsize=(15, 10))
    flux = np.multiply(avg_coil_flux.mean, 4.2e17)
    plt.plot(np.logspace(-4, 7, 999), flux[:, 0, 0])
    plt.title('Neutron energy spectrum through coil \n 4.2e17 n/s Source Rate')
    plt.xlabel('Neutron energy (eV)')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Neutron flux (n/cm2-s)')
    return fig8

def get_tbr():
    li6_breeding_tally = sp.get_tally(name='Breeder Li-6(n,alpha)T reaction')
    li6_breeding = li6_breeding_tally.get_slice(scores=['(n,Xt)'])
    breeding_dataframe = li6_breeding.get_pandas_dataframe()
    
    print("tritium production by Li-6:")
    print(breeding_dataframe)


def plot_damage_energy():
    tally = sp.get_tally(name="neutron damage_energy")
    slice = tally.get_slice(scores=['damage-energy'])
    slice.mean.shape = mesh_shape
    slice = np.add(slice.mean, 1e-15)
    data = aggregate_rectangular(slice)
    fig = plt.figure(figsize=(15, 10))
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, 4.2e17/16*), cmap='viridis', origin='lower', 
                    alpha=0.85, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e13, vmax=19))
    CS = plt.contour(np.multiply(data, 4.2e17/16), np.logspace(13, 19, 10), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('DT Neutron Damage Energy Rate\n 4.2e17 n/s Source Rate\n tungsten boride W2B5 10% vo water shield')
    plt.colorbar(im, label='Neutron Damage Energy $[eV/s]$', orientation='vertical',
                shrink=0.7, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e')
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    fig.savefig('./plots/DT damage yz')
    return fig

def plot_multiplying_flux():
    tally = sp.get_tally(name="Pb multiplying flux")
    slice = tally.get_slice(scores=['flux'])
    slice.mean.shape = mesh_shape
    slice = np.add(slice.mean, 1e-15)
    data = aggregate_rectangular(slice)
    fig = plt.figure(figsize=(15, 10))
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, 4.2e17), cmap='plasma', origin='lower', 
                    alpha=0.85, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e5, vmax=1e15))
    CS = plt.contour(np.multiply(data, 4.2e17), np.logspace(5, 15, 14), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('Fast Neutron Flux > 5 MeV for Multiplication \n 4.2e17 n/s Source Rate\n tungsten boride W2B5 10% vo water shield')
    plt.colorbar(im, label='Fast neutron flux $[n/cm^2-s]$', orientation='vertical',
                shrink=0.8, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e')
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    fig.savefig('./plots/multiplying flux yz')
    return fig

def plot_breeding():
    tally = sp.get_tally(name="Breeder mesh")
    slice = tally.get_slice(scores=['(n,Xt)'])
    slice.mean.shape = mesh_shape
    slice = np.add(slice.mean, 1e-15)
    data = aggregate_rectangular(slice)
    fig = plt.figure(figsize=(15, 10))
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, 4.2e17/16), cmap='plasma', origin='lower', 
                    alpha=0.85, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e6, vmax=1e13))
    CS = plt.contour(np.multiply(data, 4.2e17/16), np.logspace(6, 13, 14), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('(n, Xt) Breeding Sites \n 4.2e17 n/s Source Rate\n tungsten boride W2B5 10% vo water shield')
    plt.colorbar(im, label='(n, Xt) rate $[1/cm^3-s]$', orientation='vertical',
                shrink=0.8, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e')
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    fig.savefig('./plots/breeding yz')
    return fig

def plot_n2n():
    tally = sp.get_tally(name="Multiplier mesh")
    slice = tally.get_slice(scores=['(n,2n)'])
    slice.mean.shape = mesh_shape
    slice = np.add(slice.mean, 1e-15)
    data = aggregate_rectangular(slice)
    fig = plt.figure(figsize=(15, 10))
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, 4.2e17/16), cmap='plasma', origin='lower', 
                    alpha=0.85, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e6, vmax=1e13))
    CS = plt.contour(np.multiply(data, 4.2e17/16), np.logspace(6, 13, 14), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('(n, 2n) Multiplication Sites \n 4.2e17 n/s Source Rate\n tungsten boride W2B5 10% vo water shield')
    plt.colorbar(im, label='(n, 2n) rate $[1/cm^3-s]$', orientation='vertical',
                shrink=0.8, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e')
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    fig.savefig('./plots/n2n yz')
    return fig

def plot_n3n():
    tally = sp.get_tally(name="Multiplier mesh")
    slice = tally.get_slice(scores=['(n,3n)'])
    slice.mean.shape = mesh_shape
    slice = np.add(slice.mean, 1e-15)
    data = aggregate_rectangular(slice)
    fig = plt.figure(figsize=(15, 10))
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, 4.2e17/16), cmap='plasma', origin='lower', 
                    alpha=0.85, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e3, vmax=1e10))
    CS = plt.contour(np.multiply(data, 4.2e17/16), np.logspace(3, 10, 14), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('(n, 3n) Multiplication Sites \n 4.2e17 n/s Source Rate\n tungsten boride W2B5 10% vo water shield')
    plt.colorbar(im, label='(n, 3n) rate $[1/cm^3-s]$', orientation='vertical',
                shrink=0.8, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e')
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    fig.savefig('./plots/n3n yz')
    return fig

def plot_wall_load():
    fastflux_tally = sp.get_tally(name='fast flux')
    fast_flux = fastflux_tally.get_slice(scores=['flux'])
    fast_flux.mean.shape = mesh_shape
    fig3 = plt.figure(num=3, figsize=(15, 10))
    data = aggregate_rectangular(fast_flux.mean)
    #data = fast_flux.mean[:, :, 137]
    plt.plot(r_bounds, z_bounds, "g-")
    plt.plot(-r_bounds, z_bounds, "g-")
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, 4.2e17*2.25907e-12*10000), cmap='Spectral_r', origin='lower', 
                    alpha=.7, interpolation="quadric", extent=extent, norm=colors.Normalize(vmin=5e5, vmax=30e6))
    CS = plt.contour(np.multiply(data, 4.2e17*2.25907e-12*10000), np.linspace(5e5, 5e6, 10), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('DT Neutron Wall Load\n 4.2e17 n/s Source Rate')
    plt.colorbar(im, label='Wall load $[W/m^2]$', orientation='vertical',
                shrink=0.5, format='%0.00e')
    plt.clabel(CS, fmt='%0.0e')
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    plt.xlim(-100, 100)
    plt.ylim(0, 200)
    fig3.savefig('./plots/wall load yz')
    return fig3