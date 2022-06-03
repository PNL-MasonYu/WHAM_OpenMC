# %%
import openmc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from cql3d_analysis_toolbox import ion_dens
import os
#import concurrent.futures
#import threading
 
import plotting as p

def plasma_boundary(cql3d_file="WHAM_VNS_gen3_large_50keV_NBI_HFS_2p9Tres_2x2MWrf_5keV.nc"):
    """
    returns the outer plasma boundary in the outermost r-z grid of the plasma
    """
    ndwarmz, ndfz, ndtotz, solrz, solzz = ion_dens("WHAM_VNS_gen3_large_50keV_NBI_HFS_2p9Tres_2x2MWrf_5keV.nc", False)
    r_bounds = solrz[-1] * 100
    z_bounds = solzz[-1] * 100
    return r_bounds, z_bounds

def draw_psi(pleiades_file="v2q1reactor_B2.53_beta0.00.npz"):
    """
    draw a contour plot of psi
    """
    with np.load('path_to_npz_file.npz') as data:
        R = data['R']
        Z = data['Z']
        pos_extent = (R[0,0], R[0,-1], Z[0, 0], Z[0, -1])
        neg_extent = (-R[0,-1], R[0,0], Z[0, 0], Z[0, -1])
        psi = data['psi']
        plt.contour(psi, np.linspace(0,10), extent=pos_extent, origin="lower")
        plt.contour(psi, np.linspace(0,10), extent=pos_extent, origin="lower")
            

def aggregate_rectangular_single(result, r_bins=100, dim=0):
    """
    aggegate the cylindrical tally in the r direction
    The z-dimension should be specified in the dim variable
    The center of the tally will be the axis of revolution
    Single threaded implementation
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

def aggregate_rectangular_multi(result, r_bins=100, dim=0):
    """
    aggegate the cylindrical tally in the r direction
    The z-dimension should be specified in the dim variable
    The center of the tally will be the axis of revolution
    Multithreaded implementation
    """
    center = result.shape[dim-1]/2
    z_length = result.shape[dim]
    aggregate = np.zeros((z_length, r_bins*2+2))
    edges = np.linspace(0, int(center), num=r_bins)
    #print(z_length)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        future_to_slice = {executor.submit(aggregate_slice, result, dim, center, z_bins, r_bins, edges)
                           for z_bins in range(z_length)}
        for future in concurrent.futures.as_completed(future_to_slice):
            try:
                z_slice, z = future.result()
                aggregate[z, :] = z_slice
            except Exception as exp:
                print(str(z) + " generated an exception: %s" % (exp))
    return aggregate

def aggregate_slice(result, dim, center, z, r_bins, edges):
    result_slice = np.zeros(r_bins*2+2)
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
        result_slice[int(r_bins-i)] = np.mean(binned_w)
        result_slice[int(r_bins+i)] = np.mean(binned_w)
    return result_slice, z

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
statepoint_name = "statepoint.200-stainless-LiPb-worst-source.h5"
plot_dir = "./plots/"+statepoint_name.split(".")[1]
if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)
sp = openmc.StatePoint("./" + statepoint_name, autolink=False)
subtitle = "All Stainless 316 Shield (LiPb Blanket)"
r_bounds, z_bounds = plasma_boundary()

#
mesh_shape = (160, 275, 275)
extent=(-275, 275, 0, 640)

background_image = plt.imread('./background.ppm')
# %%
def plot_thermal_flux(sp):
    thermalflux_tally = sp.get_tally(name='thermal flux')
    thermal_flux = thermalflux_tally.get_slice(scores=['flux'])
    thermal_flux.mean.shape = mesh_shape
    fig1 = plt.figure(num=1, figsize=(15, 10))
    data = aggregate_rectangular_single(thermal_flux.mean)
    #data = thermal_flux.mean[:, :, 137]
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, 1e18), cmap='Spectral_r', origin='lower', 
                    alpha=.7, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=0.1, vmax=1e6))
    CS = plt.contour(np.multiply(data, 1e18), np.logspace(-1, 6, 8), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.colorbar(im, label='Thermal neutron flux $[n/cm^2-s]$', orientation='vertical',
                shrink=0.7, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e')
    plt.title('DT Thermal Flux (<0.5 eV)\n 1e18 n/s Source Rate\n tungsten boride W2B5 10% vo water shield')
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    print("saving plot to " + plot_dir + "/DT strict thermal flux yz")
    fig1.savefig(plot_dir+'/DT strict thermal flux yz')
    return

def plot_epithermal_flux(sp):
    epithermalflux_tally = sp.get_tally(name='epithermal flux')
    epithermal_flux = epithermalflux_tally.get_slice(scores=['flux'])
    epithermal_flux.mean.shape = mesh_shape
    data = aggregate_rectangular_single(epithermal_flux.mean)
    #data = epithermal_flux.mean[:, :, 137]
    fig2 = plt.figure(num=2, figsize=(15, 10))
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, 1e18), cmap='Spectral_r', origin='lower', 
                    alpha=.7, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e10, vmax=1e16))
    CS = plt.contour(np.multiply(data, 1e18), np.logspace(10, 17, 16), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('DT Epithermal Flux (0.5 eV - 100 keV)\n 1e18 n/s Source Rate\n' + subtitle)
    plt.colorbar(im, label='Epithermal neutron flux $[n/cm^2-s]$', orientation='vertical',
                shrink=0.7, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e')
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    fig2.savefig(plot_dir+'/DT strict epithermal flux yz')
    return

def plot_fast_flux(sp):
    fastflux_tally = sp.get_tally(name='fast flux')
    fast_flux = fastflux_tally.get_slice(scores=['flux'])
    fast_flux.mean.shape = mesh_shape
    fig3 = plt.figure(num=3, figsize=(15, 10))
    data = aggregate_rectangular_single(fast_flux.mean)
    #data = fast_flux.mean[:, :, 137]
    #plt.plot(r_bounds, z_bounds, "g-")
    #plt.plot(-r_bounds, z_bounds, "g-")
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, 1e18), cmap='Spectral_r', origin='lower', 
                    alpha=.7, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e8, vmax=1e15))
    CS = plt.contour(np.multiply(data, 1e18), np.logspace(8, 15, 12), origin="lower",
                     extent=extent, cmap='flag', linewidths=0.5)
    plt.title('DT Fast Flux (>100 keV)\n 1e18 n/s Source Rate\n' + subtitle)
    plt.colorbar(im, label='Fast neutron flux $[n/cm^2-s]$', orientation='vertical',
                shrink=0.7, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e')
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    print("saving plot to: " + plot_dir + "/DT strict fast flux yz")
    fig3.savefig(plot_dir+'/DT strict fast flux yz')
    return

def plot_total_heat(sp):
    heat_tally = sp.get_tally(name="neutron heat load")
    heat = heat_tally.get_slice(scores=['heating'])
    heat.mean.shape = mesh_shape
    normalized_heat = np.add(heat.mean, 1e-10)
    data = aggregate_rectangular_single(normalized_heat)
    #data = normalized_heat.mean[:, :, 137]
    fig4 = plt.figure(num=4, figsize=(15, 10))
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, 1e18*1.602e-19/8), cmap='plasma', origin='lower', 
                    alpha=0.85, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e-5, vmax=1e1))
    CS = plt.contour(np.multiply(data, 1e18*1.602e-19/8), np.logspace(-5, 1, 10), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('DT Neutron Nuclear Heating with Gamma\n 1e18 n/s Source Rate\n' + subtitle)
    plt.colorbar(im, label='Total Nuclear Heating $[W/cm^3]$', orientation='vertical',
                shrink=0.7, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e')
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    fig4.savefig(plot_dir+'/DT heating with photon yz')
    return

def plot_local_heat(sp):
    local_heat_tally = sp.get_tally(name="neutron local heat load")
    local_heat = local_heat_tally.get_slice(scores=['heating-local'])
    local_heat.mean.shape = mesh_shape
    normalized_local_heat = np.add(local_heat.mean, 1e-10)
    data = aggregate_rectangular_single(normalized_local_heat)
    #data = normalized_local_heat.mean[:, :, 137]
    fig5 = plt.figure(num=5, figsize=(15, 10))
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, 1e18*1.602e-19/8), cmap='plasma', origin='lower', 
                    alpha=0.85, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e-5, vmax=1e1))
    CS = plt.contour(np.multiply(data, 1e18*1.602e-19/8), np.logspace(-3, 1, 10), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('DT Neutron Local Heating with Gamma\n 1e18 n/s Source Rate\n' + subtitle)
    plt.colorbar(im, label='Neutron local energy deposition $[W/cm^3]$', orientation='vertical',
                shrink=0.7, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e')
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    fig5.savefig(plot_dir+'/DT local heating with photon yz')
    return

def plot_radiative_capture(sp):
    radiative_capture_tally = sp.get_tally(name="neutron radiative capture")
    radiative_capture = radiative_capture_tally.get_slice(scores=['(n,gamma)'])
    radiative_capture.mean.shape = mesh_shape
    radiative_capture = np.add(radiative_capture.mean, 1e-15)
    data = aggregate_rectangular_single(radiative_capture.mean)
    #data = radiative_capture.mean[:, :, 137]
    fig6 = plt.figure(num=6, figsize=(15, 10))
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, 1e18/16), cmap='viridis', origin='lower', 
                    alpha=0.85, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e7, vmax=1e13))
    CS = plt.contour(np.multiply(data, 1e18/16), np.logspace(7, 13, 14), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('DT Neutron Radiative Capture\n 1e18 n/s Source Rate\n' + subtitle)
    plt.colorbar(im, label='Neutron radiative capture $[#/cm^3-s]$', orientation='vertical',
                shrink=0.7, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e')
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    fig6.savefig(plot_dir+'/DT radiative capture yz')
    return

def plot_absorption(sp):
    absorption_tally = sp.get_tally(name="neutron absorption")
    absorption = absorption_tally.get_slice(scores=['absorption'])
    absorption.mean.shape = mesh_shape
    absorption = np.add(absorption.mean, 1e-15)
    data = aggregate_rectangular_single(absorption)
    #data = absorption.mean[:, :, 137]
    fig7 = plt.figure(num=7, figsize=(15, 10))
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, 1e18/16), cmap='viridis', origin='lower', 
                    alpha=0.85, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e14, vmax=1e19))
    CS = plt.contour(np.multiply(data, 1e18/16), np.logspace(14, 19, 14), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('DT Neutron Absorption\n 1e18 n/s Source Rate\n' + subtitle)
    plt.colorbar(im, label='Neutron absorption $[#/cm^3-s]$', orientation='vertical',
                shrink=0.7, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e')
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    fig7.savefig(plot_dir+'/DT absorption yz')
    return

def plot_coil_flux_energy(sp):
    avg_coil_flux_tally = sp.get_tally(name='Average neutron flux')
    avg_coil_flux = avg_coil_flux_tally.get_slice(scores=['flux'])
    fig = plt.figure(num=8, figsize=(15, 10))
    flux = np.multiply(avg_coil_flux.mean, 1e18)
    plt.plot(np.logspace(-4, 7, 999), flux[:, 0, 0])
    plt.title('Neutron energy spectrum through coil \n 1e18 n/s Source Rate')
    plt.xlabel('Neutron energy (eV)')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Neutron flux (n/cm2-s)')
    fig.savefig(plot_dir+'Coil flux energy')
    return

def get_tbr(sp):
    li6_breeding_tally = sp.get_tally(name='Breeder Li-6(n,alpha)T reaction')
    li6_breeding = li6_breeding_tally.get_slice(scores=['(n,Xt)'])
    breeding_dataframe = li6_breeding.get_pandas_dataframe()
    
    print("tritium production by Li-6:")
    print(breeding_dataframe)


def plot_damage_energy(sp):
    tally = sp.get_tally(name="neutron damage_energy")
    slice = tally.get_slice(scores=['damage-energy'])
    slice.mean.shape = mesh_shape
    slice = np.add(slice.mean, 1e-15)
    data = aggregate_rectangular_single(slice)
    fig = plt.figure(figsize=(15, 10))
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, 1e18/16), cmap='viridis', origin='lower', 
                    alpha=0.85, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e13, vmax=19))
    CS = plt.contour(np.multiply(data, 1e18/16), np.logspace(13, 19, 10), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('DT Neutron Damage Energy Rate\n 1e18 n/s Source Rate\n' + subtitle)
    plt.colorbar(im, label='Neutron Damage Energy $[eV/s]$', orientation='vertical',
                shrink=0.7, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e')
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    fig.savefig(plot_dir+'/DT damage yz')
    return

def plot_multiplying_flux(sp):
    tally = sp.get_tally(name="Pb multiplying flux")
    slice = tally.get_slice(scores=['flux'])
    slice.mean.shape = mesh_shape
    slice = np.add(slice.mean, 1e-15)
    data = aggregate_rectangular_single(slice)
    fig = plt.figure(figsize=(15, 10))
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, 1e18), cmap='plasma', origin='lower', 
                    alpha=0.85, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e5, vmax=1e15))
    CS = plt.contour(np.multiply(data, 1e18), np.logspace(5, 15, 14), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('Fast Neutron Flux > 5 MeV for Multiplication \n 1e18 n/s Source Rate\n' + subtitle)
    plt.colorbar(im, label='Fast neutron flux $[n/cm^2-s]$', orientation='vertical',
                shrink=0.8, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e')
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    fig.savefig(plot_dir+'/multiplying flux yz')
    return

def plot_breeding(sp):
    tally = sp.get_tally(name="Breeder mesh")
    slice = tally.get_slice(scores=['(n,Xt)'])
    slice.mean.shape = mesh_shape
    slice = np.add(slice.mean, 1e-15)
    data = aggregate_rectangular_single(slice)
    fig = plt.figure(figsize=(15, 10))
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, 1e18/16), cmap='plasma', origin='lower', 
                    alpha=0.85, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e6, vmax=1e13))
    CS = plt.contour(np.multiply(data, 1e18/16), np.logspace(6, 13, 14), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('(n, Xt) Breeding Sites \n 1e18 n/s Source Rate\n' + subtitle)
    plt.colorbar(im, label='(n, Xt) rate $[1/cm^3-s]$', orientation='vertical',
                shrink=0.8, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e')
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    fig.savefig(plot_dir+'/breeding yz')
    return

def plot_n2n(sp):
    tally = sp.get_tally(name="Multiplier mesh")
    slice = tally.get_slice(scores=['(n,2n)'])
    slice.mean.shape = mesh_shape
    slice = np.add(slice.mean, 1e-15)
    data = aggregate_rectangular_single(slice)
    fig = plt.figure(figsize=(15, 10))
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, 1e18/16), cmap='plasma', origin='lower', 
                    alpha=0.85, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e6, vmax=1e13))
    CS = plt.contour(np.multiply(data, 1e18/16), np.logspace(6, 13, 14), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('(n, 2n) Multiplication Sites \n 1e18 n/s Source Rate\n' + subtitle)
    plt.colorbar(im, label='(n, 2n) rate $[1/cm^3-s]$', orientation='vertical',
                shrink=0.8, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e')
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    fig.savefig(plot_dir+'/n2n yz')
    return

def plot_n3n(sp):
    tally = sp.get_tally(name="Multiplier mesh")
    slice = tally.get_slice(scores=['(n,3n)'])
    slice.mean.shape = mesh_shape
    slice = np.add(slice.mean, 1e-15)
    data = aggregate_rectangular_single(slice)
    fig = plt.figure(figsize=(15, 10))
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, 1e18/16), cmap='plasma', origin='lower', 
                    alpha=0.85, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e3, vmax=1e10))
    CS = plt.contour(np.multiply(data, 1e18/16), np.logspace(3, 10, 14), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('(n, 3n) Multiplication Sites \n 1e18 n/s Source Rate\n' + subtitle)
    plt.colorbar(im, label='(n, 3n) rate $[1/cm^3-s]$', orientation='vertical',
                shrink=0.8, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e')
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    fig.savefig(plot_dir+'/n3n yz')
    return

def plot_wall_load(sp):
    fastflux_tally = sp.get_tally(name='fast flux')
    fast_flux = fastflux_tally.get_slice(scores=['flux'])
    fast_flux.mean.shape = mesh_shape
    fig3 = plt.figure(num=3, figsize=(15, 10))
    data = aggregate_rectangular_single(fast_flux.mean)
    #data = fast_flux.mean[:, :, 137]
    plt.plot(r_bounds, z_bounds, "g-")
    plt.plot(-r_bounds, z_bounds, "g-")
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, 1e18*2.25907e-12*10000), cmap='Spectral_r', origin='lower', 
                    alpha=.7, interpolation="quadric", extent=extent, norm=colors.Normalize(vmin=5e5, vmax=30e6))
    CS = plt.contour(np.multiply(data, 1e18*2.25907e-12*10000), np.linspace(5e5, 5e6, 10), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('DT Neutron Wall Load\n 1e18 n/s Source Rate')
    plt.colorbar(im, label='Wall load $[W/m^2]$', orientation='vertical',
                shrink=0.5, format='%0.00e')
    plt.clabel(CS, fmt='%0.0e')
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    plt.xlim(-100, 100)
    plt.ylim(0, 200)
    fig3.savefig(plot_dir+'/wall load yz')
    return

#plot_fast_flux(sp)
#plot_absorption(sp)
#plot_breeding(sp)
#plot_damage_energy(sp)
#plot_epithermal_flux(sp)
#plot_total_heat(sp)
#plot_thermal_flux(sp)
#plot_epithermal_flux(sp)
"""
ff = threading.Thread(target=plot_fast_flux, args=(sp, ))
ff.start()
ab = threading.Thread(target=plot_absorption, args=(sp, ))
ab.start()
br = threading.Thread(target=plot_breeding, args=(sp, ))
br.start()
de = threading.Thread(target=plot_damage_energy, args=(sp, ))
de.start()
he = threading.Thread(target=plot_total_heat, args=(sp, ))
he.start()
th = threading.Thread(target=plot_thermal_flux, args=(sp, ))
th.start()
ep = threading.Thread(target=plot_epithermal_flux, args=(sp, ))
ep.start()
ff.join()
ab.join()
br.join()
de.join()
he.join()
th.join()
ep.join()

with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    executor.submit(plot_fast_flux, sp)
    #executor.submit(plot_wall_load, sp)
    executor.submit(plot_absorption, sp)
    executor.submit(plot_breeding, sp)
    #executor.submit(plot_damage_energy, sp)
    executor.submit(plot_local_heat, sp)
    executor.submit(plot_epithermal_flux, sp)
    executor.submit(plot_thermal_flux, sp)
    #executor.submit(plot_n2n, sp)
    #executor.submit(plot_radiative_capture, sp)
"""
# %%