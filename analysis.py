# %%
import shutil
import openmc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from cql3d_analysis_toolbox import ion_dens
import os
import concurrent.futures
import threading
 
import plotting as p

source_rate = 3.55e18
source_power = source_rate*1.6022e-13*17.6
source_rate_title = "{:.0e} Source ({:.1e} W DT Power)\n".format(source_rate, source_power)

# Setting up the plot parameters
plot_width = 275
#plot_height = 640
plot_height = 260*4
mesh_shape = (260, 275, 275)
#mesh_shape = (260*4, 1, 275)
extent=(-plot_width, plot_width, 0, plot_height)

statepoint_name = "statepoint.10-BAM-ALLW-PBLI-REALSOURCE-V5.h5"
plot_dir = "./plots/"+statepoint_name.split(".")[1]
if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)
sp = openmc.StatePoint("./" + statepoint_name, autolink=False)
subtitle = "Monolithic Tungsten shield - PBLI blanket"

def plasma_boundary(cql3d_file="WHAM_VNS_gen3_large_50keV_NBI_HFS_2p9Tres_2x2MWrf_5keV.nc"):
    """
    returns the outer plasma boundary in the outermost r-z grid of the plasma
    """
    ndwarmz, ndfz, ndtotz, solrz, solzz = ion_dens("WHAM_VNS_gen3_large_50keV_NBI_HFS_2p9Tres_2x2MWrf_5keV.nc", False)
    r_bounds = solrz[-1] * 100
    z_bounds = solzz[-1] * 100
    return r_bounds, z_bounds

def draw_psi(pleiades_file="WHAM_B3.00_beta0.30.npz", rscale=1, zscale=1, draw=True):
    """
    draw a contour plot of psi
    """
    with np.load(pleiades_file) as data:
        R = data['R'] * 100 * rscale
        Z = data['Z'] * 100 * zscale
        pos_extent = (R[0,0], R[0,-1], Z[0, 0], Z[-1, 0])
        neg_extent = (R[0,0], -R[0,-1], Z[0, 0], Z[-1, 0])
        psi = data['psi']
        min_psi = np.min(psi)
        #max_psi = np.max(psi)
        max_psi = psi[0,25]
        if draw:
            plt.contour(psi, [max_psi], extent=pos_extent, 
                        origin="lower", linewidths=3, linestyles="dashed", colors="magenta")
            plt.contour(psi, [max_psi], extent=neg_extent, 
                        origin="lower", linewidths=3, linestyles="dashed", colors="magenta")
    return psi, max_psi
        

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
    with concurrent.futures.ThreadPoolExecutor(max_workers=19) as executor:
        future_to_slice = [executor.submit(aggregate_slice, result[z_bins, :, :], result.shape[dim-1], center, r_bins, edges, z_bins)
                           for z_bins in range(z_length)]
        for future in concurrent.futures.as_completed(future_to_slice):
            try:
                z_slice, z = future.result()
                #print(z)
                aggregate[z, :] = z_slice
            except Exception as exp:
                print(str(z) + " generated an exception: %s" % (exp))
    return aggregate

def aggregate_slice(slice, x_range, center, r_bins, edges, z):
    result_slice = np.zeros(r_bins*2+2)
    #print(slice.shape)
    R = []
    W = {}
    for x in range(x_range):
        for y in range(x_range):
            r = np.sqrt((x-center)**2 + (y-center)**2)
            R.append(r)
            W[r] = slice[x, y]
    for i in range(r_bins-1):
        binned_r = [radius for radius in R if (radius <= edges[i+1] and radius > edges[i])]
        binned_w = [W[radius] for radius in binned_r]
        result_slice[int(r_bins-i)] = np.mean(binned_w)
        result_slice[int(r_bins+i)] = np.mean(binned_w)
    return result_slice, z

def plot_background(width, height):
    background_plot = p.slice_plot(basis="xz", origin=(0, 0, height/2), cwd='./background',
                                width=(width*2, height), color=p.material_color)
    background_plot.export_to_xml()
    openmc.plot_geometry(background_plot)
    shutil.copy("./background.png", plot_dir+'/background.png')
    shutil.copy("./geometry.py", plot_dir+'/geometry.py')
    shutil.copy("./boolean.py", plot_dir+'/boolean.py')

if not os.path.isdir('./plots'):
    os.makedirs('./plots')

r_bounds, z_bounds = plasma_boundary()
cyl_mesh = openmc.CylindricalMesh()
cyl_mesh.r_grid = np.linspace(0, 275, 275+1)
cyl_mesh.z_grid = np.linspace(0, 260*4, 260*4+1)
#plot_background(plot_width, plot_height)
# Read background image
background_image = plt.imread('./background.png')
def plot_result(sp, tally_name = 'thermal flux', tally_score = 'flux', mesh_dims = mesh_shape, save_aggregate = True, im_cmap = 'Spectral_r', contour_lvl = np.logspace(-1, 6, 8), plot_extent = extent, m_factor=source_rate/8,
                clabel = 'Thermal neutron flux $[n/cm^2-s]$', title = 'DT Thermal Flux (<0.5 eV)\n'+ source_rate_title + subtitle, savedir=plot_dir):
    tally = sp.get_tally(name=tally_name)
    slice = tally.get_slice(scores=[tally_score])
    slice.mean.shape = mesh_dims
    slice = np.add(slice.mean, 1e-15)
    #data_pos = np.divide(slice[:, 0, :], np.transpose(cyl_mesh.volumes[:, 0, :]))
    #data = np.concatenate([np.flip(data_pos, axis=1), data_pos], axis=1)
    #data = slice[:, :, 137]
    data = aggregate_rectangular_single(slice)
    if save_aggregate:
        np.savez(savedir + "/" + tally_name, data)
    
    fig = plt.figure(figsize=(10, 15))
    draw_psi()
    plt.imshow(background_image, extent=plot_extent)
    im = plt.imshow(np.multiply(data, m_factor), cmap=im_cmap, origin='lower', 
                    alpha=0.85, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=min(contour_lvl), vmax=max(contour_lvl)))
    CS = plt.contour(np.multiply(data, m_factor), contour_lvl, origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title(title)
    plt.colorbar(im, label=clabel, orientation='vertical', shrink=0.8, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e', fontsize=11)
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    plt.ylim(0, extent[3])
    fig.savefig(savedir+'/' + tally_name)
    return
# %%

def plot_photon_flux(sp):
    photonflux_tally = sp.get_tally(name='photon flux')
    photon_flux = photonflux_tally.get_slice(scores=['flux'])
    photon_flux.mean.shape = mesh_shape
    fig3 = plt.figure(num=3, figsize=(10, 15))
    #data_pos = np.divide(photon_flux.mean[:, 0, :], np.transpose(cyl_mesh.volumes[:, 0, :]))
    #data = np.concatenate([np.flip(data_pos, axis=1), data_pos], axis=1)
    data = aggregate_rectangular_single(photon_flux.mean)
    #data = photon_flux.mean[:, :, 137]
    draw_psi()
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, source_rate), cmap='Spectral_r', origin='lower', 
                    alpha=.7, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e8, vmax=1e15))
    CS = plt.contour(np.multiply(data, source_rate), np.logspace(8, 15, 12), origin="lower",
                     extent=extent, cmap='flag', linewidths=0.5)
    plt.title('Photon Flux (>100 keV)\n'+ source_rate_title + subtitle)
    plt.colorbar(im, label='Photon flux $[\gamma/cm^2-s]$', orientation='vertical',
                shrink=0.7, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e', fontsize=11)
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    plt.ylim(0, extent[3])
    print("saving plot to: " + plot_dir + "/DT photon flux yz")
    fig3.savefig(plot_dir+'/photon flux yz')
    return

def plot_total_heat(sp):
    heat_tally = sp.get_tally(name="neutron heat load")
    heat = heat_tally.get_slice(scores=['heating'])
    heat.mean.shape = mesh_shape
    normalized_heat = np.add(heat.mean, 1e-10)
    #data_pos = np.divide(normalized_heat[:, 0, :], np.transpose(cyl_mesh.volumes[:, 0, :]))
    #data = np.concatenate([np.flip(data_pos, axis=1), data_pos], axis=1)
    data = aggregate_rectangular_single(normalized_heat)
    #data = normalized_heat[:, :, 137]
    fig4 = plt.figure(num=4, figsize=(10, 15))
    draw_psi()
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, source_rate*1.602e-19/8), cmap='plasma', origin='lower', 
                    alpha=0.85, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e-5, vmax=1e1))
    CS = plt.contour(np.multiply(data, source_rate*1.602e-19/8), np.logspace(-5, 1, 10), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('DT Neutron Nuclear Heating with Gamma\n'+ source_rate_title + subtitle)
    plt.colorbar(im, label='Total Nuclear Heating $[W/cm^3]$', orientation='vertical',
                shrink=0.7, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e', fontsize=11)
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    plt.ylim(0, extent[3])
    fig4.savefig(plot_dir+'/DT heating with photon yz')
    return

def plot_local_heat(sp):
    local_heat_tally = sp.get_tally(name="neutron local heat load")
    local_heat = local_heat_tally.get_slice(scores=['heating-local'])
    local_heat.mean.shape = mesh_shape
    normalized_local_heat = np.add(local_heat.mean, 1e-10)
    #data_pos = np.divide(normalized_local_heat[:, 0, :], np.transpose(cyl_mesh.volumes[:, 0, :]))
    #data = np.concatenate([np.flip(data_pos, axis=1), data_pos], axis=1)
    data = aggregate_rectangular_single(normalized_local_heat)
    #data = normalized_local_heat[:, :, 137]
    fig5 = plt.figure(num=5, figsize=(10, 15))
    draw_psi()
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, source_rate*1.602e-19/8), cmap='plasma', origin='lower', 
                    alpha=0.85, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e-5, vmax=1e1))
    CS = plt.contour(np.multiply(data, source_rate*1.602e-19/8), np.logspace(-3, 1, 10), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('DT Neutron Local Heating with Gamma\n'+ source_rate_title + subtitle)
    plt.colorbar(im, label='Neutron local energy deposition $[W/cm^3]$', orientation='vertical',
                shrink=0.7, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e', fontsize=11)
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    plt.ylim(0, extent[3])
    fig5.savefig(plot_dir+'/DT local heating with photon yz')
    return

def plot_radiative_capture(sp):
    radiative_capture_tally = sp.get_tally(name="neutron radiative capture")
    radiative_capture = radiative_capture_tally.get_slice(scores=['(n,gamma)'])
    radiative_capture.mean.shape = mesh_shape
    radiative_capture = np.add(radiative_capture.mean, 1e-15)
    #data_pos = np.divide(radiative_capture[:, 0, :], np.transpose(cyl_mesh.volumes[:, 0, :]))
    #data = np.concatenate([np.flip(data_pos, axis=1), data_pos], axis=1)
    data = aggregate_rectangular_single(radiative_capture.mean)
    #data = radiative_capture[:, :, 137]
    fig6 = plt.figure(num=6, figsize=(10, 15))
    plt.imshow(background_image, extent=extent)
    draw_psi()
    im = plt.imshow(np.multiply(data, source_rate/8), cmap='viridis', origin='lower', 
                    alpha=0.85, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e7, vmax=1e13))
    CS = plt.contour(np.multiply(data, source_rate/8), np.logspace(7, 13, 14), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('DT Neutron Radiative Capture\n'+ source_rate_title + subtitle)
    plt.colorbar(im, label='Neutron radiative capture $[#/cm^3-s]$', orientation='vertical',
                shrink=0.7, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e', fontsize=11)
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    plt.ylim(0, extent[3])
    fig6.savefig(plot_dir+'/DT radiative capture yz')
    return

def plot_absorption(sp):
    absorption_tally = sp.get_tally(name="neutron absorption")
    absorption = absorption_tally.get_slice(scores=['absorption'])
    absorption.mean.shape = mesh_shape
    absorption = np.add(absorption.mean, 1e-15)
    #data_pos = np.divide(absorption[:, 0, :], np.transpose(cyl_mesh.volumes[:, 0, :]))
    #data = np.concatenate([np.flip(data_pos, axis=1), data_pos], axis=1)
    data = aggregate_rectangular_single(absorption)
    #data = absorption[:, :, 137]
    fig7 = plt.figure(num=7, figsize=(10, 15))
    draw_psi()
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, source_rate/8), cmap='viridis', origin='lower', 
                    alpha=0.85, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e14, vmax=1e19))
    CS = plt.contour(np.multiply(data, source_rate/8), np.logspace(14, 19, 14), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('DT Neutron Absorption\n'+ source_rate_title + subtitle)
    plt.colorbar(im, label='Neutron absorption $[#/cm^3-s]$', orientation='vertical',
                shrink=0.7, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e', fontsize=11)
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    plt.ylim(0, extent[3])
    fig7.savefig(plot_dir+'/DT absorption yz')
    return

def plot_coil_flux_energy(sp):
    avg_coil_flux_tally = sp.get_tally(name='Average neutron flux')
    avg_coil_flux = avg_coil_flux_tally.get_slice(scores=['flux'])
    fig = plt.figure(num=8, figsize=(10, 15))
    flux = np.multiply(avg_coil_flux.mean, source_rate)
    plt.plot(np.logspace(-4, 7, 999), flux[:, 0, 0])
    plt.title('Neutron energy spectrum through coil \n'+ source_rate_title + subtitle)
    plt.xlabel('Neutron energy (eV)')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Neutron flux (n/cm2-s)')
    fig.savefig(plot_dir+'/Coil flux energy')
    return

def get_tbr(sp):
    li6_breeding_tally = sp.get_tally(name='Breeder Li-6(n,alpha)T reaction')
    li6_breeding = li6_breeding_tally.get_slice(scores=['(n,Xt)'])
    breeding_dataframe = li6_breeding.get_pandas_dataframe()
    
    print("tritium production by Li-6:")
    print(breeding_dataframe)


def plot_np(sp):
    tally = sp.get_tally(name="Breeder mesh")
    slice = tally.get_slice(scores=['(n,p)'])
    slice.mean.shape = mesh_shape
    slice = np.add(slice.mean, 1e-15)
    #data_pos = np.divide(slice[:, 0, :], np.transpose(cyl_mesh.volumes[:, 0, :]))
    #data = np.concatenate([np.flip(data_pos, axis=1), data_pos], axis=1)
    data = aggregate_rectangular_single(slice)
    #data = slice[:, :, 137]
    fig = plt.figure(figsize=(10, 15))
    plt.imshow(background_image, extent=extent)
    #draw_psi()
    im = plt.imshow(np.multiply(data, source_rate/8), cmap='plasma', origin='lower', 
                    alpha=0.85, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e6, vmax=1e13))
    CS = plt.contour(np.multiply(data, source_rate/8), np.logspace(6, 13, 14), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('(n, p) Breeding Sites \n'+ source_rate_title + subtitle)
    plt.colorbar(im, label='(n, Xt) rate $[1/cm^3-s]$', orientation='vertical',
                shrink=0.8, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e', fontsize=11)
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    plt.ylim(0, extent[3])
    fig.savefig(plot_dir+'/breeding yz')
    return


def plot_n3n(sp):
    tally = sp.get_tally(name="Multiplier mesh")
    slice = tally.get_slice(scores=['(n,3n)'])
    slice.mean.shape = mesh_shape
    slice = np.add(slice.mean, 1e-15)
    #data_pos = np.divide(slice[:, 0, :], np.transpose(cyl_mesh.volumes[:, 0, :]))
    #data = np.concatenate([np.flip(data_pos, axis=1), data_pos], axis=1)
    #data = aggregate_rectangular_single(slice)
    data = slice[:, :, 137]
    fig = plt.figure(figsize=(10, 15))
    draw_psi()
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, source_rate/8), cmap='plasma', origin='lower', 
                    alpha=0.85, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e3, vmax=1e10))
    CS = plt.contour(np.multiply(data, source_rate/8), np.logspace(3, 10, 14), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('(n, 3n) Multiplication Sites \n'+ source_rate_title + subtitle)
    plt.colorbar(im, label='(n, 3n) rate $[1/cm^3-s]$', orientation='vertical',
                shrink=0.8, format='%0.0e')
    plt.clabel(CS, fmt='%0.0e', fontsize=11)
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    plt.ylim(0, extent[3])
    fig.savefig(plot_dir+'/n3n yz')
    return

def plot_wall_load(sp):
    fastflux_tally = sp.get_tally(name='fast flux')
    fast_flux = fastflux_tally.get_slice(scores=['flux'])
    fast_flux.mean.shape = mesh_shape
    fig3 = plt.figure(num=3, figsize=(10, 15))
    draw_psi()
    #data_pos = np.divide(fast_flux.mean[:, 0, :], np.transpose(cyl_mesh.volumes[:, 0, :]))
    #data = np.concatenate([np.flip(data_pos, axis=1), data_pos], axis=1)
    #data = aggregate_rectangular_single(fast_flux.mean)
    data = fast_flux.mean[:, :, 137]
    plt.plot(r_bounds, z_bounds, "g-")
    plt.plot(-r_bounds, z_bounds, "g-")
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, source_rate*2.25907e-12*10000), cmap='Spectral_r', origin='lower', 
                    alpha=.7, interpolation="quadric", extent=extent, norm=colors.Normalize(vmin=5e5, vmax=30e6))
    CS = plt.contour(np.multiply(data, source_rate*2.25907e-12*10000), np.linspace(5e5, 5e6, 10), origin="lower",
                    extent=extent, cmap='flag', linewidths=0.5)
    plt.title('DT Neutron Wall Load\n'+ source_rate_title + subtitle)
    plt.colorbar(im, label='Wall load $[W/m^2]$', orientation='vertical',
                shrink=0.5, format='%0.00e')
    plt.clabel(CS, fmt='%0.0e', fontsize=11)
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    plt.xlim(-100, 100)
    plt.ylim(0, 200)
    fig3.savefig(plot_dir+'/wall load yz')
    return

def plot_neutron_source(statepoint_name, wallload_factor=1):
    sp = openmc.StatePoint("./" + statepoint_name, autolink=False)
    tally = sp.get_tally(name="Pb multiplying flux")
    slice = tally.get_slice(scores=['flux'])
    slice.mean.shape = mesh_shape
    slice = np.add(slice.mean, 1e-15)
    data = aggregate_rectangular_single(slice)
    #data_pos = np.divide(slice[:, 0, :], np.transpose(cyl_mesh.volumes[:, 0, :]))
    #data = np.concatenate([np.flip(data_pos, axis=1), data_pos], axis=1)
    #data = slice[:, :, 137]
    np.savez("./plots/neutron_source", data)
    fig = plt.figure(figsize=(10, 15))
    #draw_psi()
    plt.imshow(background_image, extent=extent)
    im = plt.imshow(np.multiply(data, source_rate/8*wallload_factor), cmap='plasma', origin='lower', 
                    alpha=0.45, interpolation="quadric", extent=extent, norm=colors.LogNorm(vmin=1e14, vmax=1e16))
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    plt.ylim(0, extent[3])

    fig.savefig(plot_dir+'/neutron source figure yz')
    return

# %%