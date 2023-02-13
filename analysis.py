# %%
import shutil
import openmc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from cql3d_analysis_toolbox import ion_dens
import os
import concurrent.futures
import threading
 
import plotting as p

# Default plotting params
statepoint_name = "statepoint.10-BAM-ALLW-PBLI-REALSOURCE-V12.h5"
plot_dir = "./plots/"+statepoint_name.split(".")[1]
if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)
sp = openmc.StatePoint("./" + statepoint_name, autolink=False)
#plot_neutron_source("statepoint.10-BAM_REALSOURCE-VACUUM.h5")
source_rate = 1.5e18
#source_power = source_rate*1.6022e-13*17.6
source_power = source_rate*1.6022e-13*42.5
source_rate_title = "{:.0e} Source ({:.1e} W Catalyzed DD Power)\n".format(source_rate, source_power)
subtitle = "1 inch Beryllium multiplier - K blanket (600K)"
# Setting up the plot parameters
plot_width = 350
#plot_height = 640
plot_height = 260*4
mesh_shape = (260, 275, 275)
extent=(-plot_width, plot_width, 0, plot_height)
# Read background image
#background_image = plt.imread('./background.ppm')
background_image = plt.imread(plot_dir + '/background.png')


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
        

def aggregate_rectangular_single(result, r_bins=100, dim=0, max_r = 0):
    """
    aggegate the cylindrical tally in the r direction
    The z-dimension should be specified in the dim variable
    The center of the tally will be the axis of revolution
    Single threaded implementation
    """
    center = result.shape[dim-1]/2
    if max_r == 0:
        max_r = center
    z_length = result.shape[dim]
    aggregate = np.zeros((z_length, r_bins*2+2))
    edges = np.linspace(0, int(max_r), num=r_bins)
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
    shutil.copy("./source.py", plot_dir+'/source.py')

if not os.path.isdir('./plots'):
    os.makedirs('./plots')

r_bounds, z_bounds = plasma_boundary()
cyl_mesh = openmc.CylindricalMesh()
cyl_mesh.r_grid = np.linspace(0, 275, 275+1)
cyl_mesh.z_grid = np.linspace(0, 260*4, 260*4+1)


def plot_result(sp, tally_name = 'thermal flux', tally_score = 'flux', mesh_dims = mesh_shape, save_aggregate = True, nuclide_dim=0, im_cmap = 'Spectral_r', contour_lvl = np.logspace(-1, 6, 8), plot_extent = extent, m_factor=source_rate/8,
                clabel = 'Thermal neutron flux $[n/cm^2-s]$', title = 'DT Thermal Flux (<0.5 eV)\n'+ source_rate_title + subtitle, savedir=plot_dir, from_file=False):
    if not from_file:
        tally = sp.get_tally(name=tally_name)
        slice = tally.get_slice(scores=[tally_score])
        slice.mean.shape = mesh_dims
        if nuclide_dim == 0:
            slice =slice.mean
        else:
            slice = slice.mean[:, :, :, nuclide_dim]
        #data_pos = np.divide(slice[:, 0, :], np.transpose(cyl_mesh.volumes[:, 0, :]))
        #data = np.concatenate([np.flip(data_pos, axis=1), data_pos], axis=1)
        #data = slice[:, :, 137]
        
        data = aggregate_rectangular_single(slice, max_r=extent[1]/2)
        if save_aggregate:
            np.savez(savedir + "/" + tally_name, data)
    else:
        data = np.load(savedir + "/" + tally_name + ".npz")['arr_0']

    data = np.add(data, min(contour_lvl)/m_factor/100)
    data = np.nan_to_num(data, copy=False)
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

def plot_coil_flux_energy(sp, source_rate, source_rate_title, subtitle, plot_dir):
    avg_coil_flux_tally = sp.get_tally(name='Average neutron flux')
    avg_coil_flux = avg_coil_flux_tally.get_slice(scores=['flux'])
    fig = plt.figure(num=8, figsize=(15, 10))
    flux = np.multiply(avg_coil_flux.mean, source_rate)
    plt.plot(np.logspace(-4, 7, 999), flux[:, 0, 0])
    plt.title('Neutron energy spectrum through coil \n'+ source_rate_title + subtitle)
    plt.xlabel('Neutron energy (eV)')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Neutron flux (n/cm2-s)')
    fig.savefig(plot_dir+'/Coil flux energy')
    return

def plot_flux_energy(sp, source_rate, source_rate_title, subtitle, plot_dir, cell=6000, savedir=plot_dir, save=True):
    avg_flux_tally = sp.get_tally(name='neutron spectrum all cell')
    avg_flux = avg_flux_tally.get_slice(scores=['flux'], filters=[openmc.CellFilter], filter_bins=[(cell,)])
    avg_flux_dataframe = avg_flux.get_pandas_dataframe()
    energy = avg_flux_dataframe["energy low [eV]"]
    flux = avg_flux_dataframe["mean"] * source_rate
    fig = plt.figure(num=8, figsize=(12, 5))
    plt.plot(energy, flux)
    plt.title('Neutron energy spectrum through breeder \n'+ source_rate_title + subtitle)
    plt.xlabel('Neutron energy (eV)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([1, 20e6])
    plt.ylim([1e15, 1e20])
    plt.ylabel('Neutron flux (n/cm2-s)')
    fig.savefig(plot_dir+'/flux energy')
    if save:
        np.savez(savedir + "/neutron spectrum all cell " + str(cell), flux)
    return

def get_tbr(sp, tally_name = 'Breeder Li-6(n,alpha)T reaction'):
    li6_breeding_tally = sp.get_tally(name=tally_name)
    li6_breeding = li6_breeding_tally.get_slice(scores=['(n,Xt)'])
    breeding_dataframe = li6_breeding.get_pandas_dataframe()
    
    print("tritium production by Li-6:")
    print(breeding_dataframe)

def get_np(sp):
    breeding_tally = sp.get_tally(name='Breeder misc reaction')
    breeding = breeding_tally.get_slice(scores=['(n,p)', '(n,a)', '(n,gamma)', '(n,2n)'])
    breeding_dataframe = breeding.get_pandas_dataframe()
    
    print("Ar39 production:")
    print(breeding_dataframe)

def get_absorption_all(sp):
    tally = sp.get_tally(name="all cell neutron absorption")
    absorption = tally.get_slice(scores=['absorption'])
    absorption_dataframe = absorption.get_pandas_dataframe()
    absorption_dataframe.sort_values(by=['mean'], ascending=False, inplace=True)
    print("Neutron Absorption probability in all cells:")
    print(absorption_dataframe)

def plot_wall_load(sp, mesh_shape, extent, source_rate, source_rate_title, subtitle, plot_dir, background_image):
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

def plot_neutron_source(statepoint_name, source_rate, mesh_shape, background_image, extent, wallload_factor=1):
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

def plot_flux_heat_breed():
    source_rate = 3.55e18
    volume = 16
    # load data
    fast_flux_npzfile = np.load("plots/10-BAM-ALLW-PBLI-REALSOURCE-V12/fast flux.npz")
    fast_flux = fast_flux_npzfile['arr_0']
    heat_npzfile = np.load("plots/10-BAM-ALLW-PBLI-REALSOURCE-V12/neutron heat load.npz")
    heat = np.add(heat_npzfile['arr_0'], 1e-10)
    np.nan_to_num(heat, copy=False)
    breeder_npzfile = np.load("plots/10-BAM-ALLW-PBLI-REALSOURCE-V12/Breeder mesh.npz")
    breed = np.add(breeder_npzfile['arr_0'], 1e-15)
    damage_npzfile = np.load("plots/10-BAM-ALLW-PBLI-REALSOURCE-V12/neutron damage_energy.npz")
    damage = np.add(damage_npzfile['arr_0'], 1e-15)
    np.nan_to_num(damage, copy=False)
    vacuum_source_npzfile = np.load("plots/neutron_source.npz")
    vacuum_source = np.add(vacuum_source_npzfile['arr_0'], 1e-15)
    np.nan_to_num(vacuum_source)
    background_image = plt.imread("plots/10-BAM-ALLW-FLIBE-REALSOURCE-V12/background.png")

    fast_flux_data = np.multiply(fast_flux, source_rate/volume)
    nuclear_heat_data = np.multiply(heat, source_rate*1.602e-19/volume)
    dpa_data = np.multiply(damage, source_rate*0.8/(2*40)*3600*24*365/(volume*4.29129535/(55.845*1.66054E-24)))
    breeding_data = np.multiply(breed, source_rate*6/6.0221408e23/volume*3600*24*365*1e6)
    wall_load_data = np.multiply(vacuum_source, source_rate/volume*14e6*1.602e-19/1e6*1e4)
    
    plot_extent = (-275, 275, 0, 260*4)
    
    fig1, axs1 = plt.subplots(1, 3, layout="constrained")
    fig_scale = 2
    fig1.set_size_inches(7*fig_scale, 4*fig_scale)
    psi, max_psi = draw_psi(draw=False)

    fig2, axs2 = plt.subplots(1, 3, layout="constrained")
    fig_scale = 2
    fig2.set_size_inches(8*fig_scale, 4*fig_scale)

    axs1[0].imshow(background_image, extent=plot_extent)
    
    im0 = axs1[0].imshow(fast_flux_data, cmap="Spectral_r", origin='lower', alpha=0.7, interpolation='quadric',
                        extent=plot_extent, norm=colors.LogNorm(vmin=1e8, vmax=1e16))
    cs0 = axs1[0].contour(fast_flux_data, np.logspace(9, 16, 8), origin="lower",
                         extent=extent, cmap='flag', linewidths=0.6, norm=colors.LogNorm(vmin=1e9, vmax=1e16))
    cbar0 = fig1.colorbar(im0, ax=axs1[0], orientation="horizontal", shrink=1, format='%0.0e', pad=0.1)
    cbar0.set_label(r"Fast Neutron Flux (>100keV) $\left[\dfrac{n}{cm^2s}\right]$", fontsize=16)
    cbar0.ax.tick_params(labelsize=10)
    plt.clabel(cs0, fmt='%0.0e', fontsize=9)
    cs0.collections[1].set_linestyle('dotted')
    cs0.collections[1].set_linewidth(2)
    cs0.collections[2].set_linestyle('dashed')
    cs0.collections[2].set_linewidth(2)

    axs1[1].imshow(background_image, extent=plot_extent)
    
    im1 = axs1[1].imshow(nuclear_heat_data, cmap="plasma", origin='lower', alpha=0.7, interpolation='quadric',
                        extent=plot_extent, norm=colors.LogNorm(vmin=1e-4, vmax=1e2))
    cs1 = axs1[1].contour(nuclear_heat_data, np.logspace(-4, -1, 6), origin="lower",
                         extent=extent, cmap='winter', linewidths=1, norm=colors.LogNorm(vmin=1e-3, vmax=1e-1))
    cbar1 = fig2.colorbar(im1, ax=axs1[1], orientation="horizontal", shrink=1, format='%0.0e', pad=0.1)
    cbar1.set_label(r'Nuclear Heating $\left[\dfrac{W}{cm^3}\right]$', fontsize=16)
    cbar1.ax.tick_params(labelsize=10)
    plt.clabel(cs1, fmt='%0.0e', fontsize=9)

    axs1[2].imshow(background_image, extent=plot_extent)
    im2 = axs1[2].imshow(breeding_data, cmap="jet", origin='lower', alpha=0.8, interpolation='quadric',
                        extent=plot_extent, norm=colors.LogNorm(vmin=1e-2, vmax=1e3))
    #cs2 = axs[2].contour(breeding_data, origin="lower",
    #                     extent=extent, cmap='flag', linewidths=0.5)
    cbar2 = fig1.colorbar(im2, ax=axs1[2], orientation="horizontal", shrink=1, format='%0.0e', pad=0.1)
    cbar2.set_label(r'Tritium breeding rate $\left[\dfrac{g}{m^3yr}\right]$', fontsize=16)
    cbar2.ax.tick_params(labelsize=10)

    axs2[0].imshow(background_image, extent=plot_extent)
    im3 = axs2[0].imshow(dpa_data, cmap="gnuplot2", origin='lower', alpha=0.8, interpolation='quadric',
                        extent=plot_extent, norm=colors.LogNorm(vmin=5e-2, vmax=50))
    cs3 = axs2[0].contour(dpa_data, [0.1, 1, 10, 20], origin="lower",
                         extent=extent, cmap='hot', linewidths=0.5, norm=colors.LogNorm(vmin=5e-2, vmax=50))
    cbar3 = fig2.colorbar(im3, ax=axs2[0], orientation="horizontal", shrink=1, format='%0.0e', pad=0.1)
    cbar3.set_label(r"Fe Displacement Damage Rate $\left[\dfrac{DPA}{FPY}\right]$", fontsize=16)
    cbar3.ax.tick_params(labelsize=10)
    axs2[0].set_xlim((-100, 100))
    axs2[0].set_ylim((0, 400))
    plt.clabel(cs3, fmt='%0.0e', fontsize=9)

    axs2[2].imshow(background_image, extent=plot_extent)
    axs2[2].annotate("Mirror HTS Magnet",            [100, 460], [320, 450], fontsize=13, arrowprops=dict(arrowstyle='<|-', color='blue'))
    axs2[2].annotate("Water Cooled Tungsten Shield", [75,  350], [320, 340], fontsize=13, arrowprops=dict(arrowstyle='<|-', color='blue'))
    axs2[2].annotate("Liquid Immersion Blanket",     [200, 760], [320, 750], fontsize=13, arrowprops=dict(arrowstyle='<|-', color='blue'))
    axs2[2].annotate("Direct Convertor/Bias Rings",  [0,   880], [320, 870], fontsize=13, arrowprops=dict(arrowstyle='<|-', color='blue'))
    axs2[2].annotate("Tungsten Bioshield",           [240, 970], [320, 970], fontsize=13, arrowprops=dict(arrowstyle='<|-', color='blue'))
    axs2[2].annotate("RAFM First Wall",              [42,  120], [320, 110], fontsize=13, arrowprops=dict(arrowstyle='<|-', color='blue'))
    axs2[2].annotate("Central Coils",                [160, 225], [320, 220], fontsize=13, arrowprops=dict(arrowstyle='<|-', color='blue'))
    axs2[2].annotate("End Expander",                 [160, 610], [320, 600], fontsize=13, arrowprops=dict(arrowstyle='<|-', color='blue'))

    axs2[1].imshow(background_image, extent=plot_extent)
    im4 = axs2[1].imshow(wall_load_data, cmap="rainbow", origin='lower', alpha=0.8, interpolation='quadric',
                        extent=plot_extent, norm=colors.Normalize(vmin=1e-1, vmax=5))
    #cs4 = axs2[1].contour(np.multiply(heat, source_rate/volume), np.linspace(0, 10, 10), origin="lower",
    #                     extent=extent, cmap='flag', linewidths=0.5)
    cbar4 = fig2.colorbar(im4, ax=axs2[1], orientation="horizontal", shrink=1, format='%0.0f', pad=0.1)
    cbar4.set_label(r'Uncollided (Source) Neutron Flux $\left[\dfrac{MW}{m^2}\right]$', fontsize=16)
    cbar4.ax.tick_params(labelsize=10)
    axs2[1].set_xlim((-100, 100))
    axs2[1].set_ylim((0, 400))
    
    for ax in axs1:
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xlabel("x [cm]", fontsize=18)
        ax.set_ylabel("y [cm]", fontsize=18)

    for ax in axs2:
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xlabel("x [cm]", fontsize=18)
        ax.set_ylabel("y [cm]", fontsize=18)

    #plt.tight_layout()
    fig1.savefig("flux heat breed PBLI.png", dpi=400)

    fig2.savefig("DPA Wall Load Annotation PBLI.png", dpi=400)
    plt.show(fig2)
    plt.close()

    fig3, axs3 = plt.subplots(1, 2, layout="constrained")
    fig_scale = 2
    fig3.set_size_inches(8*fig_scale, 4*fig_scale)
    z_mesh = range(0, plot_extent[-1], 4)
    first_wall_r = int(40/(plot_extent[1]/100)+100)
    axs_dpa = axs3[0].twinx()
    axs3[0].plot(z_mesh, fast_flux_data[:, first_wall_r], label="Fast flux", color="blue")
    axs3[0].set_xlim(0, 400)
    axs3[0].set_ylabel(r"Fast Neutron Flux (>100keV) $\left[\dfrac{n}{cm^2s}\right]$", color="blue")
    axs_dpa.plot(z_mesh, dpa_data[:, first_wall_r], 'r-', label="DPA")
    axs_dpa.set_ylabel(r"Fe DPA rate $\left[\dfrac{DPA}{FPY}\right]$", color="red")
    axs3[0].set_xlabel("Distance from midplane (cm)")
    axs3[0].tick_params(axis='y', colors='blue')
    axs_dpa.tick_params(axis='y', colors='red')

    axs_heat = axs3[1].twinx()
    axs3[1].plot(z_mesh, wall_load_data[:, first_wall_r], 'g-', label="Wall load", color="green")
    axs3[1].set_xlim(0, 400)
    axs3[1].set_ylabel(r"Wall load $\left[\dfrac{MW}{m^2}\right]$", color="green")
    axs_heat.plot(z_mesh, nuclear_heat_data[:, first_wall_r], label="Heating", color="orange")
    axs_heat.set_ylabel(r"Nuclear Heating $\left[\dfrac{W}{cm^3}\right]$", color="orange")
    axs3[1].set_xlabel("Distance from midplane (cm)")
    axs3[1].tick_params(axis='y', colors='green')
    axs_heat.tick_params(axis='y', colors='orange')

    fig3.suptitle("Neutronics Parameters At First Wall \n PbLi blanket, He-cooled EUROFER-97 First Wall \n 10 MW DT Source")
    
    plt.close()
    
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MultipleLocator as ML
    from scipy.interpolate import griddata
    z_mesh = range(0, plot_extent[-1], 4)
    first_wall_r = int(40/(plot_extent[1]/100)+100)
    # Version 1, black and white figure

    # units for figsize are in inches. I generally oversize (4,3) for a 
    # half column width plot, or (7,3) for a full column width plot. 
    # This keeps the fontsize from changing wildly from show() to savefig()
    fig = plt.figure(figsize=(4,2)) 

    # set boundaries for axes
    plt.subplots_adjust(left=.07,right=.98,top=.98,bottom=.12)

    # create axes using GridSpec
    gs = GridSpec(100,100)

    axL = fig.add_subplot(gs[:,:45])
    axR = fig.add_subplot(gs[:,55:])
    z = np.divide(z_mesh, 100)
    nuFus = fast_flux_data[:, first_wall_r]
    feDPA = dpa_data[:, first_wall_r]
    wallLoad = wall_load_data[:, first_wall_r]
    nuclHeating = nuclear_heat_data[:, first_wall_r]

    # Doug's favorite named colors for R,G,B,C,M,Y,K
    # clrs = ['firebrick','forestgreen','royalblue','turquoise','orchi','gold','darkslategrey'] # These are not colorblind friendly

    l1, = axL.plot(z,nuFus/1e14,c='k')#,label='$\Gamma^n_{fast})\ [\\frac{n}{cm^2 s}]$')
    l2, = axL.plot(z,feDPA,c='k',ls='--') #,label='Fe DPA rate [$\\frac{DPA}{FPY}$]')
    leg1 = axL.legend([l1],['Fast Neutron Flux\n$[10^{14}\\frac{n}{cm^{2} s}]$'],loc='lower left',framealpha=.75,handlelength=1,fontsize=9)
    leg2 = axL.legend([l2],['Fe DPA rate\n   $[{DPA}/{FPY}]$'],loc='upper right',framealpha=.5,handlelength=1)
    axL.add_artist(leg1) # pyplot resists adding two legends. This is the workaround. 
    axL.add_artist(leg2)


    l3, = axR.plot(z,wallLoad,c='k')
    l4, = axR.plot(z,nuclHeating,c='k',ls='--')
    leg3 = axR.legend([l3],['Wall Load [$MW/m^{2}$]'],loc='lower left',framealpha=.75,handlelength=1,fontsize=9)
    leg4 = axR.legend([l4],['Nucl. Heating\n        $[W/cm^3]$'],loc='upper right',framealpha=.5,handlelength=1)
    axR.add_artist(leg3)
    axR.add_artist(leg4)

    axL.set_xlim((z[0],z[-1])) # override matplotlib default settings

    axL.xaxis.set_minor_locator(ML(0.25)) # Minor ticks should be 4-10 times as numerous
    #axL.xaxis.set_major_locator(ML(1)) # Major ticks should be 3 - 5 on an axis. 

    axL.set_xticks([0,1,2,3,4]); # set_xticks overrides xaxis.set_major_locator

    axL.set_xticklabels(['0','','','','4'])  # Min 2 numbers on a axis, make it easy to read


    # rather than use set_xlabel, I use ax.text for better adjustment. Allows for tighter borders
    axL.text(.5, -.05, 'Z [m]', transform=axL.transAxes, # transform=ax.transAxes puts the input coords in [0,1] domain
        ha='center', va='top')

    # Repeat for right axis
    axR.set_xlim((z[0],z[-1]))
    axR.xaxis.set_minor_locator(ML(0.25)) #
    axR.set_xticks([0,1,2,3,4]);
    axR.set_xticklabels(['0','','','','4'])  #
    axL.text(.5, -.05, 'Z [m]', transform=axR.transAxes, 
        ha='center', va='top')


    # Now for y-axes
    axL.set_ylim((0,13))
    axL.yaxis.set_minor_locator(ML(1))
    axL.yaxis.set_major_locator(ML(5))
    axL.tick_params(axis='both',which='both',right=True)


    axR.set_ylim((0,6))
    axR.yaxis.set_minor_locator(ML(1))
    axR.set_yticks([0,5])
    axR.tick_params(axis='both',which='both',right=True)


    figname = 'plots/openMC_exampleFig.png'
    plt.savefig(figname,dpi=300); print('Figure saved as ',figname)

    plt.show()
    plt.close()
    # Version 2, with yaxis labels and two colors, both of which are readable 
    fig4 = plt.figure()
    fig4.set_size_inches(4.5*2.54, 2*2.54)

    plt.subplots_adjust(left=.07,right=.93,top=.96,bottom=.12)
    gs = GridSpec(100,100)

    axL = fig4.add_subplot(gs[:,:42])
    axR = fig4.add_subplot(gs[:,58:])
    z_mesh = np.divide(z_mesh, 100)

    axL.plot(z_mesh,fast_flux_data[:, first_wall_r]/1e14,c='firebrick')#,label='$\Gamma^n_{fast})\ [\\frac{n}{cm^2 s}]$')
    axLL = axL.twinx()
    axLL.plot(z_mesh, dpa_data[:, first_wall_r],c='royalblue') #,label='Fe DPA rate [$\\frac{DPA}{FPY}$]')
    #leg1 = axL.legend([l1],['Fast Neutron Flux\n$[10^{14}\\frac{n}{cm^{2} s}]$'],loc='lower left',framealpha=.75,handlelength=1,fontsize=9)
    #leg2 = axL.legend([l2],['Fe DPA rate\n   $[{DPA}/{FPY}]$'],loc='upper right',framealpha=.5,handlelength=1)
    #axL.add_artist(leg1) # pyplot resists adding two legends. This is the workaround. 
    #axL.add_artist(leg2)


    axR.plot(z_mesh, wall_load_data[:, first_wall_r],c='firebrick')
    axRR = axR.twinx()
    axRR.plot(z_mesh, nuclear_heat_data[:, first_wall_r],c='royalblue')

    #leg3 = axR.legend([l3],['Wall Load [$MW/m^{2}$]'],loc='lower left',framealpha=.75,handlelength=1,fontsize=9)
    #leg4 = axR.legend([l4],['Nucl. Heating\n        $[W/cm^3]$'],loc='upper right',framealpha=.5,handlelength=1)
    #axR.add_artist(leg3)
    #axR.add_artist(leg4)

    axL.set_xlim((z_mesh[0],4)) # override matplotlib default settings
    axL.xaxis.set_minor_locator(ML(0.25)) # Minor ticks should be 4-10 times as numerous
    axL.set_xticks([0,1,2,3,4]); # set_xticks overrides xaxis.set_major_locator
    axL.set_xticklabels(['0','','','','4'])  # Min 2 numbers on a axis, make it easy to read
    axL.text(.5, -.05, 'Z [m]', transform=axL.transAxes, # transform=ax.transAxes puts the input coords in [0,1] domain
        ha='center', va='top')

    axR.set_xlim((z_mesh[0],4))
    axR.xaxis.set_minor_locator(ML(0.25)) #
    axR.set_xticks([0,1,2,3,4]);
    axR.set_xticklabels(['0','','','','4'])  #
    axL.text(.5, -.05, 'Z [m]', transform=axR.transAxes, 
        ha='center', va='top')


    # Now for y-axes
    axL.set_ylim((0,1.05*np.amax(fast_flux_data[:, first_wall_r]/1e14)))
    axLL.set_ylim((0,1.05*np.amax(dpa_data[:, first_wall_r])))
    axR.set_ylim((0,1.05*np.amax(wall_load_data[:, first_wall_r])))
    axRR.set_ylim((0,1.05*np.amax(nuclear_heat_data[:, first_wall_r])))

    axL.yaxis.set_minor_locator(ML(.5))
    axL.set_yticks([0,7])
    axL.text(-.03,.5,'Neutron Flux [$\\frac{10^{14} n}{cm^2 s}$]',transform=axL.transAxes,
        ha='right',va='center',rotation=90,color='firebrick')

    axLL.yaxis.set_minor_locator(ML(1))
    axLL.set_yticks([0,10,12]); axLL.set_yticklabels(['0','','12'])
    axLL.text(1.05,.45,'Fe DPA Rate [$\\frac{DPA}{FPY}$]',transform=axLL.transAxes,
        ha='left',va='center',rotation=90,color='royalblue')

    axR.yaxis.set_minor_locator(ML(.2))
    axR.set_yticks([0,1])
    axR.text(-.02,.5,'Wall Load [$\\frac{MW}{m^2}$]',transform=axR.transAxes,
        ha='right',va='center',rotation=90,color='firebrick')


    axRR.yaxis.set_minor_locator(ML(.5))
    axRR.set_yticks([0,5,6]); axRR.set_yticklabels(['0','','6'])
    axRR.text(1.05,.5,'Nucl. Heating [$\\frac{W}{cm^3}$]',transform=axRR.transAxes,
        ha='left',va='center',rotation=90,color='royalblue')

    #axL.yaxis.set_minor_locator(ML(1))
    #axL.yaxis.set_major_locator(ML(5))
    #axL.tick_params(axis='both',which='both',right=True)

    #axR.set_ylim((0,6))
    #axR.yaxis.set_minor_locator(ML(1))
    #axR.set_yticks([0,5])
    #axR.tick_params(axis='both',which='both',right=True)


    figname = 'plots/openMC_FWFig2.png'
    plt.savefig(figname,dpi=300); print('Figure saved as ',figname)

    plt.show()
    plt.close()
# %%