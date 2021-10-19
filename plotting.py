import openmc
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pickle

import material as m 
from geometry import r
from boolean import c
from boolean import root

# Colors of each material in plots
material_color = {m.vacuum: 'black', m.air: 'azure', m.aluminum_6061: 'lightgrey',
                  m.deuterium: 'grey', m.stainless: 'teal', m.tungsten: 'purple',
                  m.rebco: 'orange', m.magnet: 'orange', m.crispy: 'brown', m.rafm_steel: 'azure',
                  m.tungsten_carbide: 'olive', m.LiPb_breeder: 'grey', m.rings: 'teal',
                  m.cooled_tungsten_carbide: 'darkgreen', m.water: 'blue', m.he_cooled_rafm: 'green',
                  m.tungsten_boride: 'yellow', m.w2b5: 'yellow', m.TiH2: 'violet', m.TiH2: 'violet'}

for material in m.materials_list:
    if not material in material_color.keys():
        material_color[material] = 'green'

def voxel_plot(origin=(0, 0, 60), width=(250, 250, 320), pixels=(500, 500, 640)):
    voxel = openmc.Plot()
    voxel.type = 'voxel'
    voxel.origin = origin
    voxel.width = width
    voxel.pixels = pixels
    voxel.color_by = 'material'
    voxel.colors = material_color
    plots = openmc.Plots([voxel])
    return plots

def slice_plot(basis = 'yz', origin=(0, 0, 75), width=(150, 150), pixels=(2000, 2000), color=material_color, cwd='./outputs/slice'):
    plot = openmc.Plot()
    plot.basis = basis
    plot.origin = origin
    plot.width = width
    plot.pixels = pixels
    plot.color_by = 'material'
    plot.colors = color
    plot.show_overlaps = True
    plot.overlap_color = 'red'
    plot.background = 'magenta'
    plot.mask_background = 'black'
    plot.filename = cwd
    plots = openmc.Plots([plot])
    return plots

def plot_geometry(r_bounds, z_bounds):
    fig = plt.figure(num=1, figsize=(15, 10))
    plt.plot(r_bounds, z_bounds, "g-")
    plt.plot(-r_bounds, z_bounds, "g-")
    background_image = plt.imread('./slice.ppm')
    extent=(-275, 275, 0, 300)
    plt.imshow(background_image, extent=extent)
    plt.xlabel('y (cm)')
    plt.ylabel('z (cm)')
    plt.title("Model Geometry")
    fig.savefig('./plots/Model geometry')
    return fig