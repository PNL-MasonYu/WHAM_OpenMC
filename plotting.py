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
                  m.deuterium: 'grey', m.stainless: 'teal', m.tungsten: 'green',
                  m.rebco: 'orange', m.crispy: 'brown', m.rafm_steel: 'azure', 
                  m.tungsten_carbide: 'darkgreen', m.LiPb_breeder: 'grey',
                  m.cooled_tungsten_carbide: 'darkgreen', m.water: 'blue'}

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

def slice_plot(basis = 'yz', origin=(0, 0, 78-26.51125), width=(150, 150), pixels=(1000, 1000), color=material_color, cwd='./outputs/slice'):
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
    plot.filename = cwd
    plots = openmc.Plots([plot])
    return plots
