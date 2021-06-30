import openmc
from openmc.data.photon import CM_PER_ANGSTROM
from geometry import r
import material as m

def elim(region, complement):
    for x in complement:
        region &= ~x
    return region

root = openmc.Universe(universe_id=0)

root_cells = []

# Dictionary containing all cells
c = {}

#################Stainless Vacuum Chamber##################

coil = r[2002] &~ r[2001]

shield = r[1001] | r[1002] &~ r[4001]

cryostat = r[4001] &~ r[4002]

#c[1001] = openmc.Cell(1001, "Vacuum chamber", m.stainless, chamber)
#c[1002] = openmc.Cell(1002, "Deuterium neutron gas fill", m.deuterium, d2_fill)
c[2001] = openmc.Cell(2001, "CFS coil", m.rebco, coil)
c[3001] = openmc.Cell(3001, "Shield", m.tungsten, shield)
#c[3002] = openmc.Cell(3002, "Inner shield", m.tungsten, central_shield)
#c[3011] = openmc.Cell(3011, "Crispy Mix shield", m.tungsten, outer_shield)
c[4001] = openmc.Cell(4001, "Cryostat", m.stainless, cryostat)
#c[3021] = openmc.Cell(3021, "In-vessel shield", m.tungsten, inner_shield)

root_cells.extend([c[2001], c[3001], c[4001]])
root.add_cells(root_cells)

remaining_vacuum = r[1000]
all_cells = dict(root.get_all_cells())
for cell in all_cells.values():
    remaining_vacuum &= ~cell.region
c[9999] = openmc.Cell(9999, "Vacuum", m.vacuum, remaining_vacuum)

root.add_cell(c[9999])