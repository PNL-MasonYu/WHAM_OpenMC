import openmc
from openmc.data.photon import CM_PER_ANGSTROM
from geometry import r
import material as m

def elim(region, complement):
    for x in complement:
        region &= ~x
    return region

def remove_void(root):
    """
    function to remove all undefined voids in the model and replace them with vacuum
    use with caution - comes with performance penalties
    and will most likely cause excessive events from boundary crossings
    """
    remaining_vacuum = r[1000]
    all_cells = dict(root.get_all_cells())
    for cell in all_cells.values():
        remaining_vacuum &= ~cell.region
    c[9999] = openmc.Cell(9999, "Vacuum", m.vacuum, remaining_vacuum)

    root.add_cell(c[9999])
    return root

root = openmc.Universe(universe_id=0)

root_cells = []

# Dictionary containing all cells
c = {}

# Merge and split regions via boolean operations
coil = r[2002] &~ r[2001]

shield = (r[1001] | r[1002]) &~ r[3001] &~ r[5001] &~ r[5002] &~ r[5003]
shield &= ~r[5101] &~ r[5102] &~ r[5103]

crispy_shield = r[3001] &~ r[4001]

cryostat = r[4001] &~ r[4002]

first_wall = r[5001] | r[5002] | r[5003]

first_wall_support = r[5101] | r[5102] | r[5103]

breeder = r[6000] | r[6001] | r[6002]

expander_tank = r[7001] | r[7002]

vacuum = r[1901] | r[1902] | (r[4002] &~ coil) | r[6901] | r[7901] | r[7902] | r[7903]

c[1000] = openmc.Cell(1001, "Vacuum", m.vacuum, vacuum)
#c[1002] = openmc.Cell(1002, "Deuterium neutron gas fill", m.deuterium, d2_fill)
c[2001] = openmc.Cell(2001, "CFS coil", m.rebco, coil)
c[3001] = openmc.Cell(3001, "Shield", m.cooled_tungsten_carbide, shield)
c[3002] = openmc.Cell(3002, "Crispy shield", m.cooled_tungsten_carbide, crispy_shield)
#c[3011] = openmc.Cell(3011, "Crispy Mix shield", m.tungsten, outer_shield)
c[4001] = openmc.Cell(4001, "Cryostat", m.stainless, cryostat)
c[5000] = openmc.Cell(5000, "First Wall", m.tungsten, first_wall)
c[5100] = openmc.Cell(5100, "First Wall structure", m.rafm_steel, first_wall_support)
c[6000] = openmc.Cell(6000, "Breeder blanket", m.LiPb_breeder, breeder)
c[7000] = openmc.Cell(7000, "End expander tank", m.stainless, expander_tank)

all_cells = [c[key] for key in c.keys()]
root_cells.extend(all_cells)
root.add_cells(root_cells)

