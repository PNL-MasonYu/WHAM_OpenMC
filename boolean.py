from os import remove
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

divertor_coil = r[2102] &~ r[2101]

central_coil = r[2202] &~ r[2201]

cryostat = r[4001] &~ r[4002]
mid_cryostat = r[4101] &~ r[4102]
div_cryostat = r[4201] &~ r[4202]

shield = (r[1001] | r[1002]) &~ r[3001]
shield &= ~(divertor_coil)

first_wall = r[5001] | r[5002] | r[5003]

first_wall_support = r[5101] | r[5102] | r[5103]

breeder = r[6000] | r[6001] | r[6002] | r[6004] | r[6005] | r[6201]
# Subtract the region in the cylinder for best shielding
breeder &= ~(r[5201] | r[5203])
breeder &= ~(r[4101])
breeder &= ~(r[4201])

reflector = r[6100] | r[6102] | r[6203] | r[6204]

expander_tank = r[7001] | r[7002]

vacuum = r[1901] | r[1902] | (r[4002] &~ coil) | (r[4102] &~ central_coil) | (r[4202] &~ divertor_coil) | r[6901] | (r[7901] &~ r[7101]) | r[7902] | r[7903]

bias_rings = r[7101]

cryostat_shield = r[3001] &~ r[4001]

model_enclosure = -openmc.ZCylinder(0, 0, 399) & +openmc.ZPlane(0) & -openmc.ZPlane(400)

#c[1000] = openmc.Cell(1000, "Remaining vacuum", None, r[1000]&~model_enclosure)
c[1001] = openmc.Cell(1001, "Vacuum", m.vacuum, vacuum)
#c[1002] = openmc.Cell(1002, "Deuterium neutral gas fill", m.deuterium, d2_fill)

c[2001] = openmc.Cell(2001, "CFS coil", m.magnet, coil)
c[2101] = openmc.Cell(2101, "Divertor coil", m.magnet, divertor_coil)
c[2201] = openmc.Cell(2201, "Central coil", m.magnet, central_coil)
c[6000] = openmc.Cell(6000, "Breeder blanket", m.flibe, breeder)
c[3001] = openmc.Cell(3001, "Shield", m.tungsten, shield)
c[3002] = openmc.Cell(3002, "Cryostat shield", m.tungsten, cryostat_shield)
c[4001] = openmc.Cell(4001, "Cryostat", m.tungsten, cryostat)
c[4101] = openmc.Cell(4101, "Central Cryostat", m.tungsten, mid_cryostat)
c[4201] = openmc.Cell(4201, "Divertor Cryostat", m.tungsten, div_cryostat)
c[5000] = openmc.Cell(5000, "First Wall Cylinder", m.rafm_steel, r[5001])
c[5001] = openmc.Cell(5001, "First Wall Throat", m.rafm_steel, r[5002])
c[5002] = openmc.Cell(5002, "First Wall Expanding", m.rafm_steel, r[5003])
c[5100] = openmc.Cell(5100, "First Wall Cylinder structure", m.he_cooled_rafm, r[5101])
c[5101] = openmc.Cell(5101, "First Wall throat structure", m.tungsten, r[5102])
c[5102] = openmc.Cell(5102, "First Wall expanding structure", m.tungsten, r[5103])
c[5201] = openmc.Cell(5201, "Material testing - Central Cylinder", m.he_cooled_rafm, r[5201])
c[5203] = openmc.Cell(5203, "Material testing - Expanding", m.he_cooled_rafm, r[5203])
c[6100] = openmc.Cell(6100, "Breeder outer reflector", m.tungsten, reflector)
c[7000] = openmc.Cell(7000, "End expander tank", m.stainless, expander_tank)
c[7100] = openmc.Cell(7100, "Bias rings", m.rings, bias_rings)


all_cells = [c[key] for key in c.keys()]
root_cells.extend(all_cells)
root.add_cells(root_cells)

#root = remove_void(root)