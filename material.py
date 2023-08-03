# %%
import openmc
import matplotlib.pyplot as plt
from pkg_resources import ZipProvider

vacuum = openmc.Material(0, name='vacuum')
vacuum.set_density('g/cm3', 1e-40)
vacuum.add_nuclide('N14', 1)

air = openmc.Material(4, name='air')
air.set_density('g/cm3', 0.001205)
air.add_nuclide('C12', 1.48583129e-04)
air.add_nuclide('C13', 1.60703476e-06)
air.add_nuclide('N14', 7.81575372e-01)
air.add_nuclide('N15', 2.85532775e-03)
air.add_nuclide('O16', 2.10235865e-01)
air.add_nuclide('O17', 8.00842335e-05)
air.add_nuclide('O18', 4.32033365e-04)
air.add_nuclide('Ar36', 1.55828794e-05)
air.add_nuclide('Ar38', 2.93813883e-06)
air.add_nuclide('Ar40', 4.65260590e-03)

deuterium = openmc.Material(1, name='deuterium')
deuterium.set_density('kg/m3', 0.001)
deuterium.add_nuclide('H2', 1)

stainless = openmc.Material(298, name = 'stainless 304')
stainless.set_density('g/cm3', 8.0)
stainless.add_nuclide('C12', 1.81016539e-03)
stainless.add_nuclide('C13', 1.95782570e-05)
stainless.add_nuclide('Si28', 9.02042466e-03)
stainless.add_nuclide('Si29', 4.58244576e-04)
stainless.add_nuclide('Si30', 3.02431639e-04)
stainless.add_nuclide('P31', 4.07975189e-04)
stainless.add_nuclide('S32', 2.44140964e-04)
stainless.add_nuclide('S33', 1.92763157e-06)
stainless.add_nuclide('S34', 1.09232456e-05)
stainless.add_nuclide('S36', 2.57017543e-08)
stainless.add_nuclide('Cr50', 8.72312748e-03)
stainless.add_nuclide('Cr52', 1.68216831e-01)
stainless.add_nuclide('Cr53', 1.90744383e-02)
stainless.add_nuclide('Cr54', 4.74803142e-03)
stainless.add_nuclide('Mn55', 1.00006144e-02)
stainless.add_nuclide('Fe54', 4.03523669e-02)
stainless.add_nuclide('Fe56', 6.33445864e-01)
stainless.add_nuclide('Fe57', 1.46290275e-02)
stainless.add_nuclide('Fe58', 1.94685500e-03)
stainless.add_nuclide('Ni58', 5.89458369e-02)
stainless.add_nuclide('Ni60', 2.27057109e-02)
stainless.add_nuclide('Ni61', 9.87005296e-04)
stainless.add_nuclide('Ni62', 3.14709137e-03)
stainless.add_nuclide('Ni64', 8.01362752e-04)
#stainless.add_s_alpha_beta('c_Fe56')

aluminum_6061 = openmc.Material(13, name='aluminum_6061')
aluminum_6061.set_density('g/cm3', 2.7)
aluminum_6061.add_nuclide('Mg24', 8.81687922e-03)
aluminum_6061.add_nuclide('Mg25', 1.11620195e-03)
aluminum_6061.add_nuclide('Mg26', 1.22893835e-03)
aluminum_6061.add_nuclide('Al27', 9.77324713e-01)
aluminum_6061.add_nuclide('Si28', 5.34499966e-03)
aluminum_6061.add_nuclide('Si29', 2.71530132e-04)
aluminum_6061.add_nuclide('Si30', 1.79204092e-04)
aluminum_6061.add_nuclide('Ti46', 4.11473671e-05)
aluminum_6061.add_nuclide('Ti47', 3.71074438e-05)
aluminum_6061.add_nuclide('Ti48', 3.67682897e-04)
aluminum_6061.add_nuclide('Ti49', 2.69826977e-05)
aluminum_6061.add_nuclide('Ti50', 2.58355590e-05)
aluminum_6061.add_nuclide('Cr50', 4.42071668e-05)
aluminum_6061.add_nuclide('Cr52', 8.52491208e-04)
aluminum_6061.add_nuclide('Cr53', 9.66656598e-05)
aluminum_6061.add_nuclide('Cr54', 2.40621288e-05)
aluminum_6061.add_nuclide('Mn55', 4.34559057e-04)
aluminum_6061.add_nuclide('Fe54', 1.16134627e-04)
aluminum_6061.add_nuclide('Fe56', 1.82306529e-03)
aluminum_6061.add_nuclide('Fe57', 4.21025279e-05)
aluminum_6061.add_nuclide('Fe58', 5.60307355e-06)
aluminum_6061.add_nuclide('Cu63', 8.11849846e-04)
aluminum_6061.add_nuclide('Cu65', 3.62191869e-04)
aluminum_6061.add_nuclide('Zn64', 2.97894307e-04)
aluminum_6061.add_nuclide('Zn66', 1.68000999e-04)
aluminum_6061.add_nuclide('Zn67', 2.44761643e-05)
aluminum_6061.add_nuclide('Zn68', 1.11778523e-04)
aluminum_6061.add_nuclide('Zn70', 3.69565848e-06)
aluminum_6061.add_s_alpha_beta('c_Al27')

# This is basically YBCO for now
rebco = openmc.Material(name="REBCO tape")
rebco.set_density("g/cm3", 6.3)
rebco.add_element('Y', 7.6923076)
rebco.add_element('Ba', 15.3846153)
rebco.add_element('Cu', 23.0769230)
rebco.add_element('O', 53.8461538)

# Hastelloy C-276 subtrate, composition from Haynes International
hastelloy = openmc.Material(name="Hastelloy C-276")
hastelloy.set_density("g/cm3", 8.89)
hastelloy.add_element("Ni", 55, 'wo')
hastelloy.add_element("Co", 2.5, 'wo')
hastelloy.add_element("Cr", 16, 'wo')
hastelloy.add_element("Mo", 16, 'wo')
hastelloy.add_element("Fe", 5, 'wo')
hastelloy.add_element("W", 4, 'wo')
hastelloy.add_element("Mn", 1, "wo")
hastelloy.add_element("V", 0.35, "wo")
hastelloy.add_element("Cu", 0.15, "wo")

copper = openmc.Material(name="Copper")
copper.set_density("g/cm3", 8.96)
copper.add_element("Cu", 100)

magnet = openmc.Material.mix_materials([hastelloy, copper, rebco], [0.55, 0.43, 0.02], 'vo')

tungsten = openmc.Material(31, name="tungsten")
tungsten.set_density("g/cm3", 19.25)
tungsten.add_element("W", 100)

crispy = openmc.Material(81, name="crispy")
crispy.set_density('g/cm3', 1.4)
crispy.add_nuclide('H1', 6.72038450e-02)
crispy.add_nuclide('H2', 7.72933104e-06)
crispy.add_nuclide('B10', 1.36943828e-01)
crispy.add_nuclide('B11', 5.51216112e-01)
crispy.add_nuclide('C12', 2.26052782e-01)
crispy.add_nuclide('C13', 2.44492547e-03)
crispy.add_nuclide('O16', 1.34096500e-02)
crispy.add_nuclide('O17', 5.10807965e-06)
crispy.add_nuclide('O18', 2.75567455e-05)
crispy.add_nuclide('Cl35', 2.75567455e-05)
crispy.add_nuclide('Cl37', 6.51683424e-04)

# This water does not have the S_ab scattering kernels, meant for mixing
water = openmc.Material(101, name="water")
water.set_density('g/cm3', 1)
water.add_element('H', 66.666)
water.add_element('O', 33.333)

cooled_tungsten = openmc.Material.mix_materials([tungsten, water], [0.85, 0.15], 'vo')
cooled_tungsten.add_s_alpha_beta('c_H_in_H2O', 0.66666*0.15)

tungsten_carbide = openmc.Material(22, name = 'tungsten carbide')
tungsten_carbide.add_elements_from_formula('WC')
tungsten_carbide.set_density('g/cm3', 15.63)

rafm_steel = openmc.Material(900, name="EUROFER 97 RAFM Steel")
rafm_steel.add_element('Fe', 90)
rafm_steel.add_element('Cr', 9.21)
rafm_steel.add_element('C', 0.104)
rafm_steel.add_element('Mn', 0.502)
rafm_steel.add_element('V', 0.204)
rafm_steel.add_element('W', 1.148)
rafm_steel.add_element('Ta', 0.14)
rafm_steel.add_nuclide('N14', 0.0234)
rafm_steel.add_element('O', 0.001)
rafm_steel.add_element('P', 0.04)
rafm_steel.add_element('S', 0.004)
rafm_steel.add_element('B', 0.01)
rafm_steel.add_element('Ti', 0.004)
rafm_steel.add_element('Nb', 0.0012)
rafm_steel.add_element('Mo', 0.008)
rafm_steel.add_element('Ni', 0.0214)
rafm_steel.set_density('g/cm3', 7.798)
#rafm_steel.add_s_alpha_beta('c_Fe56', 0.9)

cooled_rafm_steel = openmc.Material.mix_materials([rafm_steel, water], [0.85, 0.15], 'vo')
cooled_rafm_steel.add_s_alpha_beta('c_H_in_H2O', 0.66666*0.15)

LiPb_breeder = openmc.Material(1000, name = "lead lithium eutectic breeder")
# Should double check this number... Varies with temperature
LiPb_breeder.set_density('kg/m3', 9865)
#LiPb_breeder.add_element('Li', 17, percent_type='ao', enrichment=90,
#                          enrichment_target='Li6', enrichment_type='ao')
LiPb_breeder.add_element('Li', 17, percent_type='ao')
LiPb_breeder.add_element('Pb', 83)

# Shield material with structural support and coolant mixed in
cooled_tungsten_carbide = openmc.Material.mix_materials([tungsten_carbide, water, rafm_steel], [0.35, 0.6, 0.05], 'vo')
cooled_tungsten_carbide.add_s_alpha_beta('c_H_in_H2O', 0.66666*0.1)

helium_8mpa = openmc.Material(1100, name="helium gas 8 MPa, 450C")
helium_8mpa.set_density("kg/m3", 5.323)
helium_8mpa.add_element("He", 100)

he_cooled_rafm = openmc.Material.mix_materials([helium_8mpa, rafm_steel], [0.45, 0.55], 'vo')
#print(he_cooled_rafm.density)

rings = openmc.Material.mix_materials([stainless, vacuum], [0.1, 0.9], 'vo')

tungsten_boride = openmc.Material(1200, name="tungsten boride WB")
tungsten_boride.set_density("g/cm3", 15.43)
tungsten_boride.add_elements_from_formula('WB')

WB2 = openmc.Material(1202, name="tungsten boride WB2")
WB2.set_density("g/cm3", 12.42)
WB2.add_elements_from_formula("WB2")

w2b5 = openmc.Material(1201, name="tungsten boride W2B5")
w2b5.set_density("g/cm3", 12.91)
w2b5.add_elements_from_formula("W2B5")

cooled_w2b5 = openmc.Material.mix_materials([w2b5, water], [0.9, 0.1], 'vo')

TiH2 = openmc.Material(1300, name="titanium hydride")
TiH2.set_density("g/cm3", 3.75)
TiH2.add_elements_from_formula("TiH2")

zirconium_hydride = openmc.Material(1400, name="zirconium hydride ZrH2")
zirconium_hydride.set_density("g/cm3", 5.56)
zirconium_hydride.add_elements_from_formula("ZrH2")

tantalum = openmc.Material(name="pure tantalum")
tantalum.set_density("g/cm3", 16.69)
tantalum.add_element("Tantalum", 100)

cooled_TiH2 = openmc.Material.mix_materials([TiH2, water], [0.95, 0.05], 'vo')

flibe = openmc.Material(name="Pure FLiBe molten salt")
flibe.set_density("kg/m3", 1940)
flibe.add_elements_from_formula("Li2BeF4")

beryllium = openmc.Material(name="Pure Beryllium")
beryllium.set_density("g/cm3", 1.85)
beryllium.add_element("Be", 100)

tantalum_hydride_55 = openmc.Material(name='tantalum hydride, TaH0.55')
tantalum_hydride_55.set_density("g/cm3", 16.69)
tantalum_hydride_55.add_element("Tantalum", 1/1.55)
tantalum_hydride_55.add_element("Hydrogen", 0.55/1.55)
tantalum_hydride_55.temperature = 500.0

tantalum_hydride_46 = openmc.Material(name='tantalum hydride, TaH0.46')
tantalum_hydride_46.set_density("g/cm3", 16.69)
tantalum_hydride_46.add_element("Tantalum", 1/1.46)
tantalum_hydride_46.add_element("Hydrogen", 0.46/1.46)
tantalum_hydride_46.temperature = 500.0

tantalum_hydride_39 = openmc.Material(name='tantalum hydride, TaH0.39')
tantalum_hydride_39.set_density("g/cm3", 16.69)
tantalum_hydride_39.add_element("Tantalum", 1/1.39)
tantalum_hydride_39.add_element("Hydrogen", 0.39/1.39)
tantalum_hydride_39.temperature = 500.0

tantalum_hydride_30 = openmc.Material(name='tantalum hydride, TaH0.30')
tantalum_hydride_30.set_density("g/cm3", 16.69)
tantalum_hydride_30.add_element("Tantalum", 1/1.3)
tantalum_hydride_30.add_element("Hydrogen", 0.3/1.3)
#tantalum_hydride_30.temperature = 500.0

Nak_77 = openmc.Material(name="NaK eutectic, 550C")
Nak_77.set_density("g/cm3", 0.749)
Nak_77.add_element("Na", 23, 'wo')
Nak_77.add_element("K", 77, 'wo')
Nak_77.temperature = 900
Nak_77.depletable = True

potassium = openmc.Material(name="K molten, 550C")
potassium.set_density("g/cm3", 0.82948 )
potassium.add_element("K", 100, 'ao')
potassium.temperature = 900
potassium.depletable = True

KCl = openmc.Material(name="KCl molten, 977C")
KCl.set_density("g/cm3", 1.527)
KCl.add_elements_from_formula("KCl", enrichment=90, enrichment_target="Cl37")
#KCl.add_elements_from_formula("KCl")
KCl.temperature = 1200
KCl.depletable = True

iron = openmc.Material(name='pure iron')
iron.set_density("g/cm3", 7.874)
iron.add_element("Fe", 100, 'ao')

LiH = openmc.Material(name='Lithium Hydride LiH breeder')
LiH.set_density("g/cm3", 0.775)
LiH.add_element("Li", enrichment=4.85, percent=50, enrichment_target='Li6', enrichment_type='ao')
LiH.add_element("H", 50, 'ao')
LiH.temperature = 900

LiD = openmc.Material(name='Lithium Deuteride LiD breeder')
LiD.set_density("g/cm3", 0.885)
LiD.add_element("Li", enrichment=4.85, percent=50, enrichment_target='Li6', enrichment_type='ao')
LiD.add_element("H", enrichment=100, percent=50, enrichment_target='H2', enrichment_type='ao')
LiD.temperature = 900

lead = openmc.Material(name='pure lead')
lead.set_density("g/cm3", 11.34)
lead.add_element("Pb", 100)
lead.temperature = 900

lithium = openmc.Material(name='molten lithium')
lithium.set_density("g/cm3", 0.512)
lithium.add_element("Li", 100)
lithium.temperature = 900

HfH2 = openmc.Material(name="Hafnium hydride HfH2")
HfH2.set_density("g/cm3", 11.36)
HfH2.add_elements_from_formula("HfH2")

titanium = openmc.Material(name="pure titanium")
titanium.set_density("g/cm3", 4.506)
titanium.add_element("Ti", 100)

MgO = openmc.Material(name="Magnesium oxide MgO")
MgO.set_density("g/cm3", 3.58)
MgO.add_elements_from_formula("MgO")

MgO_HfH2 = openmc.Material.mix_materials([MgO, HfH2, vacuum], [0.4, 0.4, 0.2], 'vo', name="MgO 40% vo, HfH2 40% vo, void 20% vo, Snead alloy 1")

Fe_HfH2_WB2 = openmc.Material.mix_materials([iron, HfH2, WB2, vacuum], [0.54, 0.34, 0.07, 0.05], 'vo', name="Fe 54 vol % HfH2 34 vol % WB2 7 vol % remainder void, Snead alloy 2")

Ti_HfH2 = openmc.Material.mix_materials([titanium, HfH2, vacuum], [0.5, 0.48, 0.02], 'vo', name="Ti 50 vol % HfH2 48vol % remainder void, Snead alloy 3")

B4C = openmc.Material(name="Boron carbide B4C")
B4C.set_density("g/cm3", 2.5)
B4C.add_elements_from_formula("B4C")

materials_list = [vacuum, air, deuterium, aluminum_6061, stainless, beryllium, lead, LiH, LiD,
                  rebco, magnet, tungsten, crispy, water, he_cooled_rafm, iron, lithium,
                  cooled_tungsten, tungsten_carbide, cooled_tungsten_carbide,
                  rafm_steel, LiPb_breeder, rings, tungsten_boride, WB2, w2b5, cooled_w2b5,
                  TiH2, cooled_TiH2, zirconium_hydride, copper, hastelloy, flibe, tantalum,
                  tantalum_hydride_55, tantalum_hydride_30, cooled_rafm_steel, Nak_77, potassium, KCl,
                  HfH2, titanium, MgO, MgO_HfH2, Fe_HfH2_WB2, Ti_HfH2, B4C]
materials = openmc.Materials(materials_list)
# %%
"""
fig=openmc.plot_xs(tungsten, ['capture'])
openmc.plot_xs(tungsten_carbide, ["capture"], axis=fig.get_axes()[0])
openmc.plot_xs(TiH2, ['capture'], axis=fig.get_axes()[0])
openmc.plot_xs(LiD, ["capture"], axis=fig.get_axes()[0])
openmc.plot_xs(LiH, ["capture"], axis=fig.get_axes()[0])
#openmc.plot_xs(tantalum_hydride, ["damage"], axis=fig.get_axes()[0])
plt.title("Neutron capture Macroscopic Cross Section")
plt.xlabel("Energy (eV)")
plt.xlim([1e5, 20e6])
plt.ylim([1e-5, 1e0])
plt.legend(["tungsten", "tungsten carbide", "TiH2", 'LiD', 'LiH'], loc='upper left')
#plt.xscale("linear")
#plt.ylim([1e-1, 5])
#plt.yscale("linear")
fig.savefig("./plots/W vs hydrides capture")
plt.show()
"""
# %%
