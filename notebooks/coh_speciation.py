#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
import numpy as np
import pandas as pd
from atmodeller import (
    debug_logger,
    earth_oceans_to_hydrogen_mass
)

from jaxtyping import ArrayLike

from atmodeller.classes import EquilibriumModel

from atmodeller.containers import (
    Planet,
    ChemicalSpecies,
    FixedFugacityConstraint,
    SpeciesNetwork,
    ThermodynamicState,
)

from atmodeller.output import Output
from molmass import Formula

from atmodeller.interfaces import FugacityConstraintProtocol, ThermodynamicStateProtocol
from atmodeller.eos import get_eos_models

logger = debug_logger()
logger.setLevel(logging.INFO)


# In[2]:


eos_models = get_eos_models()

#defining the gas species
H2_g = ChemicalSpecies.create_gas("H2", activity=eos_models['H2_zhang09'])
H2O_g= ChemicalSpecies.create_gas("H2O", activity=eos_models['H2O_zhang09'])
O2_g = ChemicalSpecies.create_gas("O2", activity=eos_models['O2_zhang09'])
CH4_g = ChemicalSpecies.create_gas("CH4", activity=eos_models['CH4_zhang09'])
C2H6_g = ChemicalSpecies.create_gas("C2H6", activity=eos_models['C2H6_zhang09'])
CO_g = ChemicalSpecies.create_gas("CO", activity=eos_models['CO_zhang09'])
CO2_g = ChemicalSpecies.create_gas("CO2", activity=eos_models['CO2_zhang09'])

# Add graphite (crystalline carbon)
C_cr = ChemicalSpecies.create_condensed("C")

species = SpeciesNetwork((H2_g, H2O_g, O2_g, CH4_g, C2H6_g, CO_g, CO2_g, C_cr)) #making a network among these gases


# In[3]:


planet = Planet() #creating a standard earth sized planet
model = EquilibriumModel(species)


# In[35]:


# set up mass constraints for H and C
h_kg = earth_oceans_to_hydrogen_mass(1)
print('h_kg value is',h_kg)
n_H = h_kg / 0.001008 #molar mass in kg/mol
n_C_buffer = n_H *5e10
c_kg = n_C_buffer * 0.012011
# ### Play with c_kg if activity of graphite is not unity
# c_kg=10e10 * h_kg # arbitrarily set to a high value to always ensure activity of graphite = 1 
#c_kg=9.23e+21
print('c_kg value is',c_kg)
logger.info("Mass of carbon = %0.5e", c_kg)
logger.info("Mass of hydrogen = %0.5e", h_kg)


# In[37]:


# Reproduce Figure 3 of Zhang & Duan (2009)
# give O mass as a free parameter 
# ### play with the minimum and max O_kg if needed
#xo_values = np.linspace(0.01, 0.99,num=100)
#n_O_array = (xo_values * (n_H + n_C_buffer)) / (1 - xo_values)
#o_kgs = n_O_array * 0.015999
#o_kgs = np.logspace(np.log10(0.001*h_kg),np.log10(100*h_kg), num=100)
o_kgs = np.linspace(0.001*h_kg,100*h_kg, num=100)
#o_kgs = np.linspace(1.35e+20, 2.46e+22, num=100)

log10fO2s = np.linspace(-18, -10, num=100)

print('o_kgs values are',o_kgs[0],o_kgs[99])
temperature_K = 1273 
pressure_bar = 2.4e4

state: ThermodynamicStateProtocol = ThermodynamicState(temperature_K, pressure_bar)

# Impose the fO2 and volatile masses as constraints
fugacity_constraints: dict[str, FugacityConstraintProtocol] = {
#    "O2_g": FixedFugacityConstraint(10**log10fO2s),
    "C_cd": FixedFugacityConstraint(1)
}

mass_constraints: dict[str, ArrayLike] = {
#    "C": c_kg, 
    "H": h_kg, 
    "O": o_kgs
    }

model.solve(
    state=state,
    fugacity_constraints=fugacity_constraints,
    mass_constraints=mass_constraints,
#    solver="basic",
)

output: Output = model.output
solution: dict[str, ArrayLike] = output.quick_look()

output.to_excel("COH_Fig3_ZD2009")


# In[30]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = "COH_Fig3_ZD2009.xlsx"
df_constraints = pd.read_excel(file_path, sheet_name='constraints')
df_raw = pd.read_excel(file_path, sheet_name='raw')
df_O2 = pd.read_excel(file_path, sheet_name='O2_g') 
df_H2=pd.read_excel(file_path, sheet_name='H2_g')
df_H2O=pd.read_excel(file_path, sheet_name='H2O_g')
df_CH4=pd.read_excel(file_path, sheet_name='CH4_g')
df_C2H6=pd.read_excel(file_path, sheet_name='C2H6_g')
df_CO=pd.read_excel(file_path, sheet_name='CO_g')
df_CO2=pd.read_excel(file_path, sheet_name='CO2_g')
df_C_cr=pd.read_excel(file_path, sheet_name='C_cd')
df_C=pd.read_excel(file_path, sheet_name='element_C')
df_H=pd.read_excel(file_path, sheet_name='element_H')
df_O=pd.read_excel(file_path, sheet_name='element_O')


# In[31]:


n_O = (df_H2O['gas_number'] + 2*df_CO2['gas_number'] + df_CO['gas_number'] + 2*df_O2['gas_number'])
n_H = (2*df_H2['gas_number'] + 2*df_H2O['gas_number'] + 4*df_CH4['gas_number'] + 6*df_C2H6['gas_number'])
n_C = (df_CH4['gas_number'] + 2*df_C2H6['gas_number']+df_CO2['gas_number']+df_CO['gas_number'])
#_O=df_C['gas_number']
#_H=df_H['gas_number']
#_O=df_O['gas_number']
print('n_O values are',n_O[0],n_O[99])
print('n_H values are', n_H[0],n_H[99])
print('n_C values are',n_C[0],n_C[99])
x_o = n_O / (n_O + n_H + n_C)
print(' x_o values are',x_o[0],x_o[99])
#n_O = df_constraints['O_number']
#n_H = df_constraints['H_number']
#n_C = df_raw['CH4_g']+2*df_raw['C2H6_g']+df_raw['CO_g']+df_raw['CO2_g']
#x_o = n_O / (n_O + n_H + n_C)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, 
                               gridspec_kw={'height_ratios': [1, 3]})
log_fo2 = np.log10(df_O2['fugacity'])
ax1.plot(x_o, log_fo2, color='black', lw=2)
ax1.set_ylabel('$Log f_{O2}$', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.set_title('Reproduction of Zhang & Duan (2009) Fig. 3')

df_gases = pd.DataFrame({
    'H2_g':   df_H2['gas_number'],
    'H2O_g':  df_H2O['gas_number'],
    'O2_g':   df_O2['gas_number'],
    'CH4_g':  df_CH4['gas_number'],
    'C2H6_g': df_C2H6['gas_number'],
    'CO_g':   df_CO['gas_number'],
    'CO2_g':  df_CO2['gas_number']
})
total_gas_moles = df_gases.sum(axis=1)
mole_fraction_pct = df_gases.div(total_gas_moles, axis=0) * 100

gas_columns = ['H2_g', 'H2O_g', 'O2_g', 'CH4_g', 'C2H6_g', 'CO_g', 'CO2_g']
labels = {'H2_g': 'H2', 'H2O_g': 'H2O', 'O2_g': 'O2', 'CH4_g': 'CH4', 
          'C2H6_g': 'C2H6', 'CO_g': 'CO', 'CO2_g': 'CO2'}

for col in gas_columns:
    ax2.plot(x_o, mole_fraction_pct[col], label=labels[col], lw=2)

ax2.set_xlabel('$X_O$ (Atomic Oxygen Fraction)', fontsize=12)
ax2.set_ylabel('Mole fraction(%)', fontsize=12)
ax2.set_ylim(0, 100)
ax2.set_xlim(0, 1)

for ax in [ax1, ax2]:
    ax.axvline(x=1/3, color='gray', linestyle='--', alpha=0.7, lw=1.5)


tick_positions = [0, 0.2, 1/3, 0.4, 0.6, 0.8, 1.0]
tick_labels = ['0', '0.2', '1/3', '0.4', '0.6', '0.8', '1.0']
ax2.set_xticks(tick_positions)
ax2.set_xticklabels(tick_labels)
ax2.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

plt.savefig("COH_Fig3_ZD2009.png", dpi=300, bbox_inches='tight')
plt.show()

