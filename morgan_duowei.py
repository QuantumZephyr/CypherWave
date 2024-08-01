# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 16:44:14 2023

@author: Administrator
"""


import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# Read the Excel file containing SMILES information
data = pd.read_excel('D:/data.xlsx')

# Extract the SMILES columns for reactants and products
reactant_smiles = data['SMILES']
product_smiles = data['psmi']

# Specify the bit size for generating Morgan fingerprints
bit_sizes = [300]

# Create a new DataFrame to store fingerprint information
fingerprint_data = pd.DataFrame()

# Iterate over each bit size and generate fingerprints
for bit_size in bit_sizes:
    # Generate fingerprints for reactants
    reactant_fps = []
    for smile in reactant_smiles:
        mol = Chem.MolFromSmiles(smile)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=bit_size)
        reactant_fps.append(fp.ToBitString())
    
    # Generate fingerprints for products
    product_fps = []
    for smile in product_smiles:
        mol = Chem.MolFromSmiles(smile)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=bit_size)
        product_fps.append(fp.ToBitString())

    # Add the fingerprint data to the new DataFrame
    fingerprint_data[f'Reactant_{bit_size}'] = reactant_fps
    fingerprint_data[f'Product_{bit_size}'] = product_fps

# Save the fingerprint data to a new Excel file
fingerprint_data.to_excel('D:/scaffold/2.xlsx', index=False)
