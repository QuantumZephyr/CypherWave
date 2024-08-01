# Import necessary modules

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# Read the Excel file containing SMILES information, replace the corresponding data storage path when calculating.

data = pd.read_csv('F:/Activation energies2024/RGD1CHNO_smiles.csv')

# Extract the data from the reactant SMILES column; change to product data and rerun later.

smiles = data['reactant']  # Replace 'reactant' with the corresponding column name

# Create lists to store results

adj_matrices = []
topological_indices = []

# Create a loop to iterate over each SMILES

for smi in smiles:
    # Parse SMILES to a molecule object using RDKit
    mol = Chem.MolFromSmiles(smi)
    # Calculate the adjacency matrix
    adj_matrix = Chem.GetAdjacencyMatrix(mol)
    # Calculate topological indices
    topological_index_k_1 = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    topological_index_k_2 = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    topological_index_k_3 = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    topological_index_k_4 = rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)
    SSSR = Chem.GetSymmSSSR(mol)  # Use GetSSSR function to calculate the number of rings
    topological_index_k_5 = len(SSSR)
    topological_index_k_inf = rdMolDescriptors.CalcNumRotatableBonds(mol)
    # Store results
    adj_matrices.append(adj_matrix)
    topological_indices.append([topological_index_k_1, topological_index_k_2,
topological_index_k_3, topological_index_k_4,
topological_index_k_5, topological_index_k_inf])

# Create a DataFrame containing the results

result_df = pd.DataFrame({'SMILES': smiles, 'Adjacency Matrix': adj_matrices,
                          'K_1': [indices[0] for indices in topological_indices],
                          'K_2': [indices[1] for indices in topological_indices],
                          'K_3': [indices[2] for indices in topological_indices],
                          'K_4': [indices[3] for indices in topological_indices],
                          'K_5': [indices[4] for indices in topological_indices],
                          'K_Inf': [indices[5] for indices in topological_indices]})

# Save the results to a new Excel file, change the path based on the generated table name

result_df.to_excel('F:/Activation energies2024/reactant.xlsx', index=False)
