# Import necessary modules

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# Read the CSV file containing SMILES information, replace the corresponding data storage path when calculating.

data = pd.read_csv('F:/Activation energies2024/RGD1CHNO_smiles.csv')

# Extract the data from the product SMILES column; change to reactant data and rerun later.

smiles = data['product']  # Replace 'product' with the corresponding column name

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
    topological_index_k1 = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    topological_index_k2 = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    topological_index_k3 = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    topological_index_k4 = rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)
    SSSR = Chem.GetSymmSSSR(mol)  # Use GetSSSR function to calculate the number of rings
    topological_index_k5 = len(SSSR)
    topological_index_kinf = rdMolDescriptors.CalcNumRotatableBonds(mol)
    # Store results
    adj_matrices.append(adj_matrix)
    topological_indices.append([topological_index_k1, topological_index_k2,
                                topological_index_k3, topological_index_k4,
                                topological_index_k5, topological_index_kinf])

# Create a DataFrame containing the results

result_df = pd.DataFrame({'SMILES': smiles, 'Adjacency Matrix': adj_matrices,
                          'Kappa 1': [indices[0] for indices in topological_indices],
                          'Kappa 2': [indices[1] for indices in topological_indices],
                          'Kappa 3': [indices[2] for indices in topological_indices],
                          'Kappa 4': [indices[3] for indices in topological_indices],
                          'Kappa 5': [indices[4] for indices in topological_indices],
                          'Kappa Inf': [indices[5] for indices in topological_indices]})

# Save the results to a new Excel file, change the path based on the generated table name

result_df.to_excel('F:/Activation energies2024/product.xlsx', index=False)
