from rdkit import Chem

# Input: SMILES string
smiles = "CCOC[C@]12CCC(=O)C[C@@H]1CC[C@@H]1[C@@H]2CC[C@]2(C)[C@@H](O)CC[C@@H]12" 

# Convert SMILES to RDKit Molecule object
mol = Chem.MolFromSmiles(smiles)

# Check if the conversion was successful
if mol:
    # Output: Save as MOL file
    with open("output.mol", "w") as f:
        f.write(Chem.MolToMolBlock(mol))
    print("MOL file saved as output.mol")
else:
    print("Invalid SMILES string")
