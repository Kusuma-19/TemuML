import os
import pandas as pd
import numpy as np
from rdkit.Chem import AllChem as Chem
from mordred import Calculator, descriptors

def process_df(df, smiles_col, ic50_col, ic50_units, units):
    df = df[[smiles_col, ic50_col, ic50_units]] #select only the smiles and IC50 columns
    df = df.dropna() #drop missing values
    df = df[df[ic50_units] == units] #select only the rows with the specified units
    df = df.drop(ic50_units, axis=1) #drop IC50 units column
    df.index.name = "LigandID" #rename index
    return df
def check_missing_smiles(df, smiles_col):
    def create_molecule(smiles):
        try:
            return Chem.MolFromSmiles(smiles)
        except:
            return None
    df['Molecule'] = df[smiles_col].apply(create_molecule)
    #print missing smiles
    print(df[df['Molecule'].isna()].index)
    #drop column Molecule
    df = df.drop('Molecule', axis=1)
    return df
def nanomolarconversion(df, ic50_col):
    import numpy as np
    #convert IC50 to float
    df[ic50_col] = df[ic50_col].astype(float)
    df[ic50_col] = df[ic50_col].div(1000000000) #convert nM to M
    return df 
def calculate_pic50(df, ic50_col):
    import numpy as np
    #change nM to M and calculate log
    df['pIC50'] = np.log10(df[ic50_col]) #take log of IC50
    df['pIC50'] = df['pIC50'].multiply(-1) #multiply by -1
    #round pIC50 to 2 decimal places
    df['pIC50'] = df['pIC50'].round(2)
    #drop IC50 column
    df = df.drop(ic50_col, axis=1)
    return df

def canonical_smiles(df, smiles_col):
    #import RDKit
    from rdkit.Chem import AllChem as Chem
    #generate canonical smiles
    df['canonical_smiles'] = df[smiles_col].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x), isomericSmiles=True))
    return df

def has_carbon_atoms(smiles):
    #import RDKit
    from rdkit.Chem import AllChem as Chem
    #check if molecule has carbon atoms
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        carbon_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6]
        return len(carbon_atoms) > 0
    return False

def remove_inorganic(df, smiles_col):
    df['has_carbon'] = df[smiles_col].apply(has_carbon_atoms)
    df = df[df['has_carbon'] == True]
    df = df.drop('has_carbon', axis=1)
    return df

def remove_mixtures(df, smiles_col):
    #import RDKit
    from rdkit.Chem import AllChem as Chem
    #check if molecule is a mixture using '.' as separator
    df['is_mixture'] = df[smiles_col].apply(lambda x: '.' in x)
    df = df[df['is_mixture'] == False]
    df = df.drop('is_mixture', axis=1)
    return df

def process_duplicates(df, smiles_col, pic50_col, threshold=0.2):
    import os
    df = df.reset_index(level=0) #reset index
    #first get the duplicate smiles
    duplicates = df[df.duplicated(subset=smiles_col, keep=False)].sort_values(smiles_col)
    #create array for storing index of rows to remove and rows to average
    to_remove = []
    to_average = []
    #create a loop to check the difference in pIC50 values
    for _, group in duplicates.groupby(smiles_col):
        pic50_diff = abs(group[pic50_col].max() - group[pic50_col].min())
        #if the difference is less than threshold, add the index to to_average
        if round(pic50_diff,2) <= threshold: #we need to round the difference to 2 decimal places otherwise it will not work
            to_average.append(group.index)
        else:
            to_remove.extend(group.index) #if the difference is greater than threshold, add the index to to_remove
    print(f"Save {len(to_remove)} duplicate SMILES with large pIC50 differences to save_duplicate_smiles.csv")     
    # Drop rows with pIC50 differences greater than threshold
    duplicates_high = df.loc[to_remove]
    duplicates_high.to_csv(os.path.join('datasets', 'processed','save_duplicate_smiles.csv'))
    # Drop rows with pIC50 differences greater than threshold
    df = df.drop(index=to_remove)
    print(f"Average {len(to_average)} duplicate SMILES with pIC50 differences less than {threshold}")
    # Average pIC50 values and retain one entry for duplicates with pIC50 differences less than or equal to threshold
    for indices in to_average:
        avg_pic50 = df.loc[indices, pic50_col].mean()
        df = df.drop(index=indices[1:]) #Remove all the rows in the DataFrame df with indices present in to_average, except for the first index. 
        #indices[1:] selects all indices except the first one.
        df.loc[indices[0], pic50_col] = avg_pic50 #set the pIC50 value to the average value to the average value for the first index in to_average

    df = df.set_index('LigandID') #set index back to the original index
    return df

def remove_missingdata(df):
    #drop missing data
    df = df.dropna()
    return df

def save_duplicate_smiles(df1, df2, smiles_col):
    intersect_indice = df1[smiles_col].isin(df2[smiles_col])
    #save the intersection
    dup_df1 = df1.loc[intersect_indice]
    return dup_df1

def morded_cal(df, smiles_col):
    #import RDKit
    from rdkit.Chem import AllChem as Chem
    from rdkit import Chem
    from mordred import Calculator, descriptors
    #calculate descriptors
    calc = Calculator(descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(smi) for smi in df[smiles_col]]
    des = calc.pandas(mols)
    des = des.set_index(df.index)
    return des

def remove_constant_string_des(df):
    #delete string value
    df = df.select_dtypes(exclude=['object'])
    #delete constant value
    for column in df.columns:
        if df[column].nunique() == 1:  # This checks if the column has only one unique value
            df = df.drop(column, axis=1)  # This drops the column from the DataFrame
    return df

def compute_IUPAC_Name(df, smiles_column):
    import cirpy
    iupac_names = [] 
    for smiles in df[smiles_column]:
        try:
            iupac_name = cirpy.resolve(smiles, 'iupac_name')
            print(f"'{smiles}': {iupac_name}")
        except Exception as e:
            print(f"Error with SMILES '{smiles}': {e}")
            iupac_name = ''  # or use a placeholder like 'Error' or 'Unavailable'
        iupac_names.append(iupac_name)

    # Add iupac_name to the dataframe
    df['IUPAC_Name'] = iupac_names
    print("Processing finished.")
    return df

def remove_highly_correlated_features(df, threshold=0.7):
    import pandas as pd
    import numpy as np
    # Compute pairwise correlation of columns
    corr_matrix = df.corr().abs()
    # Create a mask for the upper triangle
    upper = corr_matrix.where(
        pd.DataFrame(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool), 
                     index=corr_matrix.index, columns=corr_matrix.columns)
    )
    # Identify columns to drop based on threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    # Drop the columns from the DataFrame
    df_dropped = df.drop(columns=to_drop)
    return df_dropped