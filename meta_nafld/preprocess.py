#importing libraries
import pandas as pd
import numpy as np
from padelpy import padeldescriptor
from rdkit.Chem import AllChem as Chem
from rdkit import DataStructs
import os
import custom_preprocessing as cp
from glob import glob
from astartes.molecules import train_test_split_molecules

def canonical_smiles(df, smiles_column):
    df['canonical_smiles'] = df[smiles_column].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
    return df
def compute_fps(df, name, fol_path):
    xml_files = glob(os.path.join(name,"*.xml"))
    xml_files.sort()
    FP_list = [
    'AP2DC','AD2D','EState','CDKExt','CDK','CDKGraph','KRFPC','KRFP','MACCS','PubChem','SubFPC','SubFP']
    fp = dict(zip(FP_list, xml_files))
    print(fp)
    df['canonical_smiles'].to_csv(os.path.join(name, fol_path+'.smi'), sep='\t', index=False, header=False)
    #Calculate fingerprints
    for i in FP_list:
        padeldescriptor(mol_dir=os.path.join(name, fol_path+'.smi'),
                    d_file=os.path.join(name, fol_path,i+'.csv'),
                    descriptortypes= fp[i],
                    retainorder=True, 
                    removesalt=True,
                    threads=2,
                    detectaromaticity=True,
                    standardizetautomers=True,
                    standardizenitro=True,
                    fingerprints=True
                    )
        Fingerprint = pd.read_csv(os.path.join(name, fol_path,i+'.csv')).set_index(df.index)
        Fingerprint = Fingerprint.drop('Name', axis=1)
        Fingerprint.to_csv(os.path.join(name, fol_path,i+'.csv'))
        print(i+'.csv', 'done')
    #load at pc
    fp_at = pd.read_csv(os.path.join(name, fol_path,'AD2D.csv'    ) , index_col=0)
    fp_es = pd.read_csv(os.path.join(name, fol_path,'EState.csv'  ) , index_col=0)
    fp_ke = pd.read_csv(os.path.join(name, fol_path,'KRFP.csv'    ) , index_col=0)
    fp_pc = pd.read_csv(os.path.join(name, fol_path,'PubChem.csv' ) , index_col=0)
    fp_ss = pd.read_csv(os.path.join(name, fol_path,'SubFP.csv'   ) , index_col=0)
    fp_cd = pd.read_csv(os.path.join(name, fol_path,'CDKGraph.csv') , index_col=0)
    fp_cn = pd.read_csv(os.path.join(name, fol_path,'CDK.csv'     ) , index_col=0)
    fp_kc = pd.read_csv(os.path.join(name, fol_path,'KRFPC.csv'   ) , index_col=0)
    fp_ce = pd.read_csv(os.path.join(name, fol_path,'CDKExt.csv'  ) , index_col=0)
    fp_sc = pd.read_csv(os.path.join(name, fol_path,'SubFPC.csv'  ) , index_col=0)
    fp_ac = pd.read_csv(os.path.join(name, fol_path,'AP2DC.csv'   ) , index_col=0)
    fp_ma = pd.read_csv(os.path.join(name, fol_path,'MACCS.csv'   ) , index_col=0)
    
    fp_at.to_csv(os.path.join(name, fol_path ,'xat_'+fol_path+'.csv'))
    fp_es.to_csv(os.path.join(name, fol_path ,'xes_'+fol_path+'.csv'))
    fp_ke.to_csv(os.path.join(name, fol_path ,'xke_'+fol_path+'.csv'))
    fp_pc.to_csv(os.path.join(name, fol_path ,'xpc_'+fol_path+'.csv'))
    fp_ss.to_csv(os.path.join(name, fol_path ,'xss_'+fol_path+'.csv'))
    fp_cd.to_csv(os.path.join(name, fol_path ,'xcd_'+fol_path+'.csv'))
    fp_cn.to_csv(os.path.join(name, fol_path ,'xcn_'+fol_path+'.csv'))
    fp_kc.to_csv(os.path.join(name, fol_path ,'xkc_'+fol_path+'.csv'))
    fp_ce.to_csv(os.path.join(name, fol_path ,'xce_'+fol_path+'.csv'))
    fp_sc.to_csv(os.path.join(name, fol_path ,'xsc_'+fol_path+'.csv'))
    fp_ac.to_csv(os.path.join(name, fol_path ,'xac_'+fol_path+'.csv'))
    fp_ma.to_csv(os.path.join(name, fol_path ,'xma_'+fol_path+'.csv'))
    return fp_at, fp_es, fp_ke, fp_pc, fp_ss, fp_cd, fp_cn, fp_kc, fp_ce, fp_sc, fp_ac, fp_ma

def create_train_test_scaffold(df, smiles, acvalue_um, name, test_size):
    '''
    Create train test scaffold.
    -----
    Parameters:
    df: DataFrame
    smiles: smiles column name
    acvalue_um: Activity column
    test_size: Ratio of test size
    ----
    Return x_train, x_test, y_train, y_test
    '''
    x_train, x_test, y_train, y_test, train_index, test_index = train_test_split_molecules(molecules=df[smiles], y=df[acvalue_um], test_size=float(test_size),
    train_size=float(1.0-test_size), sampler="scaffold", random_state=0)
    #Dataframe
    x_train = pd.DataFrame(x_train, y_train.index, columns=[smiles])
    x_test  = pd.DataFrame(x_test,  y_test.index,  columns=[smiles])
    y_train = pd.DataFrame(y_train, y_train.index, columns=[acvalue_um])
    y_test  = pd.DataFrame(y_test, y_test.index,   columns=[acvalue_um])
    x_train.to_csv(os.path.join(name,"train", "x_train.csv" ))
    x_test .to_csv(os.path.join(name,"test" , "x_test.csv"  ))
    y_train.to_csv(os.path.join(name,"train", "y_train.csv" ))
    y_test .to_csv(os.path.join(name,"test" , "y_test.csv"  ))
    return x_train, x_test, y_train, y_test


def main():
    for i in ['temu']: 
        df = pd.read_csv(i+".csv", index_col=0)
        df = df.drop_duplicates(subset = ['canonical_smiles'], keep=False)
        print(i,":",  len(df))
        df.to_csv(os.path.join(i+".csv"))
        x_train, x_test, y_train, y_test = create_train_test_scaffold(df, "canonical_smiles", "class",i, test_size=0.3)
        y_train.to_csv
        print(i," train:",  len(x_train), " test:",  len(x_test))
        fp_at, fp_es, fp_ke, fp_pc, fp_ss, fp_cd, fp_cn, fp_kc, fp_ce, fp_sc, fp_ac, fp_ma = compute_fps(x_train, i, "train")
        fp_at, fp_es, fp_ke, fp_pc, fp_ss, fp_cd, fp_cn, fp_kc, fp_ce, fp_sc, fp_ac, fp_ma = compute_fps(x_test, i, "test")
if __name__ == "__main__":
    main()

    