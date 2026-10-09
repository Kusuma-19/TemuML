"""
Cleaned dataset -> scaffold split -> PaDEL fingerprints.

Input : <name>.csv   (index = Molecule ChEMBL ID, columns: canonical_smiles, class, ...)
Output layout:
    <name>/
        <name>.csv                 full cleaned dataset
        train/  x_train.csv, y_train.csv, xat_train.csv, xes_train.csv, ... (12 fingerprint files)
        test/   x_test.csv,  y_test.csv,  xat_test.csv,  xes_test.csv,  ...
        train.smi, test.smi        (PaDEL input)
PaDEL XML descriptor configs are expected in <name>/*.xml (as in your original code).
"""
import os
from glob import glob

import numpy as np
import pandas as pd
from padelpy import padeldescriptor
from astartes.molecules import train_test_split_molecules

SMILES_COL = "canonical_smiles"
LABEL_COL = "class"
TEST_SIZE = 0.3
KEEP_RAW_PADEL_CSV = False   # True keeps PaDEL's raw <FP>.csv next to the processed files

# Order must match the alphabetically sorted XML files (same assumption as your original code)
FP_LIST = ['AP2DC', 'AD2D', 'EState', 'CDKExt', 'CDK', 'CDKGraph',
           'KRFPC', 'KRFP', 'MACCS', 'PubChem', 'SubFPC', 'SubFP']

# output file prefix per fingerprint (x??_train.csv / x??_test.csv)
FP_CODE = {'AD2D': 'xat', 'EState': 'xes', 'KRFP': 'xke', 'PubChem': 'xpc',
           'SubFP': 'xss', 'CDKGraph': 'xcd', 'CDK': 'xcn', 'KRFPC': 'xkc',
           'CDKExt': 'xce', 'SubFPC': 'xsc', 'AP2DC': 'xac', 'MACCS': 'xma'}


def make_dirs(name):
    for sub in ("", "train", "test"):
        os.makedirs(os.path.join(name, sub), exist_ok=True)


def create_train_test_scaffold(df, name, test_size=TEST_SIZE):
    """Scaffold split; saves x/y train/test CSVs into <name>/train and <name>/test.
    The index (ChEMBL ID) is kept in every file."""
    result = train_test_split_molecules(
        molecules=df[SMILES_COL].to_numpy(), y=df[LABEL_COL].to_numpy(),
        test_size=float(test_size), train_size=float(1.0 - test_size),
        sampler="scaffold", random_state=0,
    )
    # As in your original code, the first two outputs are the train/test SMILES.
    # Map them back to rows of df by SMILES (unique after de-duplication), so we
    # don't depend on how astartes returns y or the index.
    smi_train = np.asarray(result[0]).ravel().astype(str)
    smi_test = np.asarray(result[1]).ravel().astype(str)

    pos = pd.Series(np.arange(len(df)), index=df[SMILES_COL].to_numpy())
    if not pos.index.is_unique:
        raise ValueError("canonical_smiles must be unique before splitting")
    train = df.iloc[pos.loc[smi_train].to_numpy()]   # KeyError if a SMILES can't be matched
    test = df.iloc[pos.loc[smi_test].to_numpy()]
    assert not (set(train.index) & set(test.index)), "train/test overlap"
    if len(train) + len(test) != len(df):
        print(f"WARNING: {len(df) - len(train) - len(test)} molecules were not assigned "
              f"to train or test (see astartes NoMatchingScaffold warning)")

    x_train, y_train = train[[SMILES_COL]], train[[LABEL_COL]]
    x_test,  y_test  = test[[SMILES_COL]],  test[[LABEL_COL]]

    x_train.to_csv(os.path.join(name, "train", "x_train.csv"))
    y_train.to_csv(os.path.join(name, "train", "y_train.csv"))
    x_test.to_csv(os.path.join(name, "test", "x_test.csv"))
    y_test.to_csv(os.path.join(name, "test", "y_test.csv"))
    return x_train, x_test, y_train, y_test


def compute_fps(df, name, split):
    """Compute the 12 PaDEL fingerprints for df (a train or test x-frame).
    Saves <name>/<split>/x??_<split>.csv and returns {fp_name: DataFrame}."""
    xml_files = sorted(glob(os.path.join(name, "*.xml")))
    if len(xml_files) != len(FP_LIST):
        raise FileNotFoundError(
            f"Expected {len(FP_LIST)} PaDEL .xml files in '{name}/', found {len(xml_files)}")
    xml = dict(zip(FP_LIST, xml_files))
    print(xml)

    split_dir = os.path.join(name, split)
    smi_path = os.path.join(name, split + ".smi")
    df[SMILES_COL].to_csv(smi_path, sep="\t", index=False, header=False)

    results = {}
    for fp_name in FP_LIST:
        raw = os.path.join(split_dir, fp_name + ".csv")
        padeldescriptor(mol_dir=smi_path, d_file=raw,
                        descriptortypes=xml[fp_name],
                        retainorder=True, removesalt=True, threads=2,
                        detectaromaticity=True, standardizetautomers=True,
                        standardizenitro=True, fingerprints=True)
        fp = pd.read_csv(raw)
        if len(fp) != len(df):
            raise ValueError(f"{fp_name}: PaDEL returned {len(fp)} rows for {len(df)} molecules")
        fp = fp.drop(columns="Name")
        fp.index = df.index                       # restore Molecule ChEMBL ID
        fp.to_csv(os.path.join(split_dir, f"{FP_CODE[fp_name]}_{split}.csv"))
        if not KEEP_RAW_PADEL_CSV:
            os.remove(raw)
        results[fp_name] = fp
        print(fp_name, split, "done")
    return results


def run(name, csv_path):
    df = pd.read_csv(csv_path, index_col=0, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    missing = [c for c in (SMILES_COL, LABEL_COL) if c not in df.columns]
    if missing:
        raise KeyError(f"{csv_path} is missing {missing}; columns found: {df.columns.tolist()}")

    n0 = len(df)
    df = df[~df[SMILES_COL].duplicated(keep="first")]   # safety net; preprocessing already dedups
    if len(df) != n0:
        print(f"Removed {n0 - len(df)} leftover duplicate SMILES")
    print(name, ":", len(df))

    make_dirs(name)
    df.to_csv(os.path.join(name, name + ".csv"))

    x_train, x_test, y_train, y_test = create_train_test_scaffold(df, name)
    print(name, " train:", len(x_train), " test:", len(x_test))

    fps_train = compute_fps(x_train, name, "train")
    fps_test = compute_fps(x_test, name, "test")
    return fps_train, fps_test


def main():
    datasets = {"temu": "temu_processed.csv"}   # folder name -> cleaned CSV from preprocess_pipeline.py
    for name, csv_path in datasets.items():
        run(name, csv_path)


if __name__ == "__main__":
    main()