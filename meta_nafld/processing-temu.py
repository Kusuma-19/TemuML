"""
Raw data -> cleaned CSV with 'canonical_smiles', keeping the Molecule ChEMBL ID.

The ChEMBL ID becomes the index (first CSV column), so it is picked up by your
processing script via read_csv(index_col=0) and carried into train/test splits
and every fingerprint file (they all do set_index(df.index)).
"""
import os
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
import custom_preprocessing as cp

RDLogger.DisableLog("rdApp.*")

_lfc = rdMolStandardize.LargestFragmentChooser()
_unch = rdMolStandardize.Uncharger()

ID_NAME = "Molecule ChEMBL ID"


def to_canonical(smi, standardize=True, isomeric=True):
    """Canonical SMILES or None (never raises)."""
    if not isinstance(smi, str) or not smi.strip():
        return None
    mol = Chem.MolFromSmiles(smi.strip())
    if mol is None:
        return None
    try:
        if standardize:
            mol = rdMolStandardize.Cleanup(mol)
            mol = _lfc.choose(mol)      # keep largest fragment (removes salts)
            mol = _unch.uncharge(mol)   # neutralize
        return Chem.MolToSmiles(mol, isomericSmiles=isomeric)
    except Exception:
        return None


def add_canonical_smiles(df, smiles_col, standardize=True, isomeric=True):
    """Adds 'canonical_smiles'; DROPS and reports rows that fail."""
    df = df.copy()
    df["canonical_smiles"] = df[smiles_col].apply(
        lambda s: to_canonical(s, standardize, isomeric)
    )
    bad = df[df["canonical_smiles"].isna()]
    if len(bad):
        print(f"Dropping {len(bad)} invalid SMILES (first IDs: {list(bad.index[:10])})")
    return df.dropna(subset=["canonical_smiles"])


def _ids_per_molecule(df, id_col):
    """canonical_smiles -> 'CHEMBL1;CHEMBL2' (all IDs that collapse to one molecule)."""
    return (df.reset_index().groupby("canonical_smiles")[id_col]
              .agg(lambda s: ";".join(sorted(set(map(str, s))))))


def run_regression(raw_path, out_path, smiles_col, ic50_col, units_col,
                   id_col="Molecule ChEMBL ID", units="nM", threshold=0.2):
    """IC50 -> pIC50 workflow (your cp functions), ChEMBL ID kept as index."""
    os.makedirs(os.path.join("datasets", "processed"), exist_ok=True)  # needed by cp.process_duplicates

    df = pd.read_csv(raw_path)
    df = df.set_index(id_col)   # ID rides along as the index through the cp functions
    df = cp.process_df(df, smiles_col, ic50_col, units_col, units)   # (renames index to 'LigandID')
    df.index.name = id_col
    df = df[pd.to_numeric(df[ic50_col], errors="coerce") > 0]        # log10(<=0) is -inf/NaN
    df = add_canonical_smiles(df, smiles_col)
    df = cp.remove_inorganic(df, "canonical_smiles")
    df = cp.nanomolarconversion(df, ic50_col)
    df = cp.calculate_pic50(df, ic50_col)

    all_ids = _ids_per_molecule(df, id_col)                          # before duplicates are merged
    df.index.name = "LigandID"                                       # what cp.process_duplicates expects
    df = cp.process_duplicates(df, "canonical_smiles", "pIC50", threshold=threshold)
    df = cp.remove_missingdata(df)

    df.index.name = id_col
    df["all_chembl_ids"] = df["canonical_smiles"].map(all_ids)       # merged IDs for averaged duplicates
    df.to_csv(out_path)
    print(f"Saved {len(df)} molecules -> {out_path}")
    return df


def run_classification(raw_path, out_path, smiles_col, label_col, id_col="Molecule ChEMBL ID"):
    """Class-label workflow: keep one copy if labels agree, drop the molecule
    entirely if labels conflict. ChEMBL ID kept as index."""
    df = pd.read_csv(raw_path)
    df = df[[id_col, smiles_col, label_col]].dropna()
    df = add_canonical_smiles(df.set_index(id_col), smiles_col)
    df = cp.remove_inorganic(df, "canonical_smiles")

    n_lab = df.groupby("canonical_smiles")[label_col].nunique()
    conflict = n_lab[n_lab > 1].index
    if len(conflict):
        print(f"Removing {len(conflict)} molecules with conflicting labels")
    df = df[~df["canonical_smiles"].isin(conflict)]

    all_ids = _ids_per_molecule(df, id_col)
    df = df[~df["canonical_smiles"].duplicated(keep="first")]
    df["all_chembl_ids"] = df["canonical_smiles"].map(all_ids)

    df = df[["canonical_smiles", label_col, "all_chembl_ids"]]
    df.index.name = id_col
    df.to_csv(out_path)
    print(f"Saved {len(df)} molecules -> {out_path}")
    return df


if __name__ == "__main__":
    # Classification (feeds your main(): temu.csv with 'canonical_smiles' + 'class')
    run_classification("temu.csv", "temu_processed.csv", smiles_col="smiles",
                       label_col="class", id_col="Molecule ChEMBL ID")

    # Regression example (uncomment and adjust column names)
    # run_regression("chembl_raw.csv", "target.csv",
    #                smiles_col="Smiles", ic50_col="Standard Value",
    #                units_col="Standard Units", id_col="Molecule ChEMBL ID", units="nM")