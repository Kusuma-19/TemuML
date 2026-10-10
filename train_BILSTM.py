"""
MLP stacking pipeline for AChE inhibitors.

Reads the files produced by your fingerprint script:
    <NAME>/train/{xat,xes,...}_train.csv, y_train.csv, x_train.csv
    <NAME>/test/ {xat,xes,...}_test.csv,  y_test.csv

Steps
  1. Load the 12 fingerprints for train and test (index = ChEMBL ID, aligned with y).
  2. Preprocess per fingerprint (drop constant columns; log1p + standardise count fingerprints).
  3. Out-of-fold (OOF) MLP predictions on train with scaffold-grouped, stratified CV.
     Test predictions = average of the fold models.
  4. Meta-model (logistic regression) on the 12 OOF probabilities, plus a simple-mean baseline.
  5. Metrics, saved models/predictions, and a kNN applicability domain computed on train fingerprints.

Requires: tensorflow, scikit-learn, rdkit, pandas, numpy, joblib
"""
import os
import random

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, models, optimizers

# ----------------------------------------------------------------------------- config
NAME = "temu"                      # folder that contains train/ and test/
SMILES_COL = "canonical_smiles"
FP_TYPES = ["xat", "xes", "xke", "xpc", "xss", "xcd",
            "xcn", "xkc", "xce", "xsc", "xac", "xma"]
SEED = 42
N_FOLDS = 5
EPOCHS = 200                       # upper limit, early stopping decides
BATCH_SIZE = 64
PATIENCE = 15
AD_FP = "xpc"                      # binary fingerprint used for the applicability domain
AD_Z = 1.0                         # threshold = mean + Z * std of train kNN distances
MODEL_TYPE = "bilstm"              # "mlp" or "bilstm" (results go to <NAME>/results_<MODEL_TYPE>)
RESULTS_DIR = os.path.join(NAME, f"results_{MODEL_TYPE}")
MODEL_DIR = os.path.join(RESULTS_DIR, "models")

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs(MODEL_DIR, exist_ok=True)


# ----------------------------------------------------------------------------- loading
def load_labels(split):
    y = pd.read_csv(os.path.join(NAME, split, f"y_{split}.csv"), index_col=0)
    return y.iloc[:, 0].astype(int)


def load_fp(split, ftype, index):
    """Load <NAME>/<split>/<ftype>_<split>.csv and align rows to `index` (the y index)."""
    path = os.path.join(NAME, split, f"{ftype}_{split}.csv")
    fp = pd.read_csv(path, index_col=0)
    if not fp.index.equals(index):
        fp = fp.loc[index]          # KeyError if IDs do not match
    return fp


def load_all():
    y_train, y_test = load_labels("train"), load_labels("test")
    fps_train = {f: load_fp("train", f, y_train.index) for f in FP_TYPES}
    fps_test = {f: load_fp("test", f, y_test.index) for f in FP_TYPES}
    smiles_train = pd.read_csv(os.path.join(NAME, "train", "x_train.csv"),
                               index_col=0).loc[y_train.index, SMILES_COL]
    return fps_train, fps_test, y_train, y_test, smiles_train


def scaffold_groups(smiles):
    """Murcko scaffold id per molecule; acyclic molecules each get their own group."""
    keys = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol is not None else ""
        keys.append(scaf if scaf else f"noscaf_{i}")
    return pd.factorize(pd.Series(keys))[0]


# ----------------------------------------------------------------------------- preprocessing
class FPPreprocessor:
    """Drop constant columns; for count fingerprints apply log1p + standardisation."""

    def fit(self, X):
        self.keep = X.columns[X.nunique() > 1]
        X = X[self.keep].fillna(0)
        self.is_count = bool(X.to_numpy().max() > 1)
        self.scaler = None
        if self.is_count:
            self.scaler = StandardScaler().fit(np.log1p(X.clip(lower=0)))
        return self

    def transform(self, X):
        X = X[self.keep].fillna(0)
        if self.is_count:
            return self.scaler.transform(np.log1p(X.clip(lower=0))).astype("float32")
        return X.to_numpy(dtype="float32")


# ----------------------------------------------------------------------------- model
def build_mlp(n_features):
    model = models.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=optimizers.Adam(1e-3), loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def build_bilstm(n_features):
    """Bits are fed as a sequence of length n_features with 1 channel.
    Kept light (one BiLSTM layer) because sequence length can reach several thousand steps.
    For the original two-layer version use: Bidirectional(LSTM(64, return_sequences=True))
    followed by Bidirectional(LSTM(32))."""
    model = models.Sequential([
        layers.Input(shape=(n_features, 1)),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=optimizers.Adam(1e-3), loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def build_model(n_features):
    return build_bilstm(n_features) if MODEL_TYPE == "bilstm" else build_mlp(n_features)


def _shape(X):
    """Add the channel axis for the BiLSTM; no-op for the MLP or if already 3D."""
    return X[..., None] if (MODEL_TYPE == "bilstm" and X.ndim == 2) else X


def fit_mlp(X, y, groups):
    """Fit one model (MLP or BiLSTM); early stopping uses a scaffold-grouped inner validation split."""
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr, va = next(inner.split(X, y, groups))
    Xs = _shape(X)
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y[tr])
    model = build_model(X.shape[1])
    model.fit(Xs[tr], y[tr], validation_data=(Xs[va], y[va]),
              epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
              class_weight={0: cw[0], 1: cw[1]},
              callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                                 restore_best_weights=True)])
    return model


def oof_and_test(ftype, X_tr, y_tr, groups, X_te):
    """OOF predictions on train (scaffold-grouped CV) and fold-averaged predictions on test."""
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y_tr))
    test_pred = np.zeros(len(X_te))
    os.makedirs(os.path.join(MODEL_DIR, ftype), exist_ok=True)
    for k, (tr, va) in enumerate(cv.split(X_tr, y_tr, groups)):
        model = fit_mlp(X_tr[tr], y_tr[tr], groups[tr])
        oof[va] = model.predict(_shape(X_tr[va]), verbose=0).ravel()
        test_pred += model.predict(_shape(X_te), verbose=0).ravel() / N_FOLDS
        model.save(os.path.join(MODEL_DIR, ftype, f"{MODEL_TYPE}_{ftype}_fold{k}.keras"))
        tf.keras.backend.clear_session()
    return oof, test_pred


# ----------------------------------------------------------------------------- metrics
def evaluate(y_true, y_prob, label, threshold=0.5):
    y_pred = (np.asarray(y_prob) > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return pd.DataFrame({
        "Accuracy": accuracy_score(y_true, y_pred),
        "Sensitivity": recall_score(y_true, y_pred),
        "Specificity": tn / (tn + fp),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "BACC": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
    }, index=[label]).round(4)


# ----------------------------------------------------------------------------- applicability domain
def knn_ad(X_train_bin, X_query_bin, k, z=AD_Z):
    """kNN applicability domain fitted on TRAIN only (Jaccard distance on binary bits).
    Train distances exclude each molecule's own match."""
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric="jaccard").fit(X_train_bin)
    d_train = nn.kneighbors(X_train_bin)[0][:, 1:].mean(axis=1)
    dk, sk = d_train.mean(), d_train.std()
    d_query = nn.kneighbors(X_query_bin, n_neighbors=k)[0].mean(axis=1)
    return d_query <= dk + z * sk, dk, sk


def run_ad(fps_train, fps_test, y_test, meta_prob_test, ks=range(3, 11)):
    Xtr = (fps_train[AD_FP].to_numpy() > 0)
    Xte = (fps_test[AD_FP].to_numpy() > 0)
    rows = []
    for k in ks:
        inside, dk, sk = knn_ad(Xtr, Xte, k)
        if inside.sum() == 0 or len(np.unique(y_test[inside])) < 2:
            continue
        m = evaluate(y_test[inside], meta_prob_test[inside], f"k={k}")
        m["Removed"] = int((~inside).sum())
        m["dk"], m["sk"] = round(dk, 4), round(sk, 4)
        rows.append(m)
    ad = pd.concat(rows)
    ad.to_csv(os.path.join(RESULTS_DIR, f"AD_metrics_{AD_FP}_z{AD_Z}.csv"))
    return ad


# ----------------------------------------------------------------------------- main
def main():
    fps_train, fps_test, y_train, y_test, smiles_train = load_all()
    print("train:", len(y_train), " test:", len(y_test),
          " | train positives: %.1f%%" % (100 * y_train.mean()))
    y_tr, y_te = y_train.to_numpy(), y_test.to_numpy()
    groups = scaffold_groups(smiles_train)
    print("scaffold groups in train:", len(np.unique(groups)))

    oof_cols, test_cols = {}, {}
    metrics_oof, metrics_test = [], []
    for ftype in FP_TYPES:
        prep = FPPreprocessor().fit(fps_train[ftype])
        joblib.dump(prep, os.path.join(MODEL_DIR, f"prep_{ftype}.joblib"))
        X_tr, X_te = prep.transform(fps_train[ftype]), prep.transform(fps_test[ftype])
        print(f"[{ftype}] features kept: {X_tr.shape[1]} | count-type: {prep.is_count}")

        oof, test_pred = oof_and_test(ftype, X_tr, y_tr, groups, X_te)
        oof_cols[ftype], test_cols[ftype] = oof, test_pred
        metrics_oof.append(evaluate(y_tr, oof, f"{MODEL_TYPE.upper()}_{ftype}_trainOOF"))
        metrics_test.append(evaluate(y_te, test_pred, f"{MODEL_TYPE.upper()}_{ftype}_test"))
        print(metrics_test[-1][["MCC", "AUC"]].to_string(header=False))

    stacked_train = pd.DataFrame(oof_cols, index=y_train.index)
    stacked_test = pd.DataFrame(test_cols, index=y_test.index)
    stacked_train.to_csv(os.path.join(RESULTS_DIR, "stacked_train_oof.csv"))
    stacked_test.to_csv(os.path.join(RESULTS_DIR, "stacked_test.csv"))

    # ---- meta-model: logistic regression on the 12 OOF probabilities
    meta = LogisticRegression(max_iter=1000)
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + 1)
    meta_oof = cross_val_predict(meta, stacked_train.values, y_tr, cv=cv,
                                 groups=groups, method="predict_proba")[:, 1]
    meta.fit(stacked_train.values, y_tr)
    meta_test = meta.predict_proba(stacked_test.values)[:, 1]
    joblib.dump(meta, os.path.join(MODEL_DIR, "meta_logreg.joblib"))

    metrics_oof.append(evaluate(y_tr, meta_oof, "Meta_LogReg_trainCV"))
    metrics_test.append(evaluate(y_te, meta_test, "Meta_LogReg_test"))
    metrics_oof.append(evaluate(y_tr, stacked_train.mean(axis=1), "Mean_ensemble_trainOOF"))
    metrics_test.append(evaluate(y_te, stacked_test.mean(axis=1), "Mean_ensemble_test"))

    pd.concat(metrics_oof).to_csv(os.path.join(RESULTS_DIR, "metrics_train_oof.csv"))
    pd.concat(metrics_test).to_csv(os.path.join(RESULTS_DIR, "metrics_test.csv"))
    pd.DataFrame({"y_true": y_te, "y_prob": meta_test, "y_pred": (meta_test > 0.5).astype(int)},
                 index=y_test.index).to_csv(os.path.join(RESULTS_DIR, "meta_test_predictions.csv"))
    print("\nTest metrics:\n", pd.concat(metrics_test).to_string())

    # ---- applicability domain (fitted on train fingerprints)
    ad = run_ad(fps_train, fps_test, y_te, meta_test)
    print("\nApplicability domain:\n", ad.to_string())


if __name__ == "__main__":
    main()