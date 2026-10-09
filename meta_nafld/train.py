import pandas as pd
import numpy as np
import os
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score, roc_auc_score, balanced_accuracy_score, roc_curve, matthews_corrcoef, precision_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import xgboost as xgb
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import GridSearchCV
import seaborn as sns

""" 
1. Load x and y
2. Train stacked model (BL + Stack + 5-CV) (save)
3. Evaluate performance of all train, cv, test (save)
4. SHAP (save)
5. Y-random (save)
6. AD (save)
"""
def y_prediction(model, x_train, y_train, col_name):
    y_pred = pd.DataFrame(model.predict(x_train), columns=[col_name]).set_index(x_train.index)
    acc = accuracy_score(y_train, y_pred)
    sen = recall_score(y_train, y_pred)  # Sensitivity is the same as recall
    mcc = matthews_corrcoef(y_train, y_pred)
    f1  = f1_score(y_train, y_pred)
    auc = roc_auc_score(y_train, y_pred)
    bcc = balanced_accuracy_score(y_train, y_pred)
    pre = precision_score(y_train, y_pred)
    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
    spc = tn / (tn + fp)

    # Create a DataFrame to store the metrics
    metrics = pd.DataFrame({
        'Accuracy': [acc],
        'Sensitivity': [sen],
        'Specificity': [spc],
        'MCC': [mcc],
        'F1 Score': [f1],
        'AUC': [auc],
        'BACC': [bcc],
        'Precision': [pre]
    }, index=[col_name])
    return y_pred, metrics
    
def y_prediction_cv(model, x_train, y_train, col_name):
    y_pred = pd.DataFrame(cross_val_predict(model,x_train,y_train, cv=5), columns=[col_name]).set_index(x_train.index)
    acc = accuracy_score(y_train, y_pred)
    sen = recall_score(y_train, y_pred)  # Sensitivity is the same as recall
    mcc = matthews_corrcoef(y_train, y_pred)
    f1  = f1_score(y_train, y_pred)
    auc = roc_auc_score(y_train, y_pred)
    bcc = balanced_accuracy_score(y_train, y_pred)
    pre = precision_score(y_train, y_pred)
    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
    spc = tn / (tn + fp)

    # Create a DataFrame to store the metrics
    metrics = pd.DataFrame({
        'Accuracy': [acc],
        'Sensitivity': [sen],
        'Specificity': [spc],
        'MCC': [mcc],
        'F1 Score': [f1],
        'AUC': [auc],
        'BACC': [bcc],
        'Precision': [pre]
    }, index=[col_name])
    return y_pred, metrics

def stacked_class(name):
    xat_train = pd.read_csv(os.path.join(name, "train", 'xat_train.csv'), index_col=0)
    xes_train = pd.read_csv(os.path.join(name, "train", 'xes_train.csv'), index_col=0)
    xke_train = pd.read_csv(os.path.join(name, "train", 'xke_train.csv'), index_col=0)
    xpc_train = pd.read_csv(os.path.join(name, "train", 'xpc_train.csv'), index_col=0)
    xss_train = pd.read_csv(os.path.join(name, "train", 'xss_train.csv'), index_col=0)
    xcd_train = pd.read_csv(os.path.join(name, "train", 'xcd_train.csv'), index_col=0)
    xcn_train = pd.read_csv(os.path.join(name, "train", 'xcn_train.csv'), index_col=0)
    xkc_train = pd.read_csv(os.path.join(name, "train", 'xkc_train.csv'), index_col=0)
    xce_train = pd.read_csv(os.path.join(name, "train", 'xce_train.csv'), index_col=0)
    xsc_train = pd.read_csv(os.path.join(name, "train", 'xsc_train.csv'), index_col=0)
    xac_train = pd.read_csv(os.path.join(name, "train", 'xac_train.csv'), index_col=0)
    xma_train = pd.read_csv(os.path.join(name, "train", 'xma_train.csv'), index_col=0)
    y_train   = pd.read_csv(os.path.join(name, "train", "y_train.csv" ), index_col=0)
    xat_test = pd.read_csv(os.path.join( name, "test",  "xat_test.csv" ), index_col=0)
    xes_test = pd.read_csv(os.path.join( name, "test",  "xes_test.csv" ), index_col=0)
    xke_test = pd.read_csv(os.path.join( name, "test",  "xke_test.csv" ), index_col=0)
    xpc_test = pd.read_csv(os.path.join( name, "test",  "xpc_test.csv" ), index_col=0)
    xss_test = pd.read_csv(os.path.join( name, "test",  "xss_test.csv" ), index_col=0)
    xcd_test = pd.read_csv(os.path.join( name, "test",  "xcd_test.csv" ), index_col=0)
    xcn_test = pd.read_csv(os.path.join( name, "test",  "xcn_test.csv" ), index_col=0)
    xkc_test = pd.read_csv(os.path.join( name, "test",  "xkc_test.csv" ), index_col=0)
    xce_test = pd.read_csv(os.path.join( name, "test",  "xce_test.csv" ), index_col=0)
    xsc_test = pd.read_csv(os.path.join( name, "test",  "xsc_test.csv" ), index_col=0)
    xac_test = pd.read_csv(os.path.join( name, "test",  "xac_test.csv" ), index_col=0)
    xma_test = pd.read_csv(os.path.join( name, "test",  "xma_test.csv" ), index_col=0)
    y_test   = pd.read_csv(os.path.join( name, "test",  "y_test.csv"), index_col=0)
    baseline_model_rf_at = RandomForestClassifier(max_depth=5, random_state=1).fit(xat_train, y_train)
    baseline_model_rf_ke = RandomForestClassifier(max_depth=5, random_state=1).fit(xke_train, y_train)
    baseline_model_rf_es = RandomForestClassifier(max_depth=5, random_state=1).fit(xes_train, y_train)
    baseline_model_rf_pc = RandomForestClassifier(max_depth=5, random_state=1).fit(xpc_train, y_train)
    baseline_model_rf_ss = RandomForestClassifier(max_depth=5, random_state=1).fit(xss_train, y_train)
    baseline_model_rf_cd = RandomForestClassifier(max_depth=5, random_state=1).fit(xcd_train, y_train)
    baseline_model_rf_cn = RandomForestClassifier(max_depth=5, random_state=1).fit(xcn_train, y_train)
    baseline_model_rf_kc = RandomForestClassifier(max_depth=5, random_state=1).fit(xkc_train, y_train)
    baseline_model_rf_ce = RandomForestClassifier(max_depth=5, random_state=1).fit(xce_train, y_train)
    baseline_model_rf_sc = RandomForestClassifier(max_depth=5, random_state=1).fit(xsc_train, y_train)
    baseline_model_rf_ac = RandomForestClassifier(max_depth=5, random_state=1).fit(xac_train, y_train)
    baseline_model_rf_ma = RandomForestClassifier(max_depth=5, random_state=1).fit(xma_train, y_train)
    dump(baseline_model_rf_at, os.path.join(name, "train", "baseline_model_rf_at.joblib"))
    dump(baseline_model_rf_ke, os.path.join(name, "train", "baseline_model_rf_ke.joblib"))
    dump(baseline_model_rf_es, os.path.join(name, "train", "baseline_model_rf_es.joblib"))
    dump(baseline_model_rf_pc, os.path.join(name, "train", "baseline_model_rf_pc.joblib"))
    dump(baseline_model_rf_ss, os.path.join(name, "train", "baseline_model_rf_ss.joblib"))
    dump(baseline_model_rf_cd, os.path.join(name, "train", "baseline_model_rf_cd.joblib"))
    dump(baseline_model_rf_cn, os.path.join(name, "train", "baseline_model_rf_cn.joblib"))
    dump(baseline_model_rf_kc, os.path.join(name, "train", "baseline_model_rf_kc.joblib"))
    dump(baseline_model_rf_ce, os.path.join(name, "train", "baseline_model_rf_ce.joblib"))
    dump(baseline_model_rf_sc, os.path.join(name, "train", "baseline_model_rf_sc.joblib"))
    dump(baseline_model_rf_ac, os.path.join(name, "train", "baseline_model_rf_ac.joblib"))
    dump(baseline_model_rf_ma, os.path.join(name, "train", "baseline_model_rf_ma.joblib"))
    yat_pred_rf_train, yat_metric_rf_train = y_prediction(baseline_model_rf_at, xat_train, y_train, "yat_pred_rf")
    yes_pred_rf_train, yes_metric_rf_train = y_prediction(baseline_model_rf_es, xes_train, y_train, "yes_pred_rf")
    yke_pred_rf_train, yke_metric_rf_train = y_prediction(baseline_model_rf_ke, xke_train, y_train, "yke_pred_rf")
    ypc_pred_rf_train, ypc_metric_rf_train = y_prediction(baseline_model_rf_pc, xpc_train, y_train, "ypc_pred_rf")
    yss_pred_rf_train, yss_metric_rf_train = y_prediction(baseline_model_rf_ss, xss_train, y_train, "yss_pred_rf")
    ycd_pred_rf_train, ycd_metric_rf_train = y_prediction(baseline_model_rf_cd, xcd_train, y_train, "ycd_pred_rf")
    ycn_pred_rf_train, ycn_metric_rf_train = y_prediction(baseline_model_rf_cn, xcn_train, y_train, "ycn_pred_rf")
    ykc_pred_rf_train, ykc_metric_rf_train = y_prediction(baseline_model_rf_kc, xkc_train, y_train, "ykc_pred_rf")
    yce_pred_rf_train, yce_metric_rf_train = y_prediction(baseline_model_rf_ce, xce_train, y_train, "yce_pred_rf")
    ysc_pred_rf_train, ysc_metric_rf_train = y_prediction(baseline_model_rf_sc, xsc_train, y_train, "ysc_pred_rf")
    yac_pred_rf_train, yac_metric_rf_train = y_prediction(baseline_model_rf_ac, xac_train, y_train, "yac_pred_rf")
    yma_pred_rf_train, yma_metric_rf_train = y_prediction(baseline_model_rf_ma, xma_train, y_train, "yma_pred_rf")
    yat_pred_rf_test,  yat_metric_rf_test   =  y_prediction(baseline_model_rf_at, xat_test, y_test, "yat_pred_rf")
    yes_pred_rf_test,  yes_metric_rf_test   =  y_prediction(baseline_model_rf_es, xes_test, y_test, "yes_pred_rf")
    yke_pred_rf_test,  yke_metric_rf_test   =  y_prediction(baseline_model_rf_ke, xke_test, y_test, "yke_pred_rf")
    ypc_pred_rf_test,  ypc_metric_rf_test   =  y_prediction(baseline_model_rf_pc, xpc_test, y_test, "ypc_pred_rf")
    yss_pred_rf_test,  yss_metric_rf_test   =  y_prediction(baseline_model_rf_ss, xss_test, y_test, "yss_pred_rf")
    ycd_pred_rf_test,  ycd_metric_rf_test   =  y_prediction(baseline_model_rf_cd, xcd_test, y_test, "ycd_pred_rf")
    ycn_pred_rf_test,  ycn_metric_rf_test   =  y_prediction(baseline_model_rf_cn, xcn_test, y_test, "ycn_pred_rf")
    ykc_pred_rf_test,  ykc_metric_rf_test   =  y_prediction(baseline_model_rf_kc, xkc_test, y_test, "ykc_pred_rf")
    yce_pred_rf_test,  yce_metric_rf_test   =  y_prediction(baseline_model_rf_ce, xce_test, y_test, "yce_pred_rf")
    ysc_pred_rf_test,  ysc_metric_rf_test   =  y_prediction(baseline_model_rf_sc, xsc_test, y_test, "ysc_pred_rf")
    yac_pred_rf_test,  yac_metric_rf_test   =  y_prediction(baseline_model_rf_ac, xac_test, y_test, "yac_pred_rf")
    yma_pred_rf_test,  yma_metric_rf_test   =  y_prediction(baseline_model_rf_ma, xma_test, y_test, "yma_pred_rf")
    yat_pred_rf_cv,    yat_metric_rf_cv  =  y_prediction_cv(baseline_model_rf_at, xat_train, y_train, "yat_pred_rf")
    yes_pred_rf_cv,    yes_metric_rf_cv  =  y_prediction_cv(baseline_model_rf_es, xes_train, y_train, "yes_pred_rf")
    yke_pred_rf_cv,    yke_metric_rf_cv  =  y_prediction_cv(baseline_model_rf_ke, xke_train, y_train, "yke_pred_rf")
    ypc_pred_rf_cv,    ypc_metric_rf_cv  =  y_prediction_cv(baseline_model_rf_pc, xpc_train, y_train, "ypc_pred_rf")
    yss_pred_rf_cv,    yss_metric_rf_cv  =  y_prediction_cv(baseline_model_rf_ss, xss_train, y_train, "yss_pred_rf")
    ycd_pred_rf_cv,    ycd_metric_rf_cv  =  y_prediction_cv(baseline_model_rf_cd, xcd_train, y_train, "ycd_pred_rf")
    ycn_pred_rf_cv,    ycn_metric_rf_cv  =  y_prediction_cv(baseline_model_rf_cn, xcn_train, y_train, "ycn_pred_rf")
    ykc_pred_rf_cv,    ykc_metric_rf_cv  =  y_prediction_cv(baseline_model_rf_kc, xkc_train, y_train, "ykc_pred_rf")
    yce_pred_rf_cv,    yce_metric_rf_cv  =  y_prediction_cv(baseline_model_rf_ce, xce_train, y_train, "yce_pred_rf")
    ysc_pred_rf_cv,    ysc_metric_rf_cv  =  y_prediction_cv(baseline_model_rf_sc, xsc_train, y_train, "ysc_pred_rf")
    yac_pred_rf_cv,    yac_metric_rf_cv  =  y_prediction_cv(baseline_model_rf_ac, xac_train, y_train, "yac_pred_rf")
    yma_pred_rf_cv,    yma_metric_rf_cv  =  y_prediction_cv(baseline_model_rf_ma, xma_train, y_train, "yma_pred_rf")
    #XGB
    baseline_model_xgb_at = xgb.XGBClassifier(objective="binary:logistic",eval_metric='auc', max_depth=5, random_state=1).fit(xat_train, y_train)
    baseline_model_xgb_ke = xgb.XGBClassifier(objective="binary:logistic",eval_metric='auc', max_depth=5, random_state=1).fit(xke_train, y_train)
    baseline_model_xgb_es = xgb.XGBClassifier(objective="binary:logistic",eval_metric='auc', max_depth=5, random_state=1).fit(xes_train, y_train)
    baseline_model_xgb_pc = xgb.XGBClassifier(objective="binary:logistic",eval_metric='auc', max_depth=5, random_state=1).fit(xpc_train, y_train)
    baseline_model_xgb_ss = xgb.XGBClassifier(objective="binary:logistic",eval_metric='auc', max_depth=5, random_state=1).fit(xss_train, y_train)
    baseline_model_xgb_cd = xgb.XGBClassifier(objective="binary:logistic",eval_metric='auc', max_depth=5, random_state=1).fit(xcd_train, y_train)
    baseline_model_xgb_cn = xgb.XGBClassifier(objective="binary:logistic",eval_metric='auc', max_depth=5, random_state=1).fit(xcn_train, y_train)
    baseline_model_xgb_kc = xgb.XGBClassifier(objective="binary:logistic",eval_metric='auc', max_depth=5, random_state=1).fit(xkc_train, y_train)
    baseline_model_xgb_ce = xgb.XGBClassifier(objective="binary:logistic",eval_metric='auc', max_depth=5, random_state=1).fit(xce_train, y_train)
    baseline_model_xgb_sc = xgb.XGBClassifier(objective="binary:logistic",eval_metric='auc', max_depth=5, random_state=1).fit(xsc_train, y_train)
    baseline_model_xgb_ac = xgb.XGBClassifier(objective="binary:logistic",eval_metric='auc', max_depth=5, random_state=1).fit(xac_train, y_train)
    baseline_model_xgb_ma = xgb.XGBClassifier(objective="binary:logistic",eval_metric='auc', max_depth=5, random_state=1).fit(xma_train, y_train)
    dump(baseline_model_xgb_at, os.path.join(name, "train", "baseline_model_xgb_at.joblib"))
    dump(baseline_model_xgb_ke, os.path.join(name, "train", "baseline_model_xgb_ke.joblib"))
    dump(baseline_model_xgb_es, os.path.join(name, "train", "baseline_model_xgb_es.joblib"))
    dump(baseline_model_xgb_pc, os.path.join(name, "train", "baseline_model_xgb_pc.joblib"))
    dump(baseline_model_xgb_ss, os.path.join(name, "train", "baseline_model_xgb_ss.joblib"))
    dump(baseline_model_xgb_cd, os.path.join(name, "train", "baseline_model_xgb_cd.joblib"))
    dump(baseline_model_xgb_cn, os.path.join(name, "train", "baseline_model_xgb_cn.joblib"))
    dump(baseline_model_xgb_kc, os.path.join(name, "train", "baseline_model_xgb_kc.joblib"))
    dump(baseline_model_xgb_ce, os.path.join(name, "train", "baseline_model_xgb_ce.joblib"))
    dump(baseline_model_xgb_sc, os.path.join(name, "train", "baseline_model_xgb_sc.joblib"))
    dump(baseline_model_xgb_ac, os.path.join(name, "train", "baseline_model_xgb_ac.joblib"))
    dump(baseline_model_xgb_ma, os.path.join(name, "train", "baseline_model_xgb_ma.joblib"))
    yat_pred_xgb_train, yat_metric_xgb_train = y_prediction(   baseline_model_xgb_at, xat_train, y_train, "yat_pred_xgb")
    yes_pred_xgb_train, yes_metric_xgb_train = y_prediction(   baseline_model_xgb_es, xes_train, y_train, "yes_pred_xgb")
    yke_pred_xgb_train, yke_metric_xgb_train = y_prediction(   baseline_model_xgb_ke, xke_train, y_train, "yke_pred_xgb")
    ypc_pred_xgb_train, ypc_metric_xgb_train = y_prediction(   baseline_model_xgb_pc, xpc_train, y_train, "ypc_pred_xgb")
    yss_pred_xgb_train, yss_metric_xgb_train = y_prediction(   baseline_model_xgb_ss, xss_train, y_train, "yss_pred_xgb")
    ycd_pred_xgb_train, ycd_metric_xgb_train = y_prediction(   baseline_model_xgb_cd, xcd_train, y_train, "ycd_pred_xgb")
    ycn_pred_xgb_train, ycn_metric_xgb_train = y_prediction(   baseline_model_xgb_cn, xcn_train, y_train, "ycn_pred_xgb")
    ykc_pred_xgb_train, ykc_metric_xgb_train = y_prediction(   baseline_model_xgb_kc, xkc_train, y_train, "ykc_pred_xgb")
    yce_pred_xgb_train, yce_metric_xgb_train = y_prediction(   baseline_model_xgb_ce, xce_train, y_train, "yce_pred_xgb")
    ysc_pred_xgb_train, ysc_metric_xgb_train = y_prediction(   baseline_model_xgb_sc, xsc_train, y_train, "ysc_pred_xgb")
    yac_pred_xgb_train, yac_metric_xgb_train = y_prediction(   baseline_model_xgb_ac, xac_train, y_train, "yac_pred_xgb")
    yma_pred_xgb_train, yma_metric_xgb_train = y_prediction(   baseline_model_xgb_ma, xma_train, y_train, "yma_pred_xgb")
    yat_pred_xgb_test,  yat_metric_xgb_test   = y_prediction(  baseline_model_xgb_at, xat_test, y_test,   "yat_pred_xgb")
    yes_pred_xgb_test,  yes_metric_xgb_test   = y_prediction(  baseline_model_xgb_es, xes_test, y_test,   "yes_pred_xgb")
    yke_pred_xgb_test,  yke_metric_xgb_test   = y_prediction(  baseline_model_xgb_ke, xke_test, y_test,   "yke_pred_xgb")
    ypc_pred_xgb_test,  ypc_metric_xgb_test   = y_prediction(  baseline_model_xgb_pc, xpc_test, y_test,   "ypc_pred_xgb")
    yss_pred_xgb_test,  yss_metric_xgb_test   = y_prediction(  baseline_model_xgb_ss, xss_test, y_test,   "yss_pred_xgb")
    ycd_pred_xgb_test,  ycd_metric_xgb_test   = y_prediction(  baseline_model_xgb_cd, xcd_test, y_test,   "ycd_pred_xgb")
    ycn_pred_xgb_test,  ycn_metric_xgb_test   = y_prediction(  baseline_model_xgb_cn, xcn_test, y_test,   "ycn_pred_xgb")
    ykc_pred_xgb_test,  ykc_metric_xgb_test   = y_prediction(  baseline_model_xgb_kc, xkc_test, y_test,   "ykc_pred_xgb")
    yce_pred_xgb_test,  yce_metric_xgb_test   = y_prediction(  baseline_model_xgb_ce, xce_test, y_test,   "yce_pred_xgb")
    ysc_pred_xgb_test,  ysc_metric_xgb_test   = y_prediction(  baseline_model_xgb_sc, xsc_test, y_test,   "ysc_pred_xgb")
    yac_pred_xgb_test,  yac_metric_xgb_test   = y_prediction(  baseline_model_xgb_ac, xac_test, y_test,   "yac_pred_xgb")
    yma_pred_xgb_test,  yma_metric_xgb_test   = y_prediction(  baseline_model_xgb_ma, xma_test, y_test,   "yma_pred_xgb")
    yat_pred_xgb_cv,    yat_metric_xgb_cv    = y_prediction_cv(baseline_model_xgb_at, xat_train, y_train, "yat_pred_xgb")
    yes_pred_xgb_cv,    yes_metric_xgb_cv    = y_prediction_cv(baseline_model_xgb_es, xes_train, y_train, "yes_pred_xgb")
    yke_pred_xgb_cv,    yke_metric_xgb_cv    = y_prediction_cv(baseline_model_xgb_ke, xke_train, y_train, "yke_pred_xgb")
    ypc_pred_xgb_cv,    ypc_metric_xgb_cv    = y_prediction_cv(baseline_model_xgb_pc, xpc_train, y_train, "ypc_pred_xgb")
    yss_pred_xgb_cv,    yss_metric_xgb_cv    = y_prediction_cv(baseline_model_xgb_ss, xss_train, y_train, "yss_pred_xgb")
    ycd_pred_xgb_cv,    ycd_metric_xgb_cv    = y_prediction_cv(baseline_model_xgb_cd, xcd_train, y_train, "ycd_pred_xgb")
    ycn_pred_xgb_cv,    ycn_metric_xgb_cv    = y_prediction_cv(baseline_model_xgb_cn, xcn_train, y_train, "ycn_pred_xgb")
    ykc_pred_xgb_cv,    ykc_metric_xgb_cv    = y_prediction_cv(baseline_model_xgb_kc, xkc_train, y_train, "ykc_pred_xgb")
    yce_pred_xgb_cv,    yce_metric_xgb_cv    = y_prediction_cv(baseline_model_xgb_ce, xce_train, y_train, "yce_pred_xgb")
    ysc_pred_xgb_cv,    ysc_metric_xgb_cv    = y_prediction_cv(baseline_model_xgb_sc, xsc_train, y_train, "ysc_pred_xgb")
    yac_pred_xgb_cv,    yac_metric_xgb_cv    = y_prediction_cv(baseline_model_xgb_ac, xac_train, y_train, "yac_pred_xgb")
    yma_pred_xgb_cv,    yma_metric_xgb_cv    = y_prediction_cv(baseline_model_xgb_ma, xma_train, y_train, "yma_pred_xgb")
    baseline_model_svc_at = SVC(kernel='rbf',gamma=0.1,C=10,random_state=1).fit(xat_train, y_train)
    baseline_model_svc_ke = SVC(kernel='rbf',gamma=0.1,C=10,random_state=1).fit(xke_train, y_train)
    baseline_model_svc_es = SVC(kernel='rbf',gamma=0.1,C=10,random_state=1).fit(xes_train, y_train)
    baseline_model_svc_pc = SVC(kernel='rbf',gamma=0.1,C=10,random_state=1).fit(xpc_train, y_train)
    baseline_model_svc_ss = SVC(kernel='rbf',gamma=0.1,C=10,random_state=1).fit(xss_train, y_train)
    baseline_model_svc_cd = SVC(kernel='rbf',gamma=0.1,C=10,random_state=1).fit(xcd_train, y_train)
    baseline_model_svc_cn = SVC(kernel='rbf',gamma=0.1,C=10,random_state=1).fit(xcn_train, y_train) #500
    baseline_model_svc_kc = SVC(kernel='rbf',gamma=0.1,C=10,random_state=1).fit(xkc_train, y_train)
    baseline_model_svc_ce = SVC(kernel='rbf',gamma=0.1,C=10,random_state=1).fit(xce_train, y_train)
    baseline_model_svc_sc = SVC(kernel='rbf',gamma=0.1,C=10,random_state=1).fit(xsc_train, y_train)
    baseline_model_svc_ac = SVC(kernel='rbf',gamma=0.1,C=10,random_state=1).fit(xac_train, y_train)
    baseline_model_svc_ma = SVC(kernel='rbf',gamma=0.1,C=10,random_state=1).fit(xma_train, y_train)
    dump(baseline_model_svc_at, os.path.join(name,"train", "baseline_model_svc_at.joblib"))
    dump(baseline_model_svc_ke, os.path.join(name,"train", "baseline_model_svc_ke.joblib"))
    dump(baseline_model_svc_es, os.path.join(name,"train", "baseline_model_svc_es.joblib"))
    dump(baseline_model_svc_pc, os.path.join(name,"train", "baseline_model_svc_pc.joblib"))
    dump(baseline_model_svc_ss, os.path.join(name,"train", "baseline_model_svc_ss.joblib"))
    dump(baseline_model_svc_cd, os.path.join(name,"train", "baseline_model_svc_cd.joblib"))
    dump(baseline_model_svc_cn, os.path.join(name,"train", "baseline_model_svc_cn.joblib"))
    dump(baseline_model_svc_kc, os.path.join(name,"train", "baseline_model_svc_kc.joblib"))
    dump(baseline_model_svc_ce, os.path.join(name,"train", "baseline_model_svc_ce.joblib"))
    dump(baseline_model_svc_sc, os.path.join(name,"train", "baseline_model_svc_sc.joblib"))
    dump(baseline_model_svc_ac, os.path.join(name,"train", "baseline_model_svc_ac.joblib"))
    dump(baseline_model_svc_ma, os.path.join(name,"train", "baseline_model_svc_ma.joblib"))
    yat_pred_svc_train, yat_metric_svc_train = y_prediction(   baseline_model_svc_at, xat_train, y_train, "yat_pred_svc")
    yes_pred_svc_train, yes_metric_svc_train = y_prediction(   baseline_model_svc_es, xes_train, y_train, "yes_pred_svc")
    yke_pred_svc_train, yke_metric_svc_train = y_prediction(   baseline_model_svc_ke, xke_train, y_train, "yke_pred_svc")
    ypc_pred_svc_train, ypc_metric_svc_train = y_prediction(   baseline_model_svc_pc, xpc_train, y_train, "ypc_pred_svc")
    yss_pred_svc_train, yss_metric_svc_train = y_prediction(   baseline_model_svc_ss, xss_train, y_train, "yss_pred_svc")
    ycd_pred_svc_train, ycd_metric_svc_train = y_prediction(   baseline_model_svc_cd, xcd_train, y_train, "ycd_pred_svc")
    ycn_pred_svc_train, ycn_metric_svc_train = y_prediction(   baseline_model_svc_cn, xcn_train, y_train, "ycn_pred_svc")
    ykc_pred_svc_train, ykc_metric_svc_train = y_prediction(   baseline_model_svc_kc, xkc_train, y_train, "ykc_pred_svc")
    yce_pred_svc_train, yce_metric_svc_train = y_prediction(   baseline_model_svc_ce, xce_train, y_train, "yce_pred_svc")
    ysc_pred_svc_train, ysc_metric_svc_train = y_prediction(   baseline_model_svc_sc, xsc_train, y_train, "ysc_pred_svc")
    yac_pred_svc_train, yac_metric_svc_train = y_prediction(   baseline_model_svc_ac, xac_train, y_train, "yac_pred_svc")
    yma_pred_svc_train, yma_metric_svc_train = y_prediction(   baseline_model_svc_ma, xma_train, y_train, "yma_pred_svc")
    yat_pred_svc_test,  yat_metric_svc_test   = y_prediction(  baseline_model_svc_at, xat_test, y_test,   "yat_pred_svc")
    yes_pred_svc_test,  yes_metric_svc_test   = y_prediction(  baseline_model_svc_es, xes_test, y_test,   "yes_pred_svc")
    yke_pred_svc_test,  yke_metric_svc_test   = y_prediction(  baseline_model_svc_ke, xke_test, y_test,   "yke_pred_svc")
    ypc_pred_svc_test,  ypc_metric_svc_test   = y_prediction(  baseline_model_svc_pc, xpc_test, y_test,   "ypc_pred_svc")
    yss_pred_svc_test,  yss_metric_svc_test   = y_prediction(  baseline_model_svc_ss, xss_test, y_test,   "yss_pred_svc")
    ycd_pred_svc_test,  ycd_metric_svc_test   = y_prediction(  baseline_model_svc_cd, xcd_test, y_test,   "ycd_pred_svc")
    ycn_pred_svc_test,  ycn_metric_svc_test   = y_prediction(  baseline_model_svc_cn, xcn_test, y_test,   "ycn_pred_svc")
    ykc_pred_svc_test,  ykc_metric_svc_test   = y_prediction(  baseline_model_svc_kc, xkc_test, y_test,   "ykc_pred_svc")
    yce_pred_svc_test,  yce_metric_svc_test   = y_prediction(  baseline_model_svc_ce, xce_test, y_test,   "yce_pred_svc")
    ysc_pred_svc_test,  ysc_metric_svc_test   = y_prediction(  baseline_model_svc_sc, xsc_test, y_test,   "ysc_pred_svc")
    yac_pred_svc_test,  yac_metric_svc_test   = y_prediction(  baseline_model_svc_ac, xac_test, y_test,   "yac_pred_svc")
    yma_pred_svc_test,  yma_metric_svc_test   = y_prediction(  baseline_model_svc_ma, xma_test, y_test,   "yma_pred_svc")
    yat_pred_svc_cv,    yat_metric_svc_cv    = y_prediction_cv(baseline_model_svc_at, xat_train, y_train, "yat_pred_svc")
    yes_pred_svc_cv,    yes_metric_svc_cv    = y_prediction_cv(baseline_model_svc_es, xes_train, y_train, "yes_pred_svc")
    yke_pred_svc_cv,    yke_metric_svc_cv    = y_prediction_cv(baseline_model_svc_ke, xke_train, y_train, "yke_pred_svc")
    ypc_pred_svc_cv,    ypc_metric_svc_cv    = y_prediction_cv(baseline_model_svc_pc, xpc_train, y_train, "ypc_pred_svc")
    yss_pred_svc_cv,    yss_metric_svc_cv    = y_prediction_cv(baseline_model_svc_ss, xss_train, y_train, "yss_pred_svc")
    ycd_pred_svc_cv,    ycd_metric_svc_cv    = y_prediction_cv(baseline_model_svc_cd, xcd_train, y_train, "ycd_pred_svc")
    ycn_pred_svc_cv,    ycn_metric_svc_cv    = y_prediction_cv(baseline_model_svc_cn, xcn_train, y_train, "ycn_pred_svc")
    ykc_pred_svc_cv,    ykc_metric_svc_cv    = y_prediction_cv(baseline_model_svc_kc, xkc_train, y_train, "ykc_pred_svc")
    yce_pred_svc_cv,    yce_metric_svc_cv    = y_prediction_cv(baseline_model_svc_ce, xce_train, y_train, "yce_pred_svc")
    ysc_pred_svc_cv,    ysc_metric_svc_cv    = y_prediction_cv(baseline_model_svc_sc, xsc_train, y_train, "ysc_pred_svc")
    yac_pred_svc_cv,    yac_metric_svc_cv    = y_prediction_cv(baseline_model_svc_ac, xac_train, y_train, "yac_pred_svc")
    yma_pred_svc_cv,    yma_metric_svc_cv    = y_prediction_cv(baseline_model_svc_ma, xma_train, y_train, "yma_pred_svc")
    print("Stacking")
    #stack
    stack_train =pd.concat([yat_pred_rf_train, yat_pred_xgb_train, yat_pred_svc_train,
                            yes_pred_rf_train, yes_pred_xgb_train, yes_pred_svc_train,
                            yke_pred_rf_train, yke_pred_xgb_train, yke_pred_svc_train,
                            ypc_pred_rf_train, ypc_pred_xgb_train, ypc_pred_svc_train,
                            yss_pred_rf_train, yss_pred_xgb_train, yss_pred_svc_train,
                            ycd_pred_rf_train, ycd_pred_xgb_train, ycd_pred_svc_train,
                            ycn_pred_rf_train, ycn_pred_xgb_train, ycn_pred_svc_train,
                            ykc_pred_rf_train, ykc_pred_xgb_train, ykc_pred_svc_train,
                            yce_pred_rf_train, yce_pred_xgb_train, yce_pred_svc_train,
                            ysc_pred_rf_train, ysc_pred_xgb_train, ysc_pred_svc_train,
                            yac_pred_rf_train, yac_pred_xgb_train, yac_pred_svc_train,
                            yma_pred_rf_train, yma_pred_xgb_train, yma_pred_svc_train,],  axis=1)
    stack_cv    =pd.concat([yat_pred_rf_cv,    yat_pred_xgb_cv,    yat_pred_svc_cv,
                            yes_pred_rf_cv,    yes_pred_xgb_cv,    yes_pred_svc_cv,
                            yke_pred_rf_cv,    yke_pred_xgb_cv,    yke_pred_svc_cv,
                            ypc_pred_rf_cv,    ypc_pred_xgb_cv,    ypc_pred_svc_cv,
                            yss_pred_rf_cv,    yss_pred_xgb_cv,    yss_pred_svc_cv,
                            ycd_pred_rf_cv,    ycd_pred_xgb_cv,    ycd_pred_svc_cv,
                            ycn_pred_rf_cv,    ycn_pred_xgb_cv,    ycn_pred_svc_cv,
                            ykc_pred_rf_cv,    ykc_pred_xgb_cv,    ykc_pred_svc_cv,
                            yce_pred_rf_cv,    yce_pred_xgb_cv,    yce_pred_svc_cv,
                            ysc_pred_rf_cv,    ysc_pred_xgb_cv,    ysc_pred_svc_cv,
                            yac_pred_rf_cv,    yac_pred_xgb_cv,    yac_pred_svc_cv,
                            yma_pred_rf_cv,    yma_pred_xgb_cv,    yma_pred_svc_cv,],  axis=1)
    stack_test  =pd.concat([yat_pred_rf_test,  yat_pred_xgb_test,  yat_pred_svc_test,
                            yes_pred_rf_test,  yes_pred_xgb_test,  yes_pred_svc_test,
                            yke_pred_rf_test,  yke_pred_xgb_test,  yke_pred_svc_test,
                            ypc_pred_rf_test,  ypc_pred_xgb_test,  ypc_pred_svc_test,
                            yss_pred_rf_test,  yss_pred_xgb_test,  yss_pred_svc_test,
                            ycd_pred_rf_test,  ycd_pred_xgb_test,  ycd_pred_svc_test,
                            ycn_pred_rf_test,  ycn_pred_xgb_test,  ycn_pred_svc_test,
                            ykc_pred_rf_test,  ykc_pred_xgb_test,  ykc_pred_svc_test,
                            yce_pred_rf_test,  yce_pred_xgb_test,  yce_pred_svc_test,
                            ysc_pred_rf_test,  ysc_pred_xgb_test,  ysc_pred_svc_test,
                            yac_pred_rf_test,  yac_pred_xgb_test,  yac_pred_svc_test,
                            yma_pred_rf_test,  yma_pred_xgb_test,  yma_pred_svc_test,],  axis=1)
    #save stack original
    stack_train.to_csv(os.path.join( name, 'stack', "stacked_train.csv"))
    stack_cv.to_csv(os.path.join(    name, 'stack', "stacked_cv.csv"))
    stack_test.to_csv(os.path.join(  name, 'stack', "stacked_test.csv"))
    #Train
    stacked_xgb = xgb.XGBClassifier(objective="binary:logistic",eval_metric='auc', random_state=1)
    
    param_grid = {
    'max_depth': [3, 5],
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    }
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=stacked_xgb, param_grid=param_grid, scoring='roc_auc', cv=5, verbose=1)
    grid_search.fit(stack_cv, y_train)
    best_params = grid_search.best_params_
    print("Best parameters found: ", best_params)
    print("Best AUC found: ", grid_search.best_score_)
    stacked_model = xgb.XGBClassifier(**best_params)
    stacked_model.set_params(objective='binary:logistic', eval_metric='auc', random_state=1)
    stacked_model.fit(stack_cv, y_train)
    dump(stacked_model, os.path.join(name, 'stack', "stacked_model.joblib"))

    y_pred_stk_train,  y_metric_stk_train = y_prediction(   stacked_model, stack_train, y_train, "y_pred_stacked")
    y_pred_stk_test ,  y_metric_stk_test  = y_prediction(   stacked_model, stack_test,  y_test,  "y_pred_stacked")
    y_pred_stk_cv   ,  y_metric_stk_cv    = y_prediction(   stacked_model, stack_cv,    y_train, "y_pred_stacked")
    y_pred_stk_train.to_csv(os.path.join( name, 'stack', "y_pred_train.csv"))
    y_pred_stk_test .to_csv(os.path.join( name, 'stack', "y_pred_test.csv"))
    y_pred_stk_cv   .to_csv(os.path.join( name, 'stack', "y_pred_cv.csv"))
    #combine metrics
    metric_train= pd.concat([yat_metric_rf_train, yat_metric_xgb_train, yat_metric_svc_train,
                            yes_metric_rf_train, yes_metric_xgb_train,  yes_metric_svc_train,
                            yke_metric_rf_train, yke_metric_xgb_train,  yke_metric_svc_train,
                            ypc_metric_rf_train, ypc_metric_xgb_train,  ypc_metric_svc_train,
                            yss_metric_rf_train, yss_metric_xgb_train,  yss_metric_svc_train,
                            ycd_metric_rf_train, ycd_metric_xgb_train,  ycd_metric_svc_train,
                            ycn_metric_rf_train, ycn_metric_xgb_train,  ycn_metric_svc_train,
                            ykc_metric_rf_train, ykc_metric_xgb_train,  ykc_metric_svc_train,
                            yce_metric_rf_train, yce_metric_xgb_train,  yce_metric_svc_train,
                            ysc_metric_rf_train, ysc_metric_xgb_train,  ysc_metric_svc_train,
                            yac_metric_rf_train, yac_metric_xgb_train,  yac_metric_svc_train,
                            yma_metric_rf_train, yma_metric_xgb_train,  yma_metric_svc_train, y_metric_stk_train],  axis=0)
    metric_cv  = pd.concat([yat_metric_rf_cv, yat_metric_xgb_cv,        yat_metric_svc_cv,
                            yes_metric_rf_cv, yes_metric_xgb_cv,        yes_metric_svc_cv,
                            yke_metric_rf_cv, yke_metric_xgb_cv,        yke_metric_svc_cv,
                            ypc_metric_rf_cv, ypc_metric_xgb_cv,        ypc_metric_svc_cv,
                            yss_metric_rf_cv, yss_metric_xgb_cv,        yss_metric_svc_cv,
                            ycd_metric_rf_cv, ycd_metric_xgb_cv,        ycd_metric_svc_cv,
                            ycn_metric_rf_cv, ycn_metric_xgb_cv,        ycn_metric_svc_cv,
                            ykc_metric_rf_cv, ykc_metric_xgb_cv,        ykc_metric_svc_cv,
                            yce_metric_rf_cv, yce_metric_xgb_cv,        yce_metric_svc_cv,
                            ysc_metric_rf_cv, ysc_metric_xgb_cv,        ysc_metric_svc_cv,
                            yac_metric_rf_cv, yac_metric_xgb_cv,        yac_metric_svc_cv,
                            yma_metric_rf_cv, yma_metric_xgb_cv,        yma_metric_svc_cv, y_metric_stk_cv],  axis=0)
    metric_test= pd.concat([yat_metric_rf_test, yat_metric_xgb_test,    yat_metric_svc_test,
                            yes_metric_rf_test, yes_metric_xgb_test,    yes_metric_svc_test,
                            yke_metric_rf_test, yke_metric_xgb_test,    yke_metric_svc_test,
                            ypc_metric_rf_test, ypc_metric_xgb_test,    ypc_metric_svc_test,
                            yss_metric_rf_test, yss_metric_xgb_test,    yss_metric_svc_test,
                            ycd_metric_rf_test, ycd_metric_xgb_test,    ycd_metric_svc_test,
                            ycn_metric_rf_test, ycn_metric_xgb_test,    ycn_metric_svc_test,
                            ykc_metric_rf_test, ykc_metric_xgb_test,    ykc_metric_svc_test,
                            yce_metric_rf_test, yce_metric_xgb_test,    yce_metric_svc_test,
                            ysc_metric_rf_test, ysc_metric_xgb_test,    ysc_metric_svc_test,
                            yac_metric_rf_test, yac_metric_xgb_test,    yac_metric_svc_test,
                            yma_metric_rf_test, yma_metric_xgb_test,    yma_metric_svc_test, y_metric_stk_test],  axis=0)
    metric_train.to_csv(os.path.join( name, "metric_train.csv"))
    metric_cv   .to_csv(os.path.join( name, "metric_cv.csv"))
    metric_test .to_csv(os.path.join( name, "metric_test.csv"))
    return stacked_model, stack_train, stack_cv, stack_test, metric_train, metric_test, metric_cv


def nearest_neighbor_AD(x_train, x_test, name, k, z=3):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean').fit(x_train)
    dump(nn, os.path.join(name, "ad_"+ str(k) +"_"+ str(z) +".joblib"))
    distance, index = nn.kneighbors(x_test)
    # Calculate mean and sd of distance in train set
    di = np.mean(distance, axis=1)
    # Find mean and sd of di
    dk = np.mean(di)
    sk = np.std(di)
    print('dk = ', dk)
    print('sk = ', sk)
    AD_status = ['within_AD' if di[i] < dk + (z * sk) else 'outside_AD' for i in range(len(di))]

    # Create DataFrame with index from x_test and the respective status
    df = pd.DataFrame(AD_status, index=x_test.index, columns=['AD_status'])
    return df

def run_ad(stacked_model, stack_cv, stack_test, y_test, name, z = 0.5):
    # Initialize lists to store metrics for plotting
    k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    MCC_values = []
    ACC_values = []
    Sen_values = []
    Spe_values = []
    AUC_values = []
    F1_values = []
    BA_values = []
    removed_compounds_values = []

    # Remove outside AD
    for i in k_values:
        print('k = ', i, 'z=', str(z))
        t = nearest_neighbor_AD(stack_cv, stack_test, name, i, z=z)
        print(t['AD_status'].value_counts())
        # Remove outside AD
        x_ad_test = stack_test[t['AD_status'] == 'within_AD']
        y_ad_test = y_test.loc[x_ad_test.index]
        y_pred_test = stacked_model.predict(x_ad_test)
        print(len(x_ad_test),len(y_ad_test), len(y_pred_test) )
        # Evaluation
        print('Test set')
        accuracy = round(accuracy_score(y_ad_test, y_pred_test), 3)
        conf_matrix = confusion_matrix(y_ad_test, y_pred_test)
        F1 = round(f1_score(y_ad_test, y_pred_test, average='weighted'), 3)
        tn, fp, fn, tp = conf_matrix.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        auc = roc_auc_score(y_ad_test, y_pred_test)
        mcc = round(matthews_corrcoef(y_ad_test, y_pred_test), 3)
        balanced_acc = round(balanced_accuracy_score(y_ad_test, y_pred_test), 3)
        print('ACC: ', accuracy, 'Sen: ', sensitivity, 'Spe: ', specificity, 'AUC: ', auc, 'MCC: ', mcc, 'F1: ', F1, 'BA: ', balanced_acc)
        # Store metrics for plotting
        MCC_values.append(mcc)
        ACC_values.append(accuracy)
        Sen_values.append(sensitivity)
        Spe_values.append(specificity)
        AUC_values.append(auc)
        F1_values.append(F1)
        BA_values.append(balanced_acc)
        removed_compounds_values.append((t['AD_status'] == 'outside_AD').sum())
    k_values   = np.array(k_values)
    MCC_values = np.array(MCC_values)
    ACC_values = np.array(ACC_values)
    Sen_values = np.array(Sen_values)
    Spe_values = np.array(Spe_values)
    AUC_values = np.array(AUC_values)
    F1_values  = np.array(F1_values)
    BA_values  = np.array(BA_values)
    removed_compounds_values = np.array(removed_compounds_values)
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
    ax1.plot(k_values, MCC_values, 'r^-', label = "MCC")
    ax1.plot(k_values, ACC_values, 'kp-', label = "ACC")
    ax1.plot(k_values, Sen_values, 'gs-', label = "Sen")
    ax1.plot(k_values, Spe_values, 'y*-', label = "Spe")
    ax1.plot(k_values, AUC_values, 'md-', label = "AUC")
    ax1.plot(k_values, F1_values, 'cX-',  label = "F1")
    ax1.plot(k_values, BA_values, 'bo-',  label = "BA")
    # Adding labels and title
    ax1.set_xlabel('k',      fontsize=12, fontstyle='italic',weight="bold")
    ax1.set_ylabel('Scores', fontsize=12, fontstyle='italic', weight='bold')
    ax1.set_xticks(k_values)
    ax1.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1.02))
    # Second plot: Bar plot for removed_compounds_values
    ax2.bar(k_values, removed_compounds_values, color='green', edgecolor='black', alpha=0.5, width=0.3)
    ax2.set_xlabel('k', fontsize=12, fontstyle='italic',weight="bold")
    ax2.set_ylabel('Removed compounds', fontsize=12, fontstyle='italic', weight='bold')
    ax2.set_xticks(k_values)
    plt.tight_layout()
    plt.savefig("AD_ToxSTK_"+name+"_"+ str(z)+ "_Classification_separated.svg", bbox_inches='tight') 
    plt.close

def y_random(stack_train, stack_cv, stack_test, y_train, y_test, metric_train, metric_test, name):
    MCC_test=[]
    MCC_train=[]
    for i in range(1,101):
      y_train=y_train.sample(frac=1,replace=False,random_state=0)
      y_test=y_test.sample(frac=1,replace=False,random_state=0)
      model=RandomForestClassifier(max_depth=5, max_features=10,n_estimators=300).fit(stack_cv, y_train)
      y_pred_MCCext=model.predict(stack_test)
      y_pred_MCCtrain=model.predict(stack_train)
      MCCext=matthews_corrcoef(y_test, y_pred_MCCext)
      MCC_test.append(MCCext)
      MCCtrain=matthews_corrcoef(y_train, y_pred_MCCtrain)
      MCC_train.append(MCCtrain)
    size=[50]
    sizes=[20]
    x=[metric_train.loc['y_pred_stacked', 'MCC']]
    y=[metric_test.loc['y_pred_stacked', 'MCC']]
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.axvline(0.5, c='black', ls=':')
    ax.axhline(0.5, c='black', ls=':')
    ax.scatter(x,y,s=size,c=['red'],marker='x', label='Our model')
    ax.scatter(MCC_train,MCC_test, c='blue',edgecolors='black', alpha=0.7, s=sizes, label='Y-randomization')
    ax.set_xlabel('$MCC_{Train}$', fontsize=14,  fontstyle='italic', weight='bold')
    ax.set_ylabel('$MCC_{Test}$', fontsize=14,  fontstyle='italic', weight='bold')
    ax.legend(loc='lower right',fontsize='small')
    # Adjust layout
    plt.tight_layout()
    plt.savefig("Y-randomization-"+name+"-classification.svg", bbox_inches='tight')
    # Show the plots
    plt.close()

def plot_importance_xgb(model, name):
    importance = model.get_booster().get_score(importance_type='gain')
    features_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)
    print(features_df.head(10))
    plt.figure(figsize=(4, 4))
    sns.barplot(x='Importance', y='Feature', data=features_df)
    plt.xlabel("Importance scores", fontsize=12, fontstyle='italic',weight="bold")
    plt.ylabel("Features", fontsize=12, fontstyle='italic',weight="bold")
    plt.tight_layout()
    plt.savefig('top_features_xgb_'+name+'.svg', bbox_inches='tight')
    plt.close()

def main():
    for name in ['nafld']: #, 
        print("#"*100) 
        print(name)
        y_train  = pd.read_csv(os.path.join(name,"train", "y_train.csv"), index_col=0)
        y_test   = pd.read_csv(os.path.join(name,"test" , "y_test.csv" ), index_col=0)
        print("Y_train")
        print(y_train)
        print("Y_test")
        print(y_test)
        stacked_model, stack_train, stack_cv, stack_test, metric_train, metric_test, metric_cv = stacked_class(name)
        print("finish train ", name)
        y_random(stack_train, stack_cv, stack_test, y_train, y_test, metric_train, metric_test, name)
        print("finish yrandom ", name)
        plot_importance_xgb(stacked_model, name)
        print("finish top features ", name)
        run_ad(stacked_model, stack_cv, stack_test, y_test, name, z=1)
        print("finish ad ", name)
if __name__ == "__main__":
    main()
        
        