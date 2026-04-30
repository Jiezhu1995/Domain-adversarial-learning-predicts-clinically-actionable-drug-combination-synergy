#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import joblib
import skopt
import tensorflow as tf
import h5py
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
from collections import Counter
#from tffm import TFFMClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from xgboost.sklearn import XGBRegressor
from sklearn import linear_model
from sklearn.linear_model import RidgeCV, LassoCV
from scipy.stats import pearsonr
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from concrete_autoencoder import ConcreteAutoencoderFeatureSelector
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, LeakyReLU, Softmax


# In[2]:


# Load training data
with h5py.File('/Users/sujie/Desktop/project2/BeatAML/training_data_blood.h5', 'r') as f:
    X_train = f['X_train'][:]
    y_train = f['y_train'][:]

# Load test data
with h5py.File('/Users/sujie/Desktop/project2/BeatAML/test_data_blood.h5', 'r') as f:
    X_test = f['X_test'][:]
    y_test = f['y_test'][:]


# In[3]:


# Create scaler object
scaler = StandardScaler()

# Fit scaler to training data and transform it
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[5]:


rowDrug_MACCS_patient_PB = pd.read_csv("/Users/sujie/Desktop/project2/BeatAML/input_PB/rowDrug_MACCS_input_PB.csv", sep = ",")
rowDrug_Sig_patient_PB = pd.read_csv("/Users/sujie/Desktop/project2/BeatAML/input_PB/rowDrug_Sig_input_PB.csv", sep = ",")
colDrug_MACCS_patient_PB = pd.read_csv("/Users/sujie/Desktop/project2/BeatAML/input_PB/colDrug_MACCS_input_PB.csv", sep = ",")
colDrug_Sig_patient_PB = pd.read_csv("/Users/sujie/Desktop/project2/BeatAML/input_PB/colDrug_Sig_input_PB.csv", sep = ",")
CellExp_patient_PB = pd.read_csv("/Users/sujie/Desktop/project2/BeatAML/input_PB/cellExp_ssGSEA_input_PB_1221.csv", sep = ",")
HSA_label_patient_PB = pd.read_csv("/Users/sujie/Desktop/project2/BeatAML/input_PB/HSA_label_PB.csv", sep = ",")


# In[6]:


input__patient_PB=pd.concat([rowDrug_MACCS_patient_PB,rowDrug_Sig_patient_PB,colDrug_MACCS_patient_PB,colDrug_Sig_patient_PB,CellExp_patient_PB,HSA_label_patient_PB],axis=1,join="inner") 
input_patient_PB_final=input__patient_PB.drop(labels = ['drug_row',"block_id","drug_col","SampleID"], axis=1)
new_data_patient_PB=input_patient_PB_final
X_patient_PB = new_data_patient_PB.drop("synergy_hsa", axis = 1).values
y_patient_PB = new_data_patient_PB.synergy_hsa.astype(float)
#X_patient_train, X_patient_test, y_patient_train, y_patient_test = train_test_split(X_patient, y_patient, train_size = 0.8, test_size = 0.2,random_state=0)


# In[7]:


# Fit scaler to training data and transform it
#X_patient_BM_scaled = scaler.transform(X_patient_BM)
# Fit scaler to training data and transform it
X_patient_PB_scaled = scaler.transform(X_patient_PB)
X_test_PB = X_patient_PB_scaled
y_test_PB = y_patient_PB


# In[8]:


print (X_train_scaled.shape[1])


# # Error prediction model

# In[60]:


test_data = pd.read_csv('BM_test_remaining_samples.csv')


# In[58]:


# =========================
from tensorflow.keras.models import load_model

import tensorflow as tf

class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return self.grad_reverse(inputs)

    @tf.custom_gradient
    def grad_reverse(self, x):
        y = tf.identity(x)

        def custom_grad(dy):
            return -dy

        return y, custom_grad


DANN = load_model(
    "DANN_50_samples_AML.keras",
    custom_objects={"GradientReversalLayer": GradientReversalLayer}
)


# In[61]:


import numpy as np
import pandas as pd
import random

from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt


# ============================================================
# Aim: improve selective/Mondrian prediction performance
def build_error_features(X_scaled, y_pred):
    """
    Augment original features with the model's own prediction signals.
    This often improves uncertainty ranking.
    """
    y_pred = np.asarray(y_pred).reshape(-1, 1)
    feats = np.concatenate(
        [
            X_scaled,
            y_pred,
            np.abs(y_pred),
            (y_pred ** 2),
        ],
        axis=1
    )
    return feats


import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

def fit_error_model_crossfit_rf(
    Xerr_cal,
    err_target_cal,
    n_splits=3,
    random_state=0,
    return_oof=False
):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # out-of-fold predictions (diagnostic)
    oof_pred = np.full(len(err_target_cal), np.nan, dtype=float)

    # 
    rf_params = dict(
        n_estimators=300,        #
        max_depth=6,             # 
        min_samples_leaf=30,     # 
        min_samples_split=10,
        max_features="sqrt",
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=random_state
    )

    # cross-fit OOF
    for tr_idx, va_idx in kf.split(Xerr_cal):
        X_tr, X_va = Xerr_cal[tr_idx], Xerr_cal[va_idx]
        y_tr = err_target_cal[tr_idx]

        fold_model = RandomForestRegressor(**rf_params)
        fold_model.fit(X_tr, y_tr)
        oof_pred[va_idx] = fold_model.predict(X_va)

    # final model on all calibration data
    final_model = RandomForestRegressor(**rf_params)
    final_model.fit(Xerr_cal, err_target_cal)

    if return_oof:
        return final_model, oof_pred
    return final_model


def selective_threshold_global(err_pred_test, kept_ratio):
    """
    Choose a threshold so that approx kept_ratio of samples are kept.
    Lower predicted error = keep.
    """
    kept_ratio = float(kept_ratio)
    kept_ratio = min(max(kept_ratio, 0.0), 1.0)
    thr = np.quantile(err_pred_test, kept_ratio)
    keep = err_pred_test <= thr
    return keep, thr


def mondrian_threshold_by_bins(err_pred_test, bin_id_test, kept_ratio):
    """
    Mondrian selective rule: apply the SAME kept_ratio within each bin.
    Returns boolean keep mask (same length as err_pred_test).
    """
    keep = np.zeros(len(err_pred_test), dtype=bool)
    for b in np.unique(bin_id_test):
        idx = np.where(bin_id_test == b)[0]
        if len(idx) == 0:
            continue
        thr = np.quantile(err_pred_test[idx], kept_ratio)
        keep[idx] = err_pred_test[idx] <= thr
    return keep


def make_bins_by_pred(y_pred, n_bins=5):
    """
    Mondrian bins based on prediction quantiles.
    (You can change to bins by SampleID group, drug class, etc.)
    """
    y_pred = np.asarray(y_pred).reshape(-1)
    qs = np.quantile(y_pred, np.linspace(0, 1, n_bins + 1))
    # ensure strictly increasing edges (handle ties)
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    bin_id = np.digitize(y_pred, qs[1:-1], right=True)  # 0..n_bins-1
    return bin_id


# ----------------------------
# Main experiment runner
# ----------------------------
def run_selective_experiment(
    test_data,
    scaler,
    DANN,
    n_reps=50,
    alpha_percentiles=range(50, 101, 10),  # 
    use_mondrian=True,
    mondrian_bins=5,
    random_seed=0
):
    random.seed(random_seed)
    np.random.seed(random_seed)

    # If you want kept_ratio targets corresponding to percentiles, use:
    # alpha_percentile=50 -> kept_ratio roughly 0.50, etc.
    kept_ratio_targets = [p / 100.0 for p in alpha_percentiles]

    # Store per-point results (NOT averaged raw; we average later by kept_ratio)
    rows = []

    for rep in range(n_reps):
        grouped = list(test_data.groupby("SampleID"))
        random.shuffle(grouped)

        calib_groups, test_groups = [], []
        for k, g in enumerate(grouped):
            (calib_groups if k % 2 == 0 else test_groups).append(g)

        def groups_to_df(groups):
            return pd.concat([df for _, df in groups], ignore_index=True)

        calibration_set = groups_to_df(calib_groups)
        new_test_set    = groups_to_df(test_groups)

        # Scale
        X_test = scaler.transform(new_test_set.drop(columns=["SampleID", "synergy_hsa"]).values)
        y_test = new_test_set["synergy_hsa"].values

        X_cal  = scaler.transform(calibration_set.drop(columns=["SampleID", "synergy_hsa"]).values)
        y_cal  = calibration_set["synergy_hsa"].values

        # DANN predictions
        y_cal_pred  = np.squeeze(DANN.predict(X_cal,  verbose=0)[0])
        y_test_pred = np.squeeze(DANN.predict(X_test, verbose=0)[0])

        # ---- uncertainty target: log-error (recommended) ----
        cal_abs_err = np.abs(y_cal - y_cal_pred)
        cal_err_target = np.log(cal_abs_err + 1e-3)  # log-error

        # ---- uncertainty features: X + y_pred signals ----
        Xerr_cal  = build_error_features(X_cal,  y_cal_pred)
        Xerr_test = build_error_features(X_test, y_test_pred)

        # fit uncertainty model
        err_model =fit_error_model_crossfit_rf(Xerr_cal, cal_err_target, random_state=random_seed+rep)

        # predict uncertainty on test
        err_pred_test = err_model.predict(Xerr_test)

        # raw metrics (full test set)
        sp_raw, _ = spearmanr(y_test, y_test_pred)
        pe_raw, _ = pearsonr(y_test, y_test_pred)
        mse_raw   = mean_squared_error(y_test, y_test_pred)

        # Mondrian bins (optional)
        if use_mondrian:
            bin_id_test = make_bins_by_pred(y_test_pred, n_bins=mondrian_bins)

        for kept_ratio in kept_ratio_targets:
            if use_mondrian:
                keep = mondrian_threshold_by_bins(err_pred_test, bin_id_test, kept_ratio)
            else:
                keep, _ = selective_threshold_global(err_pred_test, kept_ratio)

            X_keep = X_test[keep]
            y_keep = y_test[keep]

            kept_ratio_emp = len(y_keep) / max(1, len(y_test))

            if len(y_keep) < 3:
                # too few points for correlations
                rows.append({
                    "rep": rep + 1,
                    "kept_ratio_target": kept_ratio,
                    "kept_ratio_emp": kept_ratio_emp,
                    "spearman_kept": np.nan,
                    "pearson_kept": np.nan,
                    "mse_kept": np.nan,
                    "spearman_raw": sp_raw,
                    "pearson_raw": pe_raw,
                    "mse_raw": mse_raw,
                })
                continue

            y_keep_pred = np.squeeze(DANN.predict(X_keep, verbose=0)[0])

            sp_keep, _ = spearmanr(y_keep, y_keep_pred)
            pe_keep, _ = pearsonr(y_keep, y_keep_pred)
            mse_keep   = mean_squared_error(y_keep, y_keep_pred)

            rows.append({
                "rep": rep + 1,
                "kept_ratio_target": kept_ratio,
                "kept_ratio_emp": kept_ratio_emp,
                "spearman_kept": sp_keep,
                "pearson_kept": pe_keep,
                "mse_kept": mse_keep,
                "spearman_raw": sp_raw,
                "pearson_raw": pe_raw,
                "mse_raw": mse_raw,
            })

    return pd.DataFrame(rows)


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# -------------------------
# utils
# -------------------------
def _safe_sem(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size <= 1:
        return np.nan
    return x.std(ddof=1) / np.sqrt(x.size)

def _p_to_star(p):
    if np.isnan(p):
        return None
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return None

def _wilcoxon_by_kept_ratio(df, col_kept, col_raw, alternative="two-sided"):
    """
    Return dict: kept_ratio_target -> p-value
    alternative:
      - "two-sided"
      - "greater"  (kept > raw)  : good for correlations
      - "less"     (kept < raw)  : good for MSE (smaller is better)
    """
    out = {}
    for kr, g in df.groupby("kept_ratio_target"):
        kept = g[col_kept].to_numpy(float)
        raw  = g[col_raw].to_numpy(float)
        mask = ~np.isnan(kept) & ~np.isnan(raw)
        kept, raw = kept[mask], raw[mask]

        if len(kept) < 5:
            out[kr] = np.nan
            continue
        if np.allclose(kept - raw, 0):
            out[kr] = 1.0
            continue

        try:
            _, p = wilcoxon(kept, raw, zero_method="wilcox", alternative=alternative)
            out[kr] = p
        except Exception:
            out[kr] = np.nan
    return out


###plot
# set Arial font globally
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 16

def plot_bars_with_kept_ratio_line(df, out_prefix="mondrian_bar"):

    g = df.groupby("kept_ratio_target", as_index=False).agg(
        kept_ratio_emp_mean=("kept_ratio_emp", "mean"),
        kept_ratio_emp_sem=("kept_ratio_emp", _safe_sem),

        sp_keep_mean=("spearman_kept", "mean"),
        sp_keep_sem=("spearman_kept", _safe_sem),
        sp_raw_mean=("spearman_raw", "mean"),
        sp_raw_sem=("spearman_raw", _safe_sem),

        pe_keep_mean=("pearson_kept", "mean"),
        pe_keep_sem=("pearson_kept", _safe_sem),
        pe_raw_mean=("pearson_raw", "mean"),
        pe_raw_sem=("pearson_raw", _safe_sem),

        mse_keep_mean=("mse_kept", "mean"),
        mse_keep_sem=("mse_kept", _safe_sem),
        mse_raw_mean=("mse_raw", "mean"),
        mse_raw_sem=("mse_raw", _safe_sem),
    )

    labels = [f"{int(r*100)}" for r in g["kept_ratio_target"]]
    x = np.arange(len(labels))
    width = 0.38

    # p-values
    p_sp  = _wilcoxon_by_kept_ratio(df, "spearman_kept", "spearman_raw", alternative="two-sided")
    p_pe  = _wilcoxon_by_kept_ratio(df, "pearson_kept",  "pearson_raw",  alternative="two-sided")
    p_mse = _wilcoxon_by_kept_ratio(df, "mse_kept", "mse_raw", alternative="less")

    # colors
    COLOR_KEPT_FILL = "#C7DFFD"
    COLOR_RAW_FILL  = "#BFE6DF"
    COLOR_KEPT_EDGE = "#377483"
    COLOR_RAW_EDGE  = "#4F845C"

    def _add_star(ax, xi, y_top, p, y_offset):
        star = _p_to_star(p)
        if star is None:
            return
        ax.text(
            xi, y_top + y_offset, star,
            ha="center", va="bottom",
            fontsize=12, fontweight="bold"
        )

    def _one(metric):
        if metric == "sp":
            kept_m, kept_s = g["sp_keep_mean"], g["sp_keep_sem"]
            raw_m,  raw_s  = g["sp_raw_mean"],  g["sp_raw_sem"]
            ylabel, title, y_min, y_max, p_map = "Spearman correlation", "AML BM samples", 0.50, 0.65, p_sp

        elif metric == "pe":
            kept_m, kept_s = g["pe_keep_mean"], g["pe_keep_sem"]
            raw_m,  raw_s  = g["pe_raw_mean"],  g["pe_raw_sem"]
            ylabel, title, y_min, y_max, p_map = "Pearson correlation", "AML BM samples", 0.45, 0.55, p_pe

        else:
            kept_m, kept_s = g["mse_keep_mean"], g["mse_keep_sem"]
            raw_m,  raw_s  = g["mse_raw_mean"],  g["mse_raw_sem"]
            ylabel, title, y_min, y_max, p_map = "MSE", "AML BM samples", 350, 600, p_mse

        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.set_ylim(y_min, y_max)

        # clean background
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

        # bars
        ax.bar(
            x - width/2, kept_m, width,
            yerr=kept_s,
            facecolor=COLOR_KEPT_FILL,
            edgecolor=COLOR_KEPT_EDGE,
            linewidth=1.2,
            label=f"{ylabel} (kept)",
            error_kw=dict(ecolor=COLOR_KEPT_EDGE, elinewidth=0.8, capsize=3)
        )

        ax.bar(
            x + width/2, raw_m, width,
            yerr=raw_s,
            facecolor=COLOR_RAW_FILL,
            edgecolor=COLOR_RAW_EDGE,
            linewidth=1.2,
            label=f"{ylabel} (raw)",
            error_kw=dict(ecolor=COLOR_RAW_EDGE, elinewidth=0.8, capsize=3)
        )

        # labels
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Target kept ratio (%)")

        # significance stars
        yr = ax.get_ylim()[1] - ax.get_ylim()[0]
        y_offset = max(yr * 0.03, 0.01)
        for i, kr in enumerate(g["kept_ratio_target"]):
            y_top = max(
                kept_m.iloc[i] + (kept_s.iloc[i] if not np.isnan(kept_s.iloc[i]) else 0),
                raw_m.iloc[i]  + (raw_s.iloc[i]  if not np.isnan(raw_s.iloc[i])  else 0)
            )
            _add_star(ax, x[i], y_top, p_map.get(kr, np.nan), y_offset)

        # legend
        #ax.legend(
        #    loc="upper center",
        #    bbox_to_anchor=(0.5, -0.15),
        #    ncol=2,
        #    frameon=False
        #)

        fig.tight_layout()
        plt.savefig(f"{out_prefix}_{metric}_BM.pdf", bbox_inches="tight")
        plt.savefig(f"{out_prefix}_{metric}_BM.png", dpi=300, bbox_inches="tight")
        plt.show()

    _one("sp")
    _one("pe")
    _one("mse")

    return g

# ============================================================
# USAGE (run this after you have: test_data, scaler, DANN)
# ============================================================
df_res = run_selective_experiment(
     test_data=test_data,
     scaler=scaler,
     DANN=DANN,
     n_reps=200,
     alpha_percentiles=range(50, 101, 10),  # maps to kept_ratio targets 0.5..1.0
     use_mondrian=True,        # True = Mondrian per-bin thresholds
     mondrian_bins=5,          # bins by y_pred quantiles
     random_seed=0
 )
#
# Example usage:
summary = plot_bars_with_kept_ratio_line(df_res, out_prefix="mondrian_selective")
print(summary)
#


# In[56]:


test_data=input__patient_PB.drop(labels = ['drug_row',"block_id","drug_col"], axis=1)


# In[59]:


import numpy as np
import pandas as pd
import random

from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt


# ============================================================
# Aim: improve selective/Mondrian prediction performance by


def build_error_features(X_scaled, y_pred):
    """
    Augment original features with the model's own prediction signals.
    This often improves uncertainty ranking.
    """
    y_pred = np.asarray(y_pred).reshape(-1, 1)
    feats = np.concatenate(
        [
            X_scaled,
            y_pred,
            np.abs(y_pred),
            (y_pred ** 2),
        ],
        axis=1
    )
    return feats


import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

def fit_error_model_crossfit_rf(
    Xerr_cal,
    err_target_cal,
    n_splits=3,
    random_state=0,
    return_oof=False
):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # out-of-fold predictions (diagnostic)
    oof_pred = np.full(len(err_target_cal), np.nan, dtype=float)

    # 
    rf_params = dict(
        n_estimators=300,        # 
        max_depth=6,             # 
        min_samples_leaf=30,     # 
        min_samples_split=10,
        max_features="sqrt",
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=random_state
    )

    # cross-fit OOF
    for tr_idx, va_idx in kf.split(Xerr_cal):
        X_tr, X_va = Xerr_cal[tr_idx], Xerr_cal[va_idx]
        y_tr = err_target_cal[tr_idx]

        fold_model = RandomForestRegressor(**rf_params)
        fold_model.fit(X_tr, y_tr)
        oof_pred[va_idx] = fold_model.predict(X_va)

    # final model on all calibration data
    final_model = RandomForestRegressor(**rf_params)
    final_model.fit(Xerr_cal, err_target_cal)

    if return_oof:
        return final_model, oof_pred
    return final_model


def selective_threshold_global(err_pred_test, kept_ratio):
    """
    Choose a threshold so that approx kept_ratio of samples are kept.
    Lower predicted error = keep.
    """
    kept_ratio = float(kept_ratio)
    kept_ratio = min(max(kept_ratio, 0.0), 1.0)
    thr = np.quantile(err_pred_test, kept_ratio)
    keep = err_pred_test <= thr
    return keep, thr


def mondrian_threshold_by_bins(err_pred_test, bin_id_test, kept_ratio):
    """
    Mondrian selective rule: apply the SAME kept_ratio within each bin.
    Returns boolean keep mask (same length as err_pred_test).
    """
    keep = np.zeros(len(err_pred_test), dtype=bool)
    for b in np.unique(bin_id_test):
        idx = np.where(bin_id_test == b)[0]
        if len(idx) == 0:
            continue
        thr = np.quantile(err_pred_test[idx], kept_ratio)
        keep[idx] = err_pred_test[idx] <= thr
    return keep


def make_bins_by_pred(y_pred, n_bins=5):
    """
    Mondrian bins based on prediction quantiles.
    (You can change to bins by SampleID group, drug class, etc.)
    """
    y_pred = np.asarray(y_pred).reshape(-1)
    qs = np.quantile(y_pred, np.linspace(0, 1, n_bins + 1))
    # ensure strictly increasing edges (handle ties)
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    bin_id = np.digitize(y_pred, qs[1:-1], right=True)  # 0..n_bins-1
    return bin_id


# ----------------------------
# Main experiment runner
# ----------------------------
def run_selective_experiment(
    test_data,
    scaler,
    DANN,
    n_reps=200,
    alpha_percentiles=range(50, 101, 10),  # we'll map these to target kept_ratios if you want
    use_mondrian=True,
    mondrian_bins=5,
    random_seed=0
):
    random.seed(random_seed)
    np.random.seed(random_seed)

    # If you want kept_ratio targets corresponding to percentiles, use:
    # alpha_percentile=50 -> kept_ratio roughly 0.50, etc.
    kept_ratio_targets = [p / 100.0 for p in alpha_percentiles]

    # Store per-point results (NOT averaged raw; we average later by kept_ratio)
    rows = []

    for rep in range(n_reps):
        grouped = list(test_data.groupby("SampleID"))
        random.shuffle(grouped)

        calib_groups, test_groups = [], []
        for k, g in enumerate(grouped):
            (calib_groups if k % 2 == 0 else test_groups).append(g)

        def groups_to_df(groups):
            return pd.concat([df for _, df in groups], ignore_index=True)

        calibration_set = groups_to_df(calib_groups)
        new_test_set    = groups_to_df(test_groups)

        # Scale
        X_test = scaler.transform(new_test_set.drop(columns=["SampleID", "synergy_hsa"]).values)
        y_test = new_test_set["synergy_hsa"].values

        X_cal  = scaler.transform(calibration_set.drop(columns=["SampleID", "synergy_hsa"]).values)
        y_cal  = calibration_set["synergy_hsa"].values

        # DANN predictions
        y_cal_pred  = np.squeeze(DANN.predict(X_cal,  verbose=0)[0])
        y_test_pred = np.squeeze(DANN.predict(X_test, verbose=0)[0])

        # ---- uncertainty target: log-error (recommended) ----
        cal_abs_err = np.abs(y_cal - y_cal_pred)
        cal_err_target = np.log(cal_abs_err + 1e-3)  # log-error

        # ---- uncertainty features: X + y_pred signals ----
        Xerr_cal  = build_error_features(X_cal,  y_cal_pred)
        Xerr_test = build_error_features(X_test, y_test_pred)

        # fit uncertainty model
        err_model =fit_error_model_crossfit_rf(Xerr_cal, cal_err_target, random_state=random_seed+rep)

        # predict uncertainty on test
        err_pred_test = err_model.predict(Xerr_test)

        # raw metrics (full test set)
        sp_raw, _ = spearmanr(y_test, y_test_pred)
        pe_raw, _ = pearsonr(y_test, y_test_pred)
        mse_raw   = mean_squared_error(y_test, y_test_pred)

        # Mondrian bins (optional)
        if use_mondrian:
            bin_id_test = make_bins_by_pred(y_test_pred, n_bins=mondrian_bins)

        for kept_ratio in kept_ratio_targets:
            if use_mondrian:
                keep = mondrian_threshold_by_bins(err_pred_test, bin_id_test, kept_ratio)
            else:
                keep, _ = selective_threshold_global(err_pred_test, kept_ratio)

            X_keep = X_test[keep]
            y_keep = y_test[keep]

            kept_ratio_emp = len(y_keep) / max(1, len(y_test))

            if len(y_keep) < 3:
                # too few points for correlations
                rows.append({
                    "rep": rep + 1,
                    "kept_ratio_target": kept_ratio,
                    "kept_ratio_emp": kept_ratio_emp,
                    "spearman_kept": np.nan,
                    "pearson_kept": np.nan,
                    "mse_kept": np.nan,
                    "spearman_raw": sp_raw,
                    "pearson_raw": pe_raw,
                    "mse_raw": mse_raw,
                })
                continue

            y_keep_pred = np.squeeze(DANN.predict(X_keep, verbose=0)[0])

            sp_keep, _ = spearmanr(y_keep, y_keep_pred)
            pe_keep, _ = pearsonr(y_keep, y_keep_pred)
            mse_keep   = mean_squared_error(y_keep, y_keep_pred)

            rows.append({
                "rep": rep + 1,
                "kept_ratio_target": kept_ratio,
                "kept_ratio_emp": kept_ratio_emp,
                "spearman_kept": sp_keep,
                "pearson_kept": pe_keep,
                "mse_kept": mse_keep,
                "spearman_raw": sp_raw,
                "pearson_raw": pe_raw,
                "mse_raw": mse_raw,
            })

    return pd.DataFrame(rows)


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# -------------------------
# utils
# -------------------------
def _safe_sem(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size <= 1:
        return np.nan
    return x.std(ddof=1) / np.sqrt(x.size)

def _p_to_star(p):
    if np.isnan(p):
        return None
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return None

def _wilcoxon_by_kept_ratio(df, col_kept, col_raw, alternative="two-sided"):
    """
    Return dict: kept_ratio_target -> p-value
    alternative:
      - "two-sided"
      - "greater"  (kept > raw)  : good for correlations
      - "less"     (kept < raw)  : good for MSE (smaller is better)
    """
    out = {}
    for kr, g in df.groupby("kept_ratio_target"):
        kept = g[col_kept].to_numpy(float)
        raw  = g[col_raw].to_numpy(float)
        mask = ~np.isnan(kept) & ~np.isnan(raw)
        kept, raw = kept[mask], raw[mask]

        if len(kept) < 5:
            out[kr] = np.nan
            continue
        if np.allclose(kept - raw, 0):
            out[kr] = 1.0
            continue

        try:
            _, p = wilcoxon(kept, raw, zero_method="wilcox", alternative=alternative)
            out[kr] = p
        except Exception:
            out[kr] = np.nan
    return out


# -------------------------
# main plotting function
# set Arial font globally
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 16

def plot_bars_with_kept_ratio_line(df, out_prefix="mondrian_bar"):

    g = df.groupby("kept_ratio_target", as_index=False).agg(
        kept_ratio_emp_mean=("kept_ratio_emp", "mean"),
        kept_ratio_emp_sem=("kept_ratio_emp", _safe_sem),

        sp_keep_mean=("spearman_kept", "mean"),
        sp_keep_sem=("spearman_kept", _safe_sem),
        sp_raw_mean=("spearman_raw", "mean"),
        sp_raw_sem=("spearman_raw", _safe_sem),

        pe_keep_mean=("pearson_kept", "mean"),
        pe_keep_sem=("pearson_kept", _safe_sem),
        pe_raw_mean=("pearson_raw", "mean"),
        pe_raw_sem=("pearson_raw", _safe_sem),

        mse_keep_mean=("mse_kept", "mean"),
        mse_keep_sem=("mse_kept", _safe_sem),
        mse_raw_mean=("mse_raw", "mean"),
        mse_raw_sem=("mse_raw", _safe_sem),
    )

    labels = [f"{int(r*100)}" for r in g["kept_ratio_target"]]
    x = np.arange(len(labels))
    width = 0.38

    # p-values
    p_sp  = _wilcoxon_by_kept_ratio(df, "spearman_kept", "spearman_raw", alternative="two-sided")
    p_pe  = _wilcoxon_by_kept_ratio(df, "pearson_kept",  "pearson_raw",  alternative="two-sided")
    p_mse = _wilcoxon_by_kept_ratio(df, "mse_kept", "mse_raw", alternative="less")

    # colors
    COLOR_KEPT_FILL = "#C7DFFD"
    COLOR_RAW_FILL  = "#BFE6DF"
    COLOR_KEPT_EDGE = "#377483"
    COLOR_RAW_EDGE  = "#4F845C"

    def _add_star(ax, xi, y_top, p, y_offset):
        star = _p_to_star(p)
        if star is None:
            return
        ax.text(
            xi, y_top + y_offset, star,
            ha="center", va="bottom",
            fontsize=12, fontweight="bold"
        )

    def _one(metric):
        if metric == "sp":
            kept_m, kept_s = g["sp_keep_mean"], g["sp_keep_sem"]
            raw_m,  raw_s  = g["sp_raw_mean"],  g["sp_raw_sem"]
            ylabel, title, y_min, y_max, p_map = "Spearman correlation", "AML PB samples", 0.45, 0.55, p_sp

        elif metric == "pe":
            kept_m, kept_s = g["pe_keep_mean"], g["pe_keep_sem"]
            raw_m,  raw_s  = g["pe_raw_mean"],  g["pe_raw_sem"]
            ylabel, title, y_min, y_max, p_map = "Pearson correlation", "AML PB samples", 0.45, 0.60, p_pe

        else:
            kept_m, kept_s = g["mse_keep_mean"], g["mse_keep_sem"]
            raw_m,  raw_s  = g["mse_raw_mean"],  g["mse_raw_sem"]
            ylabel, title, y_min, y_max, p_map = "MSE", "AML PB samples", 450, 700, p_mse

        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.set_ylim(y_min, y_max)

        # clean background
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

        # bars
        ax.bar(
            x - width/2, kept_m, width,
            yerr=kept_s,
            facecolor=COLOR_KEPT_FILL,
            edgecolor=COLOR_KEPT_EDGE,
            linewidth=1.2,
            label=f"{ylabel} (kept)",
            error_kw=dict(ecolor=COLOR_KEPT_EDGE, elinewidth=0.8, capsize=3)
        )

        ax.bar(
            x + width/2, raw_m, width,
            yerr=raw_s,
            facecolor=COLOR_RAW_FILL,
            edgecolor=COLOR_RAW_EDGE,
            linewidth=1.2,
            label=f"{ylabel} (raw)",
            error_kw=dict(ecolor=COLOR_RAW_EDGE, elinewidth=0.8, capsize=3)
        )

        # labels
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Target kept ratio (%)")

        # significance stars
        yr = ax.get_ylim()[1] - ax.get_ylim()[0]
        y_offset = max(yr * 0.03, 0.01)
        for i, kr in enumerate(g["kept_ratio_target"]):
            y_top = max(
                kept_m.iloc[i] + (kept_s.iloc[i] if not np.isnan(kept_s.iloc[i]) else 0),
                raw_m.iloc[i]  + (raw_s.iloc[i]  if not np.isnan(raw_s.iloc[i])  else 0)
            )
            _add_star(ax, x[i], y_top, p_map.get(kr, np.nan), y_offset)

        # legend
        #ax.legend(
        #    loc="upper center",
        #    bbox_to_anchor=(0.5, -0.15),
        #    ncol=2,
        #    frameon=False
        #)

        fig.tight_layout()
        plt.savefig(f"{out_prefix}_{metric}_PB.pdf", bbox_inches="tight")
        plt.savefig(f"{out_prefix}_{metric}_PB.png", dpi=300, bbox_inches="tight")
        plt.show()

    _one("sp")
    _one("pe")
    _one("mse")

    return g
# ============================================================
# USAGE (run this after you have: test_data, scaler, DANN)
# ============================================================
df_res = run_selective_experiment(
     test_data=test_data,
     scaler=scaler,
     DANN=DANN,
     n_reps=200,
     alpha_percentiles=range(50, 101, 10),  # maps to kept_ratio targets 0.5..1.0
     use_mondrian=True,        # True = Mondrian per-bin thresholds
     mondrian_bins=5,          # bins by y_pred quantiles
     random_seed=0
 )
#
# Example usage:
summary = plot_bars_with_kept_ratio_line(df_res, out_prefix="mondrian_selective")
print(summary)
#


# ###CLL dataset

# In[51]:


test_data = pd.read_csv('CLL_test_remaining_samples.csv')


# In[52]:


# =========================
from tensorflow.keras.models import load_model

import tensorflow as tf

class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return self.grad_reverse(inputs)

    @tf.custom_gradient
    def grad_reverse(self, x):
        y = tf.identity(x)

        def custom_grad(dy):
            return -dy

        return y, custom_grad


DANN = load_model(
    "DANN_50_samples_CLL.keras",
    custom_objects={"GradientReversalLayer": GradientReversalLayer}
)


# In[53]:


import numpy as np
import pandas as pd
import random

from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt


# ============================================================
# Aim: improve selective/Mondrian prediction performance 

def build_error_features(X_scaled, y_pred):
    """
    Augment original features with the model's own prediction signals.
    This often improves uncertainty ranking.
    """
    y_pred = np.asarray(y_pred).reshape(-1, 1)
    feats = np.concatenate(
        [
            X_scaled,
            y_pred,
            np.abs(y_pred),
            (y_pred ** 2),
        ],
        axis=1
    )
    return feats


import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

def fit_error_model_crossfit_rf(
    Xerr_cal,
    err_target_cal,
    n_splits=3,
    random_state=0,
    return_oof=False
):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # out-of-fold predictions (diagnostic)
    oof_pred = np.full(len(err_target_cal), np.nan, dtype=float)

    # 
    rf_params = dict(
        n_estimators=300,        # 
        max_depth=6,             # 
        min_samples_leaf=30,     # 
        min_samples_split=10,
        max_features="sqrt",
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=random_state
    )

    # cross-fit OOF
    for tr_idx, va_idx in kf.split(Xerr_cal):
        X_tr, X_va = Xerr_cal[tr_idx], Xerr_cal[va_idx]
        y_tr = err_target_cal[tr_idx]

        fold_model = RandomForestRegressor(**rf_params)
        fold_model.fit(X_tr, y_tr)
        oof_pred[va_idx] = fold_model.predict(X_va)

    # final model on all calibration data
    final_model = RandomForestRegressor(**rf_params)
    final_model.fit(Xerr_cal, err_target_cal)

    if return_oof:
        return final_model, oof_pred
    return final_model


def selective_threshold_global(err_pred_test, kept_ratio):
    """
    Choose a threshold so that approx kept_ratio of samples are kept.
    Lower predicted error = keep.
    """
    kept_ratio = float(kept_ratio)
    kept_ratio = min(max(kept_ratio, 0.0), 1.0)
    thr = np.quantile(err_pred_test, kept_ratio)
    keep = err_pred_test <= thr
    return keep, thr


def mondrian_threshold_by_bins(err_pred_test, bin_id_test, kept_ratio):
    """
    Mondrian selective rule: apply the SAME kept_ratio within each bin.
    Returns boolean keep mask (same length as err_pred_test).
    """
    keep = np.zeros(len(err_pred_test), dtype=bool)
    for b in np.unique(bin_id_test):
        idx = np.where(bin_id_test == b)[0]
        if len(idx) == 0:
            continue
        thr = np.quantile(err_pred_test[idx], kept_ratio)
        keep[idx] = err_pred_test[idx] <= thr
    return keep


def make_bins_by_pred(y_pred, n_bins=5):
    """
    Mondrian bins based on prediction quantiles.
    (You can change to bins by SampleID group, drug class, etc.)
    """
    y_pred = np.asarray(y_pred).reshape(-1)
    qs = np.quantile(y_pred, np.linspace(0, 1, n_bins + 1))
    # ensure strictly increasing edges (handle ties)
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    bin_id = np.digitize(y_pred, qs[1:-1], right=True)  # 0..n_bins-1
    return bin_id


# ----------------------------
# Main experiment runner
# ----------------------------
def run_selective_experiment(
    test_data,
    scaler,
    DANN,
    n_reps=200,
    alpha_percentiles=range(50, 101, 10),  # we'll map these to target kept_ratios if you want
    use_mondrian=True,
    mondrian_bins=5,
    random_seed=0
):
    random.seed(random_seed)
    np.random.seed(random_seed)

    # If you want kept_ratio targets corresponding to percentiles, use:
    # alpha_percentile=50 -> kept_ratio roughly 0.50, etc.
    kept_ratio_targets = [p / 100.0 for p in alpha_percentiles]

    # Store per-point results (NOT averaged raw; we average later by kept_ratio)
    rows = []

    for rep in range(n_reps):
        grouped = list(test_data.groupby("SampleID"))
        random.shuffle(grouped)

        calib_groups, test_groups = [], []
        for k, g in enumerate(grouped):
            (calib_groups if k % 2 == 0 else test_groups).append(g)

        def groups_to_df(groups):
            return pd.concat([df for _, df in groups], ignore_index=True)

        calibration_set = groups_to_df(calib_groups)
        new_test_set    = groups_to_df(test_groups)

        # Scale
        X_test = scaler.transform(new_test_set.drop(columns=["SampleID", "HSA_score"]).values)
        y_test = new_test_set["HSA_score"].values

        X_cal  = scaler.transform(calibration_set.drop(columns=["SampleID", "HSA_score"]).values)
        y_cal  = calibration_set["HSA_score"].values

        # DANN predictions
        y_cal_pred  = np.squeeze(DANN.predict(X_cal,  verbose=0)[0])
        y_test_pred = np.squeeze(DANN.predict(X_test, verbose=0)[0])

        # ---- uncertainty target: log-error (recommended) ----
        cal_abs_err = np.abs(y_cal - y_cal_pred)
        cal_err_target = np.log(cal_abs_err + 1e-3)  # log-error

        # ---- uncertainty features: X + y_pred signals ----
        Xerr_cal  = build_error_features(X_cal,  y_cal_pred)
        Xerr_test = build_error_features(X_test, y_test_pred)

        # fit uncertainty model
        err_model =fit_error_model_crossfit_rf(Xerr_cal, cal_err_target, random_state=random_seed+rep)

        # predict uncertainty on test
        err_pred_test = err_model.predict(Xerr_test)

        # raw metrics (full test set)
        sp_raw, _ = spearmanr(y_test, y_test_pred)
        pe_raw, _ = pearsonr(y_test, y_test_pred)
        mse_raw   = mean_squared_error(y_test, y_test_pred)

        # Mondrian bins (optional)
        if use_mondrian:
            bin_id_test = make_bins_by_pred(y_test_pred, n_bins=mondrian_bins)

        for kept_ratio in kept_ratio_targets:
            if use_mondrian:
                keep = mondrian_threshold_by_bins(err_pred_test, bin_id_test, kept_ratio)
            else:
                keep, _ = selective_threshold_global(err_pred_test, kept_ratio)

            X_keep = X_test[keep]
            y_keep = y_test[keep]

            kept_ratio_emp = len(y_keep) / max(1, len(y_test))

            if len(y_keep) < 3:
                # too few points for correlations
                rows.append({
                    "rep": rep + 1,
                    "kept_ratio_target": kept_ratio,
                    "kept_ratio_emp": kept_ratio_emp,
                    "spearman_kept": np.nan,
                    "pearson_kept": np.nan,
                    "mse_kept": np.nan,
                    "spearman_raw": sp_raw,
                    "pearson_raw": pe_raw,
                    "mse_raw": mse_raw,
                })
                continue

            y_keep_pred = np.squeeze(DANN.predict(X_keep, verbose=0)[0])

            sp_keep, _ = spearmanr(y_keep, y_keep_pred)
            pe_keep, _ = pearsonr(y_keep, y_keep_pred)
            mse_keep   = mean_squared_error(y_keep, y_keep_pred)

            rows.append({
                "rep": rep + 1,
                "kept_ratio_target": kept_ratio,
                "kept_ratio_emp": kept_ratio_emp,
                "spearman_kept": sp_keep,
                "pearson_kept": pe_keep,
                "mse_kept": mse_keep,
                "spearman_raw": sp_raw,
                "pearson_raw": pe_raw,
                "mse_raw": mse_raw,
            })

    return pd.DataFrame(rows)


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# -------------------------
# utils
# -------------------------
def _safe_sem(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size <= 1:
        return np.nan
    return x.std(ddof=1) / np.sqrt(x.size)

def _p_to_star(p):
    if np.isnan(p):
        return None
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return None

def _wilcoxon_by_kept_ratio(df, col_kept, col_raw, alternative="two-sided"):
    """
    Return dict: kept_ratio_target -> p-value
    alternative:
      - "two-sided"
      - "greater"  (kept > raw)  : good for correlations
      - "less"     (kept < raw)  : good for MSE (smaller is better)
    """
    out = {}
    for kr, g in df.groupby("kept_ratio_target"):
        kept = g[col_kept].to_numpy(float)
        raw  = g[col_raw].to_numpy(float)
        mask = ~np.isnan(kept) & ~np.isnan(raw)
        kept, raw = kept[mask], raw[mask]

        if len(kept) < 5:
            out[kr] = np.nan
            continue
        if np.allclose(kept - raw, 0):
            out[kr] = 1.0
            continue

        try:
            _, p = wilcoxon(kept, raw, zero_method="wilcox", alternative=alternative)
            out[kr] = p
        except Exception:
            out[kr] = np.nan
    return out


###plot
# set Arial font globally
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 16

def plot_bars_with_kept_ratio_line(df, out_prefix="mondrian_bar"):

    g = df.groupby("kept_ratio_target", as_index=False).agg(
        kept_ratio_emp_mean=("kept_ratio_emp", "mean"),
        kept_ratio_emp_sem=("kept_ratio_emp", _safe_sem),

        sp_keep_mean=("spearman_kept", "mean"),
        sp_keep_sem=("spearman_kept", _safe_sem),
        sp_raw_mean=("spearman_raw", "mean"),
        sp_raw_sem=("spearman_raw", _safe_sem),

        pe_keep_mean=("pearson_kept", "mean"),
        pe_keep_sem=("pearson_kept", _safe_sem),
        pe_raw_mean=("pearson_raw", "mean"),
        pe_raw_sem=("pearson_raw", _safe_sem),

        mse_keep_mean=("mse_kept", "mean"),
        mse_keep_sem=("mse_kept", _safe_sem),
        mse_raw_mean=("mse_raw", "mean"),
        mse_raw_sem=("mse_raw", _safe_sem),
    )

    labels = [f"{int(r*100)}" for r in g["kept_ratio_target"]]
    x = np.arange(len(labels))
    width = 0.38

    # p-values
    p_sp  = _wilcoxon_by_kept_ratio(df, "spearman_kept", "spearman_raw", alternative="two-sided")
    p_pe  = _wilcoxon_by_kept_ratio(df, "pearson_kept",  "pearson_raw",  alternative="two-sided")
    p_mse = _wilcoxon_by_kept_ratio(df, "mse_kept", "mse_raw", alternative="less")

    # colors
    COLOR_KEPT_FILL = "#C7DFFD"
    COLOR_RAW_FILL  = "#BFE6DF"
    COLOR_KEPT_EDGE = "#377483"
    COLOR_RAW_EDGE  = "#4F845C"

    def _add_star(ax, xi, y_top, p, y_offset):
        star = _p_to_star(p)
        if star is None:
            return
        ax.text(
            xi, y_top + y_offset, star,
            ha="center", va="bottom",
            fontsize=12, fontweight="bold"
        )

    def _one(metric):
        if metric == "sp":
            kept_m, kept_s = g["sp_keep_mean"], g["sp_keep_sem"]
            raw_m,  raw_s  = g["sp_raw_mean"],  g["sp_raw_sem"]
            ylabel, title, y_min, y_max, p_map = "Spearman correlation", "CLL PB samples", 0.50, 0.70, p_sp

        elif metric == "pe":
            kept_m, kept_s = g["pe_keep_mean"], g["pe_keep_sem"]
            raw_m,  raw_s  = g["pe_raw_mean"],  g["pe_raw_sem"]
            ylabel, title, y_min, y_max, p_map = "Pearson correlation", "CLL PB samples", 0.50, 0.70, p_pe

        else:
            kept_m, kept_s = g["mse_keep_mean"], g["mse_keep_sem"]
            raw_m,  raw_s  = g["mse_raw_mean"],  g["mse_raw_sem"]
            ylabel, title, y_min, y_max, p_map = "MSE", "CLL PB samples", 0, 20, p_mse

        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.set_ylim(y_min, y_max)

        # clean background
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

        # bars
        ax.bar(
            x - width/2, kept_m, width,
            yerr=kept_s,
            facecolor=COLOR_KEPT_FILL,
            edgecolor=COLOR_KEPT_EDGE,
            linewidth=1.2,
            label=f"{ylabel} (kept)",
            error_kw=dict(ecolor=COLOR_KEPT_EDGE, elinewidth=0.8, capsize=3)
        )

        ax.bar(
            x + width/2, raw_m, width,
            yerr=raw_s,
            facecolor=COLOR_RAW_FILL,
            edgecolor=COLOR_RAW_EDGE,
            linewidth=1.2,
            label=f"{ylabel} (raw)",
            error_kw=dict(ecolor=COLOR_RAW_EDGE, elinewidth=0.8, capsize=3)
        )

        # labels
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Target kept ratio (%)")

        # significance stars
        yr = ax.get_ylim()[1] - ax.get_ylim()[0]
        y_offset = max(yr * 0.03, 0.01)
        for i, kr in enumerate(g["kept_ratio_target"]):
            y_top = max(
                kept_m.iloc[i] + (kept_s.iloc[i] if not np.isnan(kept_s.iloc[i]) else 0),
                raw_m.iloc[i]  + (raw_s.iloc[i]  if not np.isnan(raw_s.iloc[i])  else 0)
            )
            _add_star(ax, x[i], y_top, p_map.get(kr, np.nan), y_offset)

        # legend
        #ax.legend(
        #    loc="upper center",
        #    bbox_to_anchor=(0.5, -0.15),
        #    ncol=2,
        #    frameon=False
        #)

        fig.tight_layout()
        plt.savefig(f"{out_prefix}_{metric}_CLL.pdf", bbox_inches="tight")
        plt.savefig(f"{out_prefix}_{metric}_CLL.png", dpi=300, bbox_inches="tight")
        plt.show()

    _one("sp")
    _one("pe")
    _one("mse")

    return g

# ============================================================
# USAGE (run this after you have: test_data, scaler, DANN)
# ============================================================
df_res = run_selective_experiment(
     test_data=test_data,
     scaler=scaler,
     DANN=DANN,
     n_reps=200,
     alpha_percentiles=range(50, 101, 10),  # maps to kept_ratio targets 0.5..1.0
     use_mondrian=True,        # True = Mondrian per-bin thresholds
     mondrian_bins=5,          # bins by y_pred quantiles
     random_seed=0
 )
#
# Example usage:
summary = plot_bars_with_kept_ratio_line(df_res, out_prefix="mondrian_selective")
print(summary)
#


# In[ ]:




