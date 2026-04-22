#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import skopt
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
import h5py
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from xgboost.sklearn import XGBRegressor
from sklearn import linear_model
from sklearn.linear_model import RidgeCV, LassoCV
from scipy.stats import pearsonr
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from concrete_autoencoder import ConcreteAutoencoderFeatureSelector
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, LeakyReLU, Softmax
from tensorflow.keras.models import load_model

# Load the saved patient model
cell_line_model = load_model('cell_line_model_based on blood_less neuron.h5')

# Load training data
with h5py.File('training_data_blood.h5', 'r') as f:
    X_train = f['X_train'][:]
    y_train = f['y_train'][:]

# Load test data
with h5py.File('test_data_blood.h5', 'r') as f:
    X_test = f['X_test'][:]
    y_test = f['y_test'][:]

# Create scaler object
scaler = StandardScaler()

# Fit scaler to training data and transform it
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#PB data
rowDrug_MACCS_patient_PB = pd.read_csv("rowDrug_MACCS_input_PB.csv", sep = ",")
rowDrug_Sig_patient_PB = pd.read_csv("rowDrug_Sig_input_PB.csv", sep = ",")
colDrug_MACCS_patient_PB = pd.read_csv("colDrug_MACCS_input_PB.csv", sep = ",")
colDrug_Sig_patient_PB = pd.read_csv("colDrug_Sig_input_PB.csv", sep = ",")
CellExp_patient_PB = pd.read_csv("cellExp_ssGSEA_input_PB_1221.csv", sep = ",")
HSA_label_patient_PB = pd.read_csv("HSA_label_PB.csv", sep = ",")
input__patient_PB=pd.concat([rowDrug_MACCS_patient_PB,rowDrug_Sig_patient_PB,colDrug_MACCS_patient_PB,colDrug_Sig_patient_PB,CellExp_patient_PB,HSA_label_patient_PB],axis=1,join="inner") 
input_patient_PB_final=input__patient_PB.drop(labels = ['drug_row',"block_id","drug_col","SampleID"], axis=1)
new_data_patient_PB=input_patient_PB_final
X_patient_PB = new_data_patient_PB.drop("synergy_hsa", axis = 1).values
y_patient_PB = new_data_patient_PB.synergy_hsa.astype(float)


#BM
rowDrug_MACCS_patient_BM = pd.read_csv("rowDrug_MACCS_input_BM.csv", sep = ",")
rowDrug_Sig_patient_BM = pd.read_csv("rowDrug_Sig_input_BM.csv", sep = ",")
colDrug_MACCS_patient_BM = pd.read_csv("colDrug_MACCS_input_BM.csv", sep = ",")
colDrug_Sig_patient_BM = pd.read_csv("colDrug_Sig_input_BM.csv", sep = ",")
CellExp_patient_BM = pd.read_csv("cellExp_ssGSEA_input_BM_1221.csv", sep = ",")
HSA_label_patient_BM = pd.read_csv("HSA_label_BM.csv", sep = ",")
input__patient_BM=pd.concat([rowDrug_MACCS_patient_BM,rowDrug_Sig_patient_BM,colDrug_MACCS_patient_BM,colDrug_Sig_patient_BM,CellExp_patient_BM,HSA_label_patient_BM],axis=1,join="inner") 
input_patient_BM_final=input__patient_BM.drop(labels = ['drug_row',"block_id","drug_col"], axis=1)
new_data_patient_BM=input_patient_BM_final

bm_sample_ids = new_data_patient_BM["SampleID"].unique()
print("Total BM patients:", len(bm_sample_ids))
LABEL_COL = "synergy_hsa"
bm_data = new_data_patient_BM.copy()

# Fit scaler to training data and transform it
X_patient_PB_scaled = scaler.transform(X_patient_PB)
X_test_PB = X_patient_PB_scaled
y_test_PB = y_patient_PB
print (X_train_scaled.shape[1])

####DANN model
import random
# Gradient Reversal Layer
seeds=281200696
np.random.seed(seeds)
random.seed(seeds)
tf.random.set_seed(seeds)
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

# Domain classifier layers (more layers)
def domain_classifier(input_shape):
    model = Sequential([
        # 1st hidden layer
        Dense(256, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3, seed=seeds),

        # 2th hidden layer (new)
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3, seed=seeds),

        # output
        Dense(1, activation='sigmoid')
    ])
    return model


# Task-specific layers (for patient data, more layers)
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
def task_specific(input_shape):
    return Sequential([
        Dense(512, activation='relu', input_shape=input_shape,
              kernel_regularizer=regularizers.l2(1e-4)),
        BatchNormalization(),
        Dropout(0.2, seed=seeds),
        Dense(128, activation='relu',
              kernel_regularizer=regularizers.l2(1e-4)),
        BatchNormalization(),
        Dropout(0.2, seed=seeds),
        Dense(1, activation='linear')
    ])
# Remove the last layer of the cell line model to create a feature extractor
feature_extractor = Model(inputs=cell_line_model.input, outputs=cell_line_model.layers[-2].output)
feature_extractor.trainable = True


from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
# Transfer learning model
input_layer = Input(shape=(3565,))
features = feature_extractor(input_layer)
grl = GradientReversalLayer()(features)
domain_output = domain_classifier(features.shape[1:])(grl)
task_output = task_specific(features.shape[1:])(features)


def build_and_compile_model():
    input_layer = Input(shape=(3565,))
    features = feature_extractor(input_layer)
    grl = GradientReversalLayer()(features)
    domain_output = domain_classifier(features.shape[1:])(grl)
    task_output = task_specific(features.shape[1:])(features)

    model = Model(inputs=input_layer, outputs=[task_output, domain_output])
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss=['mean_squared_error', 'binary_crossentropy'],
        loss_weights=[1.0, 0.0]   #  domain loss_weight = 0
    )
    return model


# Create and compile the model
transfer_model = build_and_compile_model()

# Save the initial weights
initial_weights = transfer_model.get_weights()


import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

class DomainWeightScheduler(Callback):
    def __init__(self, max_lambda=0.1, n_epochs=300):
        super().__init__()
        self.max_lambda = max_lambda
        self.n_epochs = n_epochs

    def on_epoch_begin(self, epoch, logs=None):
        # p from 0-1
        p = epoch / float(self.n_epochs)
        # DANN ---lambda 
        lam = self.max_lambda * (2. / (1. + np.exp(-10 * p)) - 1)
        # modifying the loss_weights： [task_loss_weight, domain_loss_weight]
        self.model.loss_weights = [1.0, lam]


# -----------------------------
# Patient-only ablation model
# -----------------------------
def build_patient_only_model(input_dim=3565, lr=0.0005):
    """
    Patient-only neural network:
    - no cell-line pretraining
    - no transfer learning
    - no domain classifier
    - no gradient reversal
    - trained only on patient BM data
    """
    tf.keras.utils.set_random_seed(seeds)

    input_layer = Input(shape=(input_dim,), name="patient_only_input")

    x = Dense(
        512,
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-4)
    )(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.2, seed=seeds)(x)

    x = Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2, seed=seeds)(x)

    output = Dense(1, activation='linear')(x)

    model = Model(inputs=input_layer, outputs=output, name="patient_only_model")
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='mean_squared_error'
    )
    return model

patient_only_template = build_patient_only_model(input_dim=3565, lr=0.0005)
initial_weights_patient_only = patient_only_template.get_weights()


from scipy.stats import spearmanr

unique_combinations = {}

# keep three models
model_names = [
    "DANN_BM", "DANN_PB", "DANN_train",
    "patient_only_BM", "patient_only_PB", "patient_only_train"
]

# initialize result containers
r2_scores = {m: [] for m in model_names}
corr_coeffs = {m: [] for m in model_names}
mse_scores = {m: [] for m in model_names}
pearson_coeffs = {m: [] for m in model_names}

avg_r2_scores = {m: {} for m in model_names}
avg_corr_coeffs = {m: {} for m in model_names}
avg_mse_scores = {m: {} for m in model_names}
avg_pearson_coeffs = {m: {} for m in model_names}

all_r2_scores = {m: {} for m in model_names}
all_corr_coeffs = {m: {} for m in model_names}
all_mse_scores = {m: {} for m in model_names}
all_pearson_coeffs = {m: {} for m in model_names}

for n_samples in samples:
    print(f"\n========== Use {n_samples} BM patient(s) for training ==========")
    unique_combinations[n_samples] = set()

    for i in range(repeat):
        print("-" * 60)
        print(f"[INFO] n_samples = {n_samples}, iteration = {i+1}/{repeat}")

        # 1) select BM patient SampleIDs
        while True:
            if n_samples == 1:
                bm_ids_modified = [
                    sid for sid in bm_sample_ids
                    if sid not in ["16-00351", "16-00292", "16-00113", "15-00858", "15-00688"]
                ]
            else:
                bm_ids_modified = bm_sample_ids

            selected_sample_ids = tuple(
                sorted(np.random.choice(bm_ids_modified, n_samples, replace=False))
            )

            if selected_sample_ids not in unique_combinations[n_samples]:
                break

        print("[DEBUG] Selected BM SampleIDs:", selected_sample_ids)
        unique_combinations[n_samples].add(selected_sample_ids)

        iter_name = f"{n_samples}_samples_repeat_{i+1}"
        selected_samples["iteration"].append(iter_name)
        selected_samples["n_samples"].append(n_samples)
        selected_samples["samples"].append(selected_sample_ids)

        # 2) BM subset
        subset_train_BM = bm_data[bm_data["SampleID"].isin(selected_sample_ids)].copy()
        print("[DEBUG] BM train subset shape:", subset_train_BM.shape)

        X_patient_BM = subset_train_BM.drop(columns=["SampleID", LABEL_COL]).values.astype(np.float32)
        X_patient_BM_scaled = scaler.transform(X_patient_BM)
        y = subset_train_BM[LABEL_COL].values.astype(np.float32)

        # 3) BM test
        subset_test_BM = bm_data[~bm_data["SampleID"].isin(selected_sample_ids)].copy()
        print("[DEBUG] BM test subset shape:", subset_test_BM.shape)

        X_test_BM = subset_test_BM.drop(columns=["SampleID", LABEL_COL]).values.astype(np.float32)
        X_test_BM = scaler.transform(X_test_BM)
        y_test_BM = subset_test_BM[LABEL_COL].values.astype(np.float32)

        # 4) identify matched cell line data
        num_rows = subset_train_BM.shape[0]
        train_size = num_rows / 6829
        print(f"[DEBUG] num_rows = {num_rows}, train_size = {train_size:.4f}")

        X_train_use, X_test_no, y_train_use, y_test_no = train_test_split(
            X_train_scaled,
            y_train,
            train_size=train_size,
            random_state=seeds
        )
        print("[DEBUG] X_train_use shape:", X_train_use.shape)

        # 5) BM + cell line for DANN only
        X_combined = np.concatenate([X_patient_BM_scaled, X_train_use], axis=0).astype(np.float32)
        y_combined = np.concatenate([y, y_train_use], axis=0).astype(np.float32)
        domain_labels = np.concatenate([
            np.zeros(len(X_patient_BM_scaled)),   # BM domain
            np.ones(len(X_train_use))             # cell line domain
        ]).astype(np.float32)

        # ---------------- DANN ----------------
        print(f"[INFO] ({iter_name}) Training DANN model...")
        DANN = build_and_compile_model()
        DANN.set_weights(initial_weights)
        scheduler = DomainWeightScheduler(max_lambda=0.1, n_epochs=300)

        DANN.fit(
            X_combined,
            [y_combined, domain_labels],
            epochs=300,
            batch_size=32,
            verbose=0,
            callbacks=[scheduler]
        )

        # BM test
        DANN_output_BM = DANN.predict(X_test_BM, verbose=0)
        y_pred_DANN_BM = np.asarray(DANN_output_BM[0], dtype=float).reshape(-1)
        y_true_BM = np.asarray(y_test_BM, dtype=float).reshape(-1)

        r2_scores["DANN_BM"].append(r2_score(y_true_BM, y_pred_DANN_BM))
        corr_coeffs["DANN_BM"].append(spearmanr(y_true_BM, y_pred_DANN_BM)[0])
        mse_scores["DANN_BM"].append(mean_squared_error(y_true_BM, y_pred_DANN_BM))
        pearson_coeffs["DANN_BM"].append(pearsonr(y_true_BM, y_pred_DANN_BM)[0])

        # PB test
        DANN_output_PB = DANN.predict(X_test_PB, verbose=0)
        y_pred_DANN_PB = np.asarray(DANN_output_PB[0], dtype=float).reshape(-1)
        y_true_PB = np.asarray(y_test_PB, dtype=float).reshape(-1)

        r2_scores["DANN_PB"].append(r2_score(y_true_PB, y_pred_DANN_PB))
        corr_coeffs["DANN_PB"].append(spearmanr(y_true_PB, y_pred_DANN_PB)[0])
        mse_scores["DANN_PB"].append(mean_squared_error(y_true_PB, y_pred_DANN_PB))
        pearson_coeffs["DANN_PB"].append(pearsonr(y_true_PB, y_pred_DANN_PB)[0])

        # train (BM + cell line)
        DANN_output_train = DANN.predict(X_combined, verbose=0)
        y_pred_DANN_train = np.asarray(DANN_output_train[0], dtype=float).reshape(-1)
        y_true_combined = np.asarray(y_combined, dtype=float).reshape(-1)

        r2_scores["DANN_train"].append(r2_score(y_true_combined, y_pred_DANN_train))
        corr_coeffs["DANN_train"].append(spearmanr(y_true_combined, y_pred_DANN_train)[0])
        mse_scores["DANN_train"].append(mean_squared_error(y_true_combined, y_pred_DANN_train))
        pearson_coeffs["DANN_train"].append(pearsonr(y_true_combined, y_pred_DANN_train)[0])

        print(f"[INFO] ({iter_name}) DANN done.")

        # ---------------- patient-only ----------------
        print(f"[INFO] ({iter_name}) Training patient-only model...")
        patient_only = build_patient_only_model(input_dim=X_patient_BM_scaled.shape[1], lr=0.0005)

        early_stop_patient = EarlyStopping(
            monitor="loss",
            patience=20,
            restore_best_weights=True
        )

        patient_only.fit(
            X_patient_BM_scaled,
            y,
            epochs=300,
            batch_size=32,
            verbose=0,
            callbacks=[early_stop_patient]
        )

        # BM test
        y_pred_patient_BM = np.asarray(
            patient_only.predict(X_test_BM, verbose=0),
            dtype=float
        ).reshape(-1)

        r2_scores["patient_only_BM"].append(r2_score(y_test_BM, y_pred_patient_BM))
        corr_coeffs["patient_only_BM"].append(spearmanr(y_test_BM, y_pred_patient_BM)[0])
        mse_scores["patient_only_BM"].append(mean_squared_error(y_test_BM, y_pred_patient_BM))
        pearson_coeffs["patient_only_BM"].append(pearsonr(y_test_BM, y_pred_patient_BM)[0])

        # PB test
        y_pred_patient_PB = np.asarray(
            patient_only.predict(X_test_PB, verbose=0),
            dtype=float
        ).reshape(-1)

        r2_scores["patient_only_PB"].append(r2_score(y_test_PB, y_pred_patient_PB))
        corr_coeffs["patient_only_PB"].append(spearmanr(y_test_PB, y_pred_patient_PB)[0])
        mse_scores["patient_only_PB"].append(mean_squared_error(y_test_PB, y_pred_patient_PB))
        pearson_coeffs["patient_only_PB"].append(pearsonr(y_test_PB, y_pred_patient_PB)[0])

        # train
        y_pred_patient_train = np.asarray(
            patient_only.predict(X_patient_BM_scaled, verbose=0),
            dtype=float
        ).reshape(-1)

        r2_scores["patient_only_train"].append(r2_score(y, y_pred_patient_train))
        corr_coeffs["patient_only_train"].append(spearmanr(y, y_pred_patient_train)[0])
        mse_scores["patient_only_train"].append(mean_squared_error(y, y_pred_patient_train))
        pearson_coeffs["patient_only_train"].append(pearsonr(y, y_pred_patient_train)[0])

        print(f"[INFO] ({iter_name}) patient-only done.")

        # ---------- save results ----------
        for model in model_names:
            all_r2_scores[model].setdefault(n_samples, []).append(r2_scores[model][-1])
            all_corr_coeffs[model].setdefault(n_samples, []).append(corr_coeffs[model][-1])
            all_mse_scores[model].setdefault(n_samples, []).append(mse_scores[model][-1])
            all_pearson_coeffs[model].setdefault(n_samples, []).append(pearson_coeffs[model][-1])

    # ---------- average ----------
    print(f"[INFO] Finished all repeats for n_samples = {n_samples}, computing averages...")

    for model in model_names:
        avg_r2_scores[model][n_samples] = np.mean(r2_scores[model]) if len(r2_scores[model]) > 0 else np.nan
        avg_corr_coeffs[model][n_samples] = np.mean(corr_coeffs[model]) if len(corr_coeffs[model]) > 0 else np.nan
        avg_mse_scores[model][n_samples] = np.mean(mse_scores[model]) if len(mse_scores[model]) > 0 else np.nan
        avg_pearson_coeffs[model][n_samples] = np.mean(pearson_coeffs[model]) if len(pearson_coeffs[model]) > 0 else np.nan

        print(
            f"{model} | n={n_samples} | "
            f"R2={avg_r2_scores[model][n_samples]:.3f}, "
            f"Spearman={avg_corr_coeffs[model][n_samples]:.3f}, "
            f"Pearson={avg_pearson_coeffs[model][n_samples]:.3f}, "
            f"MSE={avg_mse_scores[model][n_samples]:.3f}"
        )

    # ---------- reset ----------
    for model in model_names:
        r2_scores[model] = []
        corr_coeffs[model] = []
        mse_scores[model] = []
        pearson_coeffs[model] = []






