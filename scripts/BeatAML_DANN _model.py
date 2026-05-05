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


#load cell line model

from tensorflow.keras.models import load_model
# Load the saved patient model
cell_line_model = load_model('/data/Cell line/cell_line_model_based on blood_less neuron.h5')
# Load training data
with h5py.File('/data/Cell line/training_data_blood.h5', 'r') as f:
    X_train = f['X_train'][:]
    y_train = f['y_train'][:]

# Load test data
with h5py.File('/data/Cell line/test_data_blood.h5', 'r') as f:
    X_test = f['X_test'][:]
    y_test = f['y_test'][:]
    # Create scaler object
scaler = StandardScaler()

# Fit scaler to training data and transform it
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# PB data
rowDrug_MACCS_patient_PB = pd.read_csv("/data/PB/rowDrug_MACCS_input_PB.csv", sep = ",")
rowDrug_Sig_patient_PB = pd.read_csv("/data/PB/rowDrug_Sig_input_PB.csv", sep = ",")
colDrug_MACCS_patient_PB = pd.read_csv("/data/PB/colDrug_MACCS_input_PB.csv", sep = ",")
colDrug_Sig_patient_PB = pd.read_csv("/data/PB/colDrug_Sig_input_PB.csv", sep = ",")
CellExp_patient_PB = pd.read_csv("/data/PB/cellExp_ssGSEA_input_PB_1221.csv", sep = ",")
HSA_label_patient_PB = pd.read_csv("/data/PB/HSA_label_PB.csv", sep = ",")


# PB data processing
input__patient_PB=pd.concat([rowDrug_MACCS_patient_PB,rowDrug_Sig_patient_PB,colDrug_MACCS_patient_PB,colDrug_Sig_patient_PB,CellExp_patient_PB,HSA_label_patient_PB],axis=1,join="inner") 
input_patient_PB_final=input__patient_PB.drop(labels = ['drug_row',"block_id","drug_col","SampleID"], axis=1)
new_data_patient_PB=input_patient_PB_final
X_patient_PB = new_data_patient_PB.drop("synergy_hsa", axis = 1).values
y_patient_PB = new_data_patient_PB.synergy_hsa.astype(float)

####BM data
rowDrug_MACCS_patient_BM = pd.read_csv("/data/BM/rowDrug_MACCS_input_BM.csv", sep = ",")
rowDrug_Sig_patient_BM = pd.read_csv("/data/BM/rowDrug_Sig_input_BM.csv", sep = ",")
colDrug_MACCS_patient_BM = pd.read_csv("/data/BM/colDrug_MACCS_input_BM.csv", sep = ",")
colDrug_Sig_patient_BM = pd.read_csv("/data/BM/colDrug_Sig_input_BM.csv", sep = ",")
CellExp_patient_BM = pd.read_csv("/data/BM/cellExp_ssGSEA_input_BM_1221.csv", sep = ",")
HSA_label_patient_BM = pd.read_csv("/data/BM/HSA_label_BM.csv", sep = ",")
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


# DANN

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
        Dropout(0.5, seed=seeds),
        Dense(128, activation='relu',
              kernel_regularizer=regularizers.l2(1e-4)),
        BatchNormalization(),
        Dropout(0.5, seed=seeds),
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
        loss_weights=[1.0, 0.0]   # initial domain loss weight = 0
    )
    return model


# Create and compile the model
transfer_model = build_and_compile_model()

# Save the initial weights
initial_weights = transfer_model.get_weights()


# 
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

class DomainWeightScheduler(Callback):
    def __init__(self, max_lambda=0.1, n_epochs=300):
        super().__init__()
        self.max_lambda = max_lambda
        self.n_epochs = n_epochs

    def on_epoch_begin(self, epoch, logs=None):
        #p is set from 0 to 1
        p = epoch / float(self.n_epochs)
        #lambda
        lam = self.max_lambda * (2. / (1. + np.exp(-10 * p)) - 1)
        # modifying the loss_weights： [task_loss_weight, domain_loss_weight]
        self.model.loss_weights = [1.0, lam]

###loop
from scipy.stats import spearmanr
samples = [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,30,40,50]
repeat = 5

r2_scores = {'xgb_train': [],'xgb_BM': [],'xgb_PB': [],'rf_train': [], 'rf_BM': [],'rf_PB': [], 'knn_train': [],'knn_BM': [],'knn_PB': [], 'lr_train': [],'lr_BM': [],'lr_PB': [], 'DANN_train': [],'DANN_BM': [],'DANN_PB': []}  # Add method of machine learning here
corr_coeffs = {'xgb_train': [],'xgb_BM': [],'xgb_PB': [],'rf_train': [], 'rf_BM': [],'rf_PB': [], 'knn_train': [],'knn_BM': [],'knn_PB': [], 'lr_train': [],'lr_BM': [],'lr_PB': [], 'DANN_train': [],'DANN_BM': [],'DANN_PB': []}
mse_scores = {'xgb_train': [],'xgb_BM': [],'xgb_PB': [],'rf_train': [], 'rf_BM': [],'rf_PB': [], 'knn_train': [],'knn_BM': [],'knn_PB': [], 'lr_train': [],'lr_BM': [],'lr_PB': [], 'DANN_train': [],'DANN_BM': [],'DANN_PB': []}
avg_r2_scores = {'xgb_train': {},'xgb_BM': {},'xgb_PB': {}, 'rf_train': {},'rf_BM': {},'rf_PB': {}, 'knn_train': {},'knn_BM': {},'knn_PB': {}, 'lr_train': {},'lr_BM': {},'lr_PB': {}, 'DANN_train': {},'DANN_BM': {},'DANN_PB': {}}
all_r2_scores = {'xgb_train': {},'xgb_BM': {},'xgb_PB': {}, 'rf_train': {},'rf_BM': {},'rf_PB': {}, 'knn_train': {},'knn_BM': {},'knn_PB': {}, 'lr_train': {},'lr_BM': {},'lr_PB': {}, 'DANN_train': {},'DANN_BM': {},'DANN_PB': {}}
all_corr_coeffs = {'xgb_train': {},'xgb_BM': {},'xgb_PB': {}, 'rf_train': {},'rf_BM': {},'rf_PB': {}, 'knn_train': {},'knn_BM': {},'knn_PB': {}, 'lr_train': {},'lr_BM': {},'lr_PB': {}, 'DANN_train': {},'DANN_BM': {},'DANN_PB': {}}
all_mse_scores = {'xgb_train': {},'xgb_BM': {},'xgb_PB': {}, 'rf_train': {},'rf_BM': {},'rf_PB': {}, 'knn_train': {},'knn_BM': {},'knn_PB': {}, 'lr_train': {},'lr_BM': {},'lr_PB': {}, 'DANN_train': {},'DANN_BM': {},'DANN_PB': {}}
avg_corr_coeffs = {'xgb_train': {},'xgb_BM': {},'xgb_PB': {}, 'rf_train': {},'rf_BM': {},'rf_PB': {},'knn_train': {},'knn_BM': {},'knn_PB': {}, 'lr_train': {},'lr_BM': {},'lr_PB': {}, 'DANN_train': {},'DANN_BM': {},'DANN_PB': {}}
avg_mse_scores = {'xgb_train': {},'xgb_BM': {},'xgb_PB': {}, 'rf_train': {},'rf_BM': {},'rf_PB': {}, 'knn_train': {},'knn_BM': {},'knn_PB': {}, 'lr_train': {},'lr_BM': {},'lr_PB': {}, 'DANN_train': {},'DANN_BM': {},'DANN_PB': {}}

# Add these dictionaries along with your other dictionaries
pearson_coeffs = {'xgb_train': [],'xgb_BM': [],'xgb_PB': [],'rf_train': [], 'rf_BM': [],'rf_PB': [],'knn_train': [],'knn_BM': [],'knn_PB': [], 'lr_train': [],'lr_BM': [],'lr_PB': [], 'DANN_train': [],'DANN_BM': [],'DANN_PB': []}
avg_pearson_coeffs = {'xgb_train': {},'xgb_BM': {},'xgb_PB': {}, 'rf_train': {},'rf_BM': {},'rf_PB': {}, 'knn_train': {},'knn_BM': {},'knn_PB': {}, 'lr_train': {},'lr_BM': {},'lr_PB': {}, 'DANN_train': {},'DANN_BM': {},'DANN_PB': {}}
all_pearson_coeffs = {'xgb_train': {},'xgb_BM': {},'xgb_PB': {}, 'rf_train': {},'rf_BM': {},'rf_PB': {}, 'knn_train': {},'knn_BM': {},'knn_PB': {}, 'lr_train': {},'lr_BM': {},'lr_PB': {}, 'DANN_train': {},'DANN_BM': {},'DANN_PB': {}}

lower_ci_corr_coeffs = {'xgb_train': {},'xgb_BM': {},'xgb_PB': {}, 'rf_train': {},'rf_BM': {},'rf_PB': {}, 'knn_train': {},'knn_BM': {},'knn_PB': {}, 'lr_train': {},'lr_BM': {},'lr_PB': {}, 'DANN_train': {},'DANN_BM': {},'DANN_PB': {}}
upper_ci_corr_coeffs = {'xgb_train': {},'xgb_BM': {},'xgb_PB': {}, 'rf_train': {},'rf_BM': {},'rf_PB': {}, 'knn_train': {},'knn_BM': {},'knn_PB': {}, 'lr_train': {},'lr_BM': {},'lr_PB': {}, 'DANN_train': {},'DANN_BM': {},'DANN_PB': {}}
# Create a dictionary to store selected samples in each iteration
selected_samples = {'iteration': [], 'samples': []}

###
unique_combinations = {}

for n_samples in samples:
    print(f"\n========== Use {n_samples} BM patient(s) for training ==========")
    unique_combinations[n_samples] = set()
    
    for i in range(repeat):
        print("-" * 60)
        print(f"[INFO] n_samples = {n_samples}, iteration = {i+1}/{repeat}")
        
        # 1) BM--SampleID
        while True:
            if n_samples == 1:
                bm_ids_modified = [
                    sid for sid in bm_sample_ids
                    if sid not in ["AML23", "AML40", "AML72", "AML88", "AML91"]
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
        selected_samples["samples"].append(selected_sample_ids)
        
        # 2) BM subset
        subset_train_BM = bm_data[bm_data["SampleID"].isin(selected_sample_ids)].copy()
        print("[DEBUG] BM train subset shape:", subset_train_BM.shape)
        
        X_patient_BM = subset_train_BM.drop(
            columns=["SampleID", LABEL_COL]
        ).values.astype(np.float32)
        X_patient_BM_scaled = scaler.transform(X_patient_BM)
        y = subset_train_BM[LABEL_COL].values.astype(np.float32)
        
        # 3) BM test set
        subset_test_BM = bm_data[~bm_data["SampleID"].isin(selected_sample_ids)].copy()
        print("[DEBUG] BM test subset shape:", subset_test_BM.shape)
        
        X_test_BM = subset_test_BM.drop(
            columns=["SampleID", LABEL_COL]
        ).values.astype(np.float32)
        X_test_BM = scaler.transform(X_test_BM)
        y_test_BM = subset_test_BM[LABEL_COL].values.astype(np.float32)
        
        # 4) identifying the cell line data
        num_rows = subset_train_BM.shape[0]
        train_size = num_rows / 6829
        print(f"[DEBUG] num_rows = {num_rows}, train_size = {train_size:.4f}")
        
        X_train_use, X_test_no, y_train_use, y_test_no = train_test_split(
            X_train_scaled, y_train,
            train_size=train_size,
            random_state=seeds
        )
        print("[DEBUG] X_train_use shape:", X_train_use.shape)
        
        # 5) combine BM + cell line samples
        X_combined = np.concatenate(
            [X_patient_BM_scaled, X_train_use],
            axis=0
        ).astype(np.float32)
        y_combined = np.concatenate(
            [y, y_train_use],
            axis=0
        ).astype(np.float32)
        domain_labels = np.concatenate([
            np.zeros(len(X_patient_BM_scaled)),   # BM domain
            np.ones(len(X_train_use))             # cell line domain
        ]).astype(np.float32)
        
        # ---------------- XGB ----------------
        xgb = XGBRegressor(random_state=seeds, max_depth=3, n_estimators=30)
        xgb.fit(X_patient_BM_scaled, y)
        
        y_pred_xgb = xgb.predict(X_test_BM)
        r2_xgb = r2_score(y_test_BM, y_pred_xgb)
        r2_scores['xgb_BM'].append(r2_xgb)
        spearman_corr_xgb, _ = spearmanr(y_test_BM, y_pred_xgb)
        corr_coeffs['xgb_BM'].append(spearman_corr_xgb)
        mse_xgb = mean_squared_error(y_test_BM, y_pred_xgb)
        mse_scores['xgb_BM'].append(mse_xgb)
        pearson_corr_xgb, _ = pearsonr(y_test_BM, y_pred_xgb)
        pearson_coeffs['xgb_BM'].append(pearson_corr_xgb)
        
        y_pred_xgb_PB = xgb.predict(X_test_PB)
        r2_xgb_PB = r2_score(y_test_PB, y_pred_xgb_PB)
        r2_scores['xgb_PB'].append(r2_xgb_PB)
        spearman_corr_xgb_PB, _ = spearmanr(y_test_PB, y_pred_xgb_PB)
        corr_coeffs['xgb_PB'].append(spearman_corr_xgb_PB)
        mse_xgb_PB = mean_squared_error(y_test_PB, y_pred_xgb_PB)
        mse_scores['xgb_PB'].append(mse_xgb_PB)
        pearson_corr_xgb_PB, _ = pearsonr(y_test_PB, y_pred_xgb_PB)
        pearson_coeffs['xgb_PB'].append(pearson_corr_xgb_PB)
        
        y_pred_xgb_train = xgb.predict(X_patient_BM_scaled)
        r2_xgb_train = r2_score(y, y_pred_xgb_train)
        r2_scores['xgb_train'].append(r2_xgb_train)
        spearman_corr_xgb_train, _ = spearmanr(y, y_pred_xgb_train)
        corr_coeffs['xgb_train'].append(spearman_corr_xgb_train)
        mse_xgb_train = mean_squared_error(y, y_pred_xgb_train)
        mse_scores['xgb_train'].append(mse_xgb_train)
        pearson_corr_xgb_train, _ = pearsonr(y, y_pred_xgb_train)
        pearson_coeffs['xgb_train'].append(pearson_corr_xgb_train)
        print(f"[INFO] ({iter_name}) XGB done. pearson_corr_xgb={pearson_corr_xgb:.3f}, pearson_corr_xgb_PB={pearson_corr_xgb_PB:.3f}")
        
        # ---------------- RF ----------------
        print(f"[INFO] ({iter_name}) Training RandomForestRegressor...")
        rf = RandomForestRegressor(random_state=seeds, max_depth=3, n_estimators=30)
        rf.fit(X_patient_BM_scaled, y)
        
        y_pred_rf = rf.predict(X_test_BM)
        r2_rf = r2_score(y_test_BM, y_pred_rf)
        r2_scores['rf_BM'].append(r2_rf)
        spearman_corr_rf, _ = spearmanr(y_test_BM, y_pred_rf)
        corr_coeffs['rf_BM'].append(spearman_corr_rf)
        mse_rf = mean_squared_error(y_test_BM, y_pred_rf)
        mse_scores['rf_BM'].append(mse_rf)
        pearson_corr_rf, _ = pearsonr(y_test_BM, y_pred_rf)
        pearson_coeffs['rf_BM'].append(pearson_corr_rf)
        
        y_pred_rf_PB = rf.predict(X_test_PB)
        r2_rf_PB = r2_score(y_test_PB, y_pred_rf_PB)
        r2_scores['rf_PB'].append(r2_rf_PB)
        spearman_corr_rf_PB, _ = spearmanr(y_test_PB, y_pred_rf_PB)
        corr_coeffs['rf_PB'].append(spearman_corr_rf_PB)
        mse_rf_PB = mean_squared_error(y_test_PB, y_pred_rf_PB)
        mse_scores['rf_PB'].append(mse_rf_PB)
        pearson_corr_rf_PB, _ = pearsonr(y_test_PB, y_pred_rf_PB)
        pearson_coeffs['rf_PB'].append(pearson_corr_rf_PB)
        print(f"[INFO] ({iter_name}) RF done. pearson_corr_rf={pearson_corr_rf:.3f}, pearson_corr_rf_PB={pearson_corr_rf_PB:.3f}")
        
        y_pred_rf_train = rf.predict(X_patient_BM_scaled)
        r2_rf_train = r2_score(y, y_pred_rf_train)
        r2_scores['rf_train'].append(r2_rf_train)
        spearman_corr_rf_train, _ = spearmanr(y, y_pred_rf_train)
        corr_coeffs['rf_train'].append(spearman_corr_rf_train)
        mse_rf_train = mean_squared_error(y, y_pred_rf_train)
        mse_scores['rf_train'].append(mse_rf_train)
        pearson_corr_rf_train, _ = pearsonr(y, y_pred_rf_train)
        pearson_coeffs['rf_train'].append(pearson_corr_rf_train)
        
        # ---------------- KNN ----------------
        print(f"[INFO] ({iter_name}) Training KNN...")
        n_train_samples = X_patient_BM_scaled.shape[0]
        if n_train_samples >= 2:
            n_neighbors = min(5, n_train_samples - 1)
        else:
            n_neighbors = 1
        
        knn = KNeighborsRegressor(n_neighbors=n_neighbors)
        knn.fit(X_patient_BM_scaled, y)
        
        y_pred_knn = knn.predict(X_test_BM)
        r2_knn = r2_score(y_test_BM, y_pred_knn)
        r2_scores['knn_BM'].append(r2_knn)
        spearman_corr_knn, _ = spearmanr(y_test_BM, y_pred_knn)
        corr_coeffs['knn_BM'].append(spearman_corr_knn)
        mse_knn = mean_squared_error(y_test_BM, y_pred_knn)
        mse_scores['knn_BM'].append(mse_knn)
        pearson_corr_knn, _ = pearsonr(y_test_BM, y_pred_knn)
        pearson_coeffs['knn_BM'].append(pearson_corr_knn)
        
        y_pred_knn_PB = knn.predict(X_test_PB)
        r2_knn_PB = r2_score(y_test_PB, y_pred_knn_PB)
        r2_scores['knn_PB'].append(r2_knn_PB)
        spearman_corr_knn_PB, _ = spearmanr(y_test_PB, y_pred_knn_PB)
        corr_coeffs['knn_PB'].append(spearman_corr_knn_PB)
        mse_knn_PB = mean_squared_error(y_test_PB, y_pred_knn_PB)
        mse_scores['knn_PB'].append(mse_knn_PB)
        pearson_corr_knn_PB, _ = pearsonr(y_test_PB, y_pred_knn_PB)
        pearson_coeffs['knn_PB'].append(pearson_corr_knn_PB)
        print(f"[INFO] ({iter_name}) KNN done. pearson_corr_knn={pearson_corr_knn:.3f}, pearson_corr_knn_PB={pearson_corr_knn_PB:.3f}")
        
        y_pred_knn_train = knn.predict(X_patient_BM_scaled)
        r2_knn_train = r2_score(y, y_pred_knn_train)
        r2_scores['knn_train'].append(r2_knn_train)
        spearman_corr_knn_train, _ = spearmanr(y, y_pred_knn_train)
        corr_coeffs['knn_train'].append(spearman_corr_knn_train)
        mse_knn_train = mean_squared_error(y, y_pred_knn_train)
        mse_scores['knn_train'].append(mse_knn_train)
        pearson_corr_knn_train, _ = pearsonr(y, y_pred_knn_train)
        pearson_coeffs['knn_train'].append(pearson_corr_knn_train)
        
        # ---------------- Linear Regression ----------------
        print(f"[INFO] ({iter_name}) Training LinearRegression...")
        n_train_samples = X_patient_BM_scaled.shape[0]

        # if the number of samples >= 2
        if n_train_samples >= 2:

            alphas = np.logspace(-3, 3, 13)  # 1e-3 ... 1e3
            # set CV
            cv_splits = min(5, n_train_samples)

            lr = RidgeCV(alphas=alphas, fit_intercept=True, cv=cv_splits)
            lr.fit(X_patient_BM_scaled, y)
        else:
        # only a sample, using LinearRegression

            lr = LinearRegression(fit_intercept=True)
            lr.fit(X_patient_BM_scaled, y)
        
        y_pred_lr = lr.predict(X_test_BM)
        r2_lr = r2_score(y_test_BM, y_pred_lr)
        r2_scores['lr_BM'].append(r2_lr)
        spearman_corr_lr, _ = spearmanr(y_test_BM, y_pred_lr)
        corr_coeffs['lr_BM'].append(spearman_corr_lr)
        mse_lr = mean_squared_error(y_test_BM, y_pred_lr)
        mse_scores['lr_BM'].append(mse_lr)
        pearson_corr_lr, _ = pearsonr(y_test_BM, y_pred_lr)
        pearson_coeffs['lr_BM'].append(pearson_corr_lr)
        
        y_pred_lr_PB = lr.predict(X_test_PB)
        r2_lr_PB = r2_score(y_test_PB, y_pred_lr_PB)
        r2_scores['lr_PB'].append(r2_lr_PB)
        spearman_corr_lr_PB, _ = spearmanr(y_test_PB, y_pred_lr_PB)
        corr_coeffs['lr_PB'].append(spearman_corr_lr_PB)
        mse_lr_PB = mean_squared_error(y_test_PB, y_pred_lr_PB)
        mse_scores['lr_PB'].append(mse_lr_PB)
        pearson_corr_lr_PB, _ = pearsonr(y_test_PB, y_pred_lr_PB)
        pearson_coeffs['lr_PB'].append(pearson_corr_lr_PB)
        print(f"[INFO] ({iter_name}) LinearRegression done. pearson_corr_lr={pearson_corr_lr:.3f}, pearson_corr_lr_PB={pearson_corr_lr_PB:.3f}")
        
        y_pred_lr_train = lr.predict(X_patient_BM_scaled)
        r2_lr_train = r2_score(y, y_pred_lr_train)
        r2_scores['lr_train'].append(r2_lr_train)
        spearman_corr_lr_train, _ = spearmanr(y, y_pred_lr_train)
        corr_coeffs['lr_train'].append(spearman_corr_lr_train)
        mse_lr_train = mean_squared_error(y, y_pred_lr_train)
        mse_scores['lr_train'].append(mse_lr_train)
        pearson_corr_lr_train, _ = pearsonr(y, y_pred_lr_train)
        pearson_coeffs['lr_train'].append(pearson_corr_lr_train)
        
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
        DANN_output  = DANN.predict(X_test_BM, verbose=0)
        y_pred_DANN = np.asarray(DANN_output[0], dtype=float).reshape(-1)
        y_true_BM = y_test_BM.reshape(-1)
        
        r2_DANN = r2_score(y_true_BM, y_pred_DANN)
        r2_scores['DANN_BM'].append(r2_DANN)
        spearman_corr_DANN, _ = spearmanr(y_true_BM, y_pred_DANN)
        corr_coeffs['DANN_BM'].append(spearman_corr_DANN)
        mse_DANN = mean_squared_error(y_true_BM, y_pred_DANN)
        mse_scores['DANN_BM'].append(mse_DANN)
        pearson_corr_DANN, _ = pearsonr(y_true_BM, y_pred_DANN)
        pearson_coeffs['DANN_BM'].append(pearson_corr_DANN)
        print(f"[INFO] ({iter_name}) DANN BM done. pearson_corr_DANN={pearson_corr_DANN:.3f}")
        
        # PB test
        DANN_output_PB = DANN.predict(X_test_PB, verbose=0)
        y_pred_DANN_PB = np.asarray(DANN_output_PB[0], dtype=float).reshape(-1)
        y_true_PB = np.asarray(y_test_PB, dtype=float).reshape(-1)
        r2_DANN_PB = r2_score(y_test_PB, y_pred_DANN_PB)
        r2_scores['DANN_PB'].append(r2_DANN_PB) 
        spearman_corr_DANN_PB, _ = spearmanr(y_test_PB, y_pred_DANN_PB)
        corr_coeffs['DANN_PB'].append(spearman_corr_DANN_PB)
        mse_DANN_PB = mean_squared_error(y_test_PB, y_pred_DANN_PB)
        mse_scores['DANN_PB'].append(mse_DANN_PB)
        pearson_corr_DANN_PB, _ = pearsonr(y_test_PB, y_pred_DANN_PB)
        pearson_coeffs['DANN_PB'].append(pearson_corr_DANN_PB)
        print(f"[INFO] ({iter_name}) DANN PB done. pearson_corr_DANN_PB={pearson_corr_DANN_PB:.3f}")
        
        # train (BM + cell line)
        DANN_output_train  = DANN.predict(X_combined, verbose=0)
        y_pred_DANN_train = np.asarray(DANN_output_train[0], dtype=float).reshape(-1)
        y_true_combined = y_combined.reshape(-1)
        
        r2_DANN_train = r2_score(y_true_combined, y_pred_DANN_train)
        r2_scores['DANN_train'].append(r2_DANN_train)
        spearman_corr_DANN_train, _ = spearmanr(y_true_combined, y_pred_DANN_train)
        corr_coeffs['DANN_train'].append(spearman_corr_DANN_train)
        mse_DANN_train = mean_squared_error(y_true_combined, y_pred_DANN_train)
        mse_scores['DANN_train'].append(mse_DANN_train)
        pearson_corr_DANN_train, _ = pearsonr(y_true_combined, y_pred_DANN_train)
        pearson_coeffs['DANN_train'].append(pearson_corr_DANN_train)
        print(f"[INFO] ({iter_name}) DANN train done. R2={r2_DANN_train:.3f}")
        
        # ---------- save in all_* ----------
        for model in r2_scores.keys():
            if n_samples not in all_r2_scores[model]:
                all_r2_scores[model][n_samples] = []
            all_r2_scores[model][n_samples].append(r2_scores[model][-1])
            
            if n_samples not in all_corr_coeffs[model]:
                all_corr_coeffs[model][n_samples] = []
            all_corr_coeffs[model][n_samples].append(corr_coeffs[model][-1])
            
            if n_samples not in all_mse_scores[model]:
                all_mse_scores[model][n_samples] = []
            all_mse_scores[model][n_samples].append(mse_scores[model][-1])
            
            if n_samples not in all_pearson_coeffs[model]:
                all_pearson_coeffs[model][n_samples] = []
            all_pearson_coeffs[model][n_samples].append(pearson_coeffs[model][-1])
    
    # --------------------
    print(f"[INFO] Finished all repeats for n_samples = {n_samples}, computing averages...")
    for model, scores in r2_scores.items():
        avg_r2 = np.mean(scores)
        avg_corr_coeff = np.mean(corr_coeffs[model])
        avg_mse = np.mean(mse_scores[model])
        avg_r2_scores[model][n_samples] = avg_r2
        avg_corr_coeffs[model][n_samples] = avg_corr_coeff
        avg_mse_scores[model][n_samples] = avg_mse
        print(f"Average R2 score for {model} with {n_samples} samples: {avg_r2:.2f}")
    
    for model, scores in pearson_coeffs.items():
        avg_pearson = np.mean(scores)
        avg_pearson_coeffs[model][n_samples] = avg_pearson
        print(f"Average Pearson correlation for {model} with {n_samples} samples: {avg_pearson:.2f}")
    
    #
    for model in r2_scores:
        r2_scores[model] = []
        corr_coeffs[model] = []
        mse_scores[model] = []
        pearson_coeffs[model] = []
### Plot the average correlation coefficients for BM
plt.figure()
for model, scores in avg_corr_coeffs.items():
    if '_BM' in model:
        x = list(scores.keys())
        y = list(scores.values())
        plt.plot(x, y, label=model)

plt.xlabel('Number of samples')
plt.ylabel('Average Correlation Coefficient (BM)')
plt.legend()

# Save the plot as a high-quality PDF
plt.savefig('avg_corr_coeff_plot_BM_5times_blood_try.pdf', dpi=900, format='pdf')
plt.show()

# Plot the average correlation coefficients for PB
plt.figure()
for model, scores in avg_corr_coeffs.items():
    if '_PB' in model:
        x = list(scores.keys())
        y = list(scores.values())
        plt.plot(x, y, label=model)

plt.xlabel('Number of samples')
plt.ylabel('Average Correlation Coefficient (PB)')
plt.legend()

# Save the plot as a high-quality PDF
plt.savefig('avg_corr_coeff_plot_PB_5times_blood_try.pdf', dpi=900, format='pdf')
plt.show()
###save the model--based on 50 samples
DANN.save("DANN_50_samples_AML.keras")
####save the summary files
def write_all_results_to_csv(file_name, results_dict):
    with open(file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        header = ['n_samples'] + list(results_dict.keys())
        csv_writer.writerow(header)

        for n_samples in results_dict[next(iter(results_dict))].keys():
            for i, scores in enumerate(zip(*[results_dict[model][n_samples] for model in results_dict])):
                row = [n_samples] + list(scores)
                csv_writer.writerow(row)

# Write all R2 scores, correlation coefficients, and MSEs to separate CSV files
write_all_results_to_csv('all_r2_scores_5times_AML_blood.csv', all_r2_scores)
write_all_results_to_csv('all_corr_coeffs_5times_AML_blood.csv', all_corr_coeffs)
write_all_results_to_csv('all_mse_scores_5times_AML_blood.csv', all_mse_scores)
write_all_results_to_csv('all_pearson_coeffs_5times_AML_blood.csv', all_pearson_coeffs)
