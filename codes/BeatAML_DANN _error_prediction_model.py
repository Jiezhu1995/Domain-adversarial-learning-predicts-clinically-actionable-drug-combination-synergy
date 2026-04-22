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


# PB data
rowDrug_MACCS_patient_PB = pd.read_csv("rowDrug_MACCS_input_PB.csv", sep = ",")
rowDrug_Sig_patient_PB = pd.read_csv("rowDrug_Sig_input_PB.csv", sep = ",")
colDrug_MACCS_patient_PB = pd.read_csv("colDrug_MACCS_input_PB.csv", sep = ",")
colDrug_Sig_patient_PB = pd.read_csv("colDrug_Sig_input_PB.csv", sep = ",")
CellExp_patient_PB = pd.read_csv("cellExp_ssGSEA_input_PB_1221.csv", sep = ",")
HSA_label_patient_PB = pd.read_csv("HSA_label_PB.csv", sep = ",")


# PB data processing
input__patient_PB=pd.concat([rowDrug_MACCS_patient_PB,rowDrug_Sig_patient_PB,colDrug_MACCS_patient_PB,colDrug_Sig_patient_PB,CellExp_patient_PB,HSA_label_patient_PB],axis=1,join="inner") 
input_patient_PB_final=input__patient_PB.drop(labels = ['drug_row',"block_id","drug_col","SampleID"], axis=1)
new_data_patient_PB=input_patient_PB_final
X_patient_PB = new_data_patient_PB.drop("synergy_hsa", axis = 1).values
y_patient_PB = new_data_patient_PB.synergy_hsa.astype(float)

####BM data
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

DANN.save("DANN_50_samples_AML.keras")



####Error Predictin Model

test_data = pd.read_csv('BM_test_remaining_samples.csv')

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import itertools
import pickle
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr
from joblib import parallel_backend
from pathlib import Path
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor


# ============================================================

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
        n_estimators=300,      
        max_depth=6,             
        min_samples_leaf=30,     
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


# training the final  model
X_full = scaler.transform(
    test_data.drop(columns=["SampleID", "synergy_hsa"]).values
)
y_full = test_data["synergy_hsa"].values

y_full_pred = np.squeeze(DANN.predict(X_full, verbose=0)[0])
Xerr_full = build_error_features(X_full, y_full_pred)
err_target_full = np.log(np.abs(y_full - y_full_pred) + 1e-3)

err_model_final = fit_error_model_crossfit_rf(
    Xerr_full,
    err_target_full,
    random_state=0
)

joblib.dump({
    "error_model": err_model_final,
    "scaler": scaler
}, "error_pipeline.pkl")



# ===== 6. save the model =====
joblib.dump({
    "error_model": err_model_final,
    "scaler": scaler
}, "error_pipeline.pkl")



###plot
import numpy as np
from scipy.stats import wilcoxon

def wilcoxon_kept_vs_raw(df_res, metric="pearson", use_empirical=False, alternative="two-sided"):
    """
    Wilcoxon signed-rank test comparing kept vs raw across reps for each kept_ratio_target.

    Parameters
    ----------
    df_res : pd.DataFrame
        Output of run_selective_experiment().
    metric : str
        One of {"spearman", "pearson", "mse"}.
    use_empirical : bool
        If True, group by empirical kept_ratio_emp (binned by target) is NOT recommended.
        Keep False to use stable kept_ratio_target grid.
    alternative : str
        'two-sided', 'greater' (kept > raw), or 'less' (kept < raw).

    Returns
    -------
    pd.DataFrame
        One row per kept_ratio_target with n pairs, W, pvalue.
    """
    metric = metric.lower()
    col_kept = f"{metric}_kept"
    col_raw  = f"{metric}_raw"

    if col_kept not in df_res.columns or col_raw not in df_res.columns:
        raise ValueError(f"Columns {col_kept} / {col_raw} not found in df_res.")

    group_col = "kept_ratio_target" if not use_empirical else "kept_ratio_emp"

    results = []
    for kr, g in df_res.groupby(group_col):
        kept = g[col_kept].to_numpy(dtype=float)
        raw  = g[col_raw].to_numpy(dtype=float)

        # drop NaN pairs
        mask = ~np.isnan(kept) & ~np.isnan(raw)
        kept = kept[mask]
        raw  = raw[mask]

        n = len(kept)
        if n < 5:
            results.append({"kept_ratio": kr, "n": n, "W": np.nan, "p": np.nan, "note": "too few pairs"})
            continue

        diff = kept - raw
        if np.allclose(diff, 0):
            results.append({"kept_ratio": kr, "n": n, "W": 0.0, "p": 1.0, "note": "all differences ~0"})
            continue

        try:
            # zero_method='wilcox' drops zero diffs; good default
            W, p = wilcoxon(kept, raw, alternative=alternative, zero_method="wilcox")
            results.append({"kept_ratio": kr, "n": n, "W": float(W), "p": float(p), "note": ""})
        except Exception as e:
            results.append({"kept_ratio": kr, "n": n, "W": np.nan, "p": np.nan, "note": str(e)})

    out = pd.DataFrame(results).sort_values("kept_ratio").reset_index(drop=True)
    return out


# -------------------------
# USAGE
# -------------------------
labels = [str(p) for p in range(50, 101, 10)]  # x-axis labels if you want

# Pearson: kept vs raw
wil_pe = wilcoxon_kept_vs_raw(df_res, metric="pearson", alternative="two-sided")
print("Wilcoxon (Pearson kept vs raw) by kept_ratio_target:")
print(wil_pe)

# Spearman: kept vs raw
wil_sp = wilcoxon_kept_vs_raw(df_res, metric="spearman", alternative="two-sided")
print("\nWilcoxon (Spearman kept vs raw) by kept_ratio_target:")
print(wil_sp)

# MSE: kept vs raw
wil_mse = wilcoxon_kept_vs_raw(df_res, metric="mse", alternative="two-sided")
print("\nWilcoxon (MSE kept vs raw) by kept_ratio_target:")
print(wil_mse)



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
# -------------------------
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

    # correlations: usually expect kept > raw  (one-sided "greater") or two-sided
    p_sp  = _wilcoxon_by_kept_ratio(df, "spearman_kept", "spearman_raw", alternative="two-sided")
    p_pe  = _wilcoxon_by_kept_ratio(df, "pearson_kept",  "pearson_raw",  alternative="two-sided")

    # MSE: expect kept < raw  (recommended)
    p_mse = _wilcoxon_by_kept_ratio(df, "mse_kept", "mse_raw", alternative="less")

    # -------- style --------
    COLOR_KEPT_FILL = "#C7DFFD"
    COLOR_RAW_FILL  = "#BFE6DF"
    COLOR_KEPT_EDGE = "#377483"
    COLOR_RAW_EDGE  = "#4F845C"
    COLOR_LINE = "#F6C8A8"

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
            ylabel, title, y_min, y_max, p_map = "Spearman correlation", "Spearman (kept vs raw)", 0.44, 0.56, p_sp

        elif metric == "pe":
            kept_m, kept_s = g["pe_keep_mean"], g["pe_keep_sem"]
            raw_m,  raw_s  = g["pe_raw_mean"],  g["pe_raw_sem"]
            ylabel, title, y_min, y_max, p_map = "Pearson correlation", "Pearson (kept vs raw)", 0.5, 0.64, p_pe

        else:  # mse
            kept_m, kept_s = g["mse_keep_mean"], g["mse_keep_sem"]
            raw_m,  raw_s  = g["mse_raw_mean"],  g["mse_raw_sem"]
            ylabel, title, y_min, y_max, p_map = "MSE", "MSE (kept vs raw) + Wilcoxon significance", 300, 600, p_mse

        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax2 = ax.twinx()
        ax.set_ylim(y_min, y_max)

        # ---- remove background ----
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax2.set_facecolor("white")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax.grid(False)

        # bars (error bar color = edge)
        ax.bar(
            x - width/2, kept_m, width,
            yerr=kept_s,
            facecolor=COLOR_KEPT_FILL,
            edgecolor=COLOR_KEPT_EDGE,
            linewidth=1.2,
            label=f"{ylabel} (kept)",
            error_kw=dict(ecolor=COLOR_KEPT_EDGE, elinewidth=0.8, capsize=3, capthick=0.8)
        )

        ax.bar(
            x + width/2, raw_m, width,
            yerr=raw_s,
            facecolor=COLOR_RAW_FILL,
            edgecolor=COLOR_RAW_EDGE,
            linewidth=1.2,
            label=f"{ylabel} (raw)",
            error_kw=dict(ecolor=COLOR_RAW_EDGE, elinewidth=0.8, capsize=3, capthick=0.8)
        )

        # kept ratio line (errorbar color = line)
        ax2.set_ylim(0, 1.05)
        ax2.errorbar(
            x, g["kept_ratio_emp_mean"],
            yerr=g["kept_ratio_emp_sem"],
            marker="o", color=COLOR_LINE,
            linewidth=1.8, capsize=3,
            ecolor=COLOR_LINE, elinewidth=0.8,
            label="Kept ratio"
        )

        ax.set_ylabel(ylabel)
        ax2.set_ylabel("Kept ratio")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Target kept ratio (%)")

        # ---- add significance stars (NOW includes MSE) ----
        yr = ax.get_ylim()[1] - ax.get_ylim()[0]
        y_offset = max(yr * 0.03, 0.01)
        for i, kr in enumerate(g["kept_ratio_target"]):
            y_top = max(
                kept_m.iloc[i] + (kept_s.iloc[i] if not np.isnan(kept_s.iloc[i]) else 0),
                raw_m.iloc[i]  + (raw_s.iloc[i]  if not np.isnan(raw_s.iloc[i])  else 0)
            )
            _add_star(ax, x[i], y_top, p_map.get(kr, np.nan), y_offset)

        # legend below
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(
            h1 + h2, l1 + l2,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
            ncol=3,
            frameon=False
        )

        fig.tight_layout()
        plt.savefig(f"{out_prefix}_{metric}.pdf", bbox_inches="tight")
        plt.savefig(f"{out_prefix}_{metric}.png", dpi=300, bbox_inches="tight")
        plt.show()

    _one("sp")
    _one("pe")
    _one("mse")

    return g


# Example usage:
summary = plot_bars_with_kept_ratio_line(df_res, out_prefix="mondrian_selective")
print(summary)


# PB test

test_data=input__patient_PB.drop(labels = ['drug_row',"block_id","drug_col"], axis=1)

# ============================================================

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
        n_estimators=300,      
        max_depth=6,             
        min_samples_leaf=30,     
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
# -------------------------
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

    # correlations: usually expect kept > raw  (one-sided "greater") or two-sided
    p_sp  = _wilcoxon_by_kept_ratio(df, "spearman_kept", "spearman_raw", alternative="two-sided")
    p_pe  = _wilcoxon_by_kept_ratio(df, "pearson_kept",  "pearson_raw",  alternative="two-sided")

    # MSE: expect kept < raw  (recommended)
    p_mse = _wilcoxon_by_kept_ratio(df, "mse_kept", "mse_raw", alternative="less")

    # -------- style --------
    COLOR_KEPT_FILL = "#C7DFFD"
    COLOR_RAW_FILL  = "#BFE6DF"
    COLOR_KEPT_EDGE = "#377483"
    COLOR_RAW_EDGE  = "#4F845C"
    COLOR_LINE = "#F6C8A8"

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
            ylabel, title, y_min, y_max, p_map = "Spearman correlation", "Spearman (kept vs raw)", 0.44, 0.56, p_sp

        elif metric == "pe":
            kept_m, kept_s = g["pe_keep_mean"], g["pe_keep_sem"]
            raw_m,  raw_s  = g["pe_raw_mean"],  g["pe_raw_sem"]
            ylabel, title, y_min, y_max, p_map = "Pearson correlation", "Pearson (kept vs raw)", 0.5, 0.60, p_pe

        else:  # mse
            kept_m, kept_s = g["mse_keep_mean"], g["mse_keep_sem"]
            raw_m,  raw_s  = g["mse_raw_mean"],  g["mse_raw_sem"]
            ylabel, title, y_min, y_max, p_map = "MSE", "MSE (kept vs raw) + Wilcoxon significance", 350, 650, p_mse

        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax2 = ax.twinx()
        ax.set_ylim(y_min, y_max)

        # ---- remove background ----
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax2.set_facecolor("white")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax.grid(False)

        # bars (error bar color = edge)
        ax.bar(
            x - width/2, kept_m, width,
            yerr=kept_s,
            facecolor=COLOR_KEPT_FILL,
            edgecolor=COLOR_KEPT_EDGE,
            linewidth=1.2,
            label=f"{ylabel} (kept)",
            error_kw=dict(ecolor=COLOR_KEPT_EDGE, elinewidth=0.8, capsize=3, capthick=0.8)
        )

        ax.bar(
            x + width/2, raw_m, width,
            yerr=raw_s,
            facecolor=COLOR_RAW_FILL,
            edgecolor=COLOR_RAW_EDGE,
            linewidth=1.2,
            label=f"{ylabel} (raw)",
            error_kw=dict(ecolor=COLOR_RAW_EDGE, elinewidth=0.8, capsize=3, capthick=0.8)
        )

        # kept ratio line (errorbar color = line)
        ax2.set_ylim(0, 1.05)
        ax2.errorbar(
            x, g["kept_ratio_emp_mean"],
            yerr=g["kept_ratio_emp_sem"],
            marker="o", color=COLOR_LINE,
            linewidth=1.8, capsize=3,
            ecolor=COLOR_LINE, elinewidth=0.8,
            label="Kept ratio"
        )

        ax.set_ylabel(ylabel)
        ax2.set_ylabel("Kept ratio")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Target kept ratio (%)")

        # ---- add significance stars (NOW includes MSE) ----
        yr = ax.get_ylim()[1] - ax.get_ylim()[0]
        y_offset = max(yr * 0.03, 0.01)
        for i, kr in enumerate(g["kept_ratio_target"]):
            y_top = max(
                kept_m.iloc[i] + (kept_s.iloc[i] if not np.isnan(kept_s.iloc[i]) else 0),
                raw_m.iloc[i]  + (raw_s.iloc[i]  if not np.isnan(raw_s.iloc[i])  else 0)
            )
            _add_star(ax, x[i], y_top, p_map.get(kr, np.nan), y_offset)

        # legend below
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(
            h1 + h2, l1 + l2,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
            ncol=3,
            frameon=False
        )

        fig.tight_layout()
        plt.savefig(f"{out_prefix}_{metric}.pdf", bbox_inches="tight")
        plt.savefig(f"{out_prefix}_{metric}.png", dpi=300, bbox_inches="tight")
        plt.show()

    _one("sp")
    _one("pe")
    _one("mse")

    return g


# Example usage:
summary = plot_bars_with_kept_ratio_line(df_res, out_prefix="mondrian_selective")
print(summary)



