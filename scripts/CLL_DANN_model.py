#!/usr/bin/env python
# coding: utf-8

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import skopt
import joblib
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
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
from keras.datasets import mnist
from keras.layers import Dense, Dropout, LeakyReLU, Softmax
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
print (X_train_scaled.shape[1])
rowDrug_MACCS_patient_CLL = pd.read_csv("/data/CLL/rowDrug_MACCS_input.csv", sep = ",")
rowDrug_Sig_patient_CLL = pd.read_csv("/data/CLL/rowDrug_Sig_input.csv", sep = ",")
colDrug_MACCS_patient_CLL = pd.read_csv("/data/CLL/colDrug_MACCS_input.csv", sep = ",")
colDrug_Sig_patient_CLL = pd.read_csv("/data/CLL/colDrug_Sig_input.csv", sep = ",")
CellExp_patient_CLL = pd.read_csv("/data/CLL/cellExp_input.csv", sep = ",")
HSA_label_patient_CLL = pd.read_csv("/data/CLL/HSA_label.csv", sep = ",")

input__patient_CLL=pd.concat([rowDrug_MACCS_patient_CLL,rowDrug_Sig_patient_CLL,colDrug_MACCS_patient_CLL,colDrug_Sig_patient_CLL,CellExp_patient_CLL,HSA_label_patient_CLL],axis=1,join="inner") 
input_patient_CLL_final=input__patient_CLL.drop(labels = ['drug_row',"block_id","drug_col"], axis=1)
new_data_patient_CLL=input_patient_CLL_final
df=input__patient_CLL

input__patient_CLL=pd.concat([rowDrug_MACCS_patient_CLL,rowDrug_Sig_patient_CLL,colDrug_MACCS_patient_CLL,colDrug_Sig_patient_CLL,CellExp_patient_CLL,HSA_label_patient_CLL],axis=1,join="inner") 

df=input__patient_CLL


# # DANN model
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
        Dense(32, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3, seed=seeds),

        # 2th hidden layer (new)
        Dense(16, activation='relu'),
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
        Dense(64, activation='relu', input_shape=input_shape,
              kernel_regularizer=regularizers.l2(1e-4)),
        BatchNormalization(),
        Dropout(0.2, seed=seeds),
        Dense(8, activation='relu',
              kernel_regularizer=regularizers.l2(1e-4)),
        BatchNormalization(),
        Dropout(0.2, seed=seeds),
        Dense(1, activation='linear')
    ])

# Remove the last layer of the cell line model to create a feature extractor
feature_extractor = Model(inputs=cell_line_model.input, outputs=cell_line_model.layers[-1].output)
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
        loss_weights=[1.0, 0.0]   # 初始时 domain loss 权重 = 0
    )
    return model


# Create and compile the model
transfer_model = build_and_compile_model()

# Save the initial weights
initial_weights = transfer_model.get_weights()

from tensorflow.keras.callbacks import Callback

class DomainWeightScheduler(Callback):
    def __init__(self, max_lambda=0.1, n_epochs=300):
        super().__init__()
        self.max_lambda = max_lambda
        self.n_epochs = n_epochs

    def on_epoch_begin(self, epoch, logs=None):
        # p 从 0 到 1 逐渐变化
        p = epoch / float(self.n_epochs)
        # DANN 论文里常用的 lambda 调度公式
        lam = self.max_lambda * (2. / (1. + np.exp(-10 * p)) - 1)
        # 动态修改模型的 loss_weights： [task_loss_weight, domain_loss_weight]
        self.model.loss_weights = [1.0, lam]

samples = [1,2, 3, 4, 5, 6, 7, 8, 9, 10,12,14,16,18,20,30,40,50]
repeat = 5


r2_scores = {'xgb_train': [],'xgb_CLL': [], 'rf_train': [],'rf_CLL': [], 'knn_train': [],'knn_CLL': [], 'lr_train': [],'lr_CLL': [],'DANN_train': [],'DANN_CLL': []}  # Add method of machine learning here
corr_coeffs = {'xgb_train': [],'xgb_CLL': [], 'rf_train': [],'rf_CLL': [],'knn_train': [],'knn_CLL': [], 'lr_train': [],'lr_CLL': [],'DANN_train': [],'DANN_CLL': []}
mse_scores = {'xgb_train': [],'xgb_CLL': [], 'rf_train': [],'rf_CLL': [], 'knn_train': [],'knn_CLL': [], 'lr_train': [],'lr_CLL': [],'DANN_train': [],'DANN_CLL': []}
avg_r2_scores = {'xgb_train': {},'xgb_CLL': {},'rf_train': {},'rf_CLL': {}, 'knn_train': {},'knn_CLL': {}, 'lr_train': {},'lr_CLL': {}, 'DANN_train': {},'DANN_CLL': {}}
all_r2_scores = {'xgb_train': {},'xgb_CLL': {},'rf_train': {},'rf_CLL': {}, 'knn_train': {},'knn_CLL': {}, 'lr_train': {},'lr_CLL': {}, 'DANN_train': {},'DANN_CLL': {}}
all_corr_coeffs = {'xgb_train': {},'xgb_CLL': {},'rf_train': {},'rf_CLL': {}, 'knn_train': {},'knn_CLL': {}, 'lr_train': {},'lr_CLL': {}, 'DANN_train': {},'DANN_CLL': {}}
all_mse_scores = {'xgb_train': {},'xgb_CLL': {},'rf_train': {},'rf_CLL': {}, 'knn_train': {},'knn_CLL': {}, 'lr_train': {},'lr_CLL': {}, 'DANN_train': {},'DANN_CLL': {}}
avg_corr_coeffs = {'xgb_train': {},'xgb_CLL': {},'rf_train': {},'rf_CLL': {}, 'knn_train': {},'knn_CLL': {}, 'lr_train': {},'lr_CLL': {}, 'DANN_train': {},'DANN_CLL': {}}
avg_mse_scores = {'xgb_train': {},'xgb_CLL': {},'rf_train': {},'rf_CLL': {}, 'knn_train': {},'knn_CLL': {}, 'lr_train': {},'lr_CLL': {}, 'DANN_train': {},'DANN_CLL': {}}
pearson_coeffs = {'xgb_train': [],'xgb_CLL': [], 'rf_train': [],'rf_CLL': [], 'knn_train': [],'knn_CLL': [], 'lr_train': [],'lr_CLL': [],'DANN_train': [],'DANN_CLL': []}
all_pearson_coeffs = {'xgb_train': {},'xgb_CLL': {},'rf_train': {},'rf_CLL': {}, 'knn_train': {},'knn_CLL': {}, 'lr_train': {},'lr_CLL': {}, 'DANN_train': {},'DANN_CLL': {}}
avg_pearson_coeffs = {'xgb_train': {},'xgb_CLL': {},'rf_train': {},'rf_CLL': {}, 'knn_train': {},'knn_CLL': {}, 'lr_train': {},'lr_CLL': {}, 'DANN_train': {},'DANN_CLL': {}}
lower_ci_pearson_coeffs = {'xgb_train': {},'xgb_CLL': {},'rf_train': {},'rf_CLL': {}, 'knn_train': {},'knn_CLL': {}, 'lr_train': {},'lr_CLL': {}, 'DANN_train': {},'DANN_CLL': {}}
upper_ci_pearson_coeffs = {'xgb_train': {},'xgb_CLL': {},'rf_train': {},'rf_CLL': {}, 'knn_train': {},'knn_CLL': {}, 'lr_train': {},'lr_CLL': {}, 'DANN_train': {},'DANN_CLL': {}}
# Add these dictionaries along with your other dictionaries
lower_ci_corr_coeffs = {'xgb_train': {},'xgb_CLL': {},'rf_train': {},'rf_CLL': {}, 'knn_train': {},'knn_CLL': {}, 'lr_train': {},'lr_CLL': {}, 'DANN_train': {},'DANN_CLL': {}}
upper_ci_corr_coeffs = {'xgb_train': {},'xgb_CLL': {},'rf_train': {},'rf_CLL': {}, 'knn_train': {},'knn_CLL': {}, 'lr_train': {},'lr_CLL': {}, 'DANN_train': {},'DANN_CLL': {}}
# Create a dictionary to store selected samples in each iteration
selected_samples = {'iteration': [], 'samples': []}
# Create a dictionary to store the unique combinations of samples
unique_combinations = {}
### divide the all patient data into training set and test set, and the sample of test set is not in the training set. 


id_col = "SampleID"
label_col = "HSA_score"

train_data = pd.read_csv("./Data/CLL/training_20.csv")
test_data  = pd.read_csv("./Data/CLL/test_another.csv")

# 合并成一个总表（所有样本）
all_data = pd.concat([train_data, test_data], ignore_index=True)

# 所有可抽样的 SampleID
all_sample_ids = all_data[id_col].unique()

print("Total unique samples:", len(all_sample_ids))


from sklearn.linear_model import Ridge
import numpy as np


all_data = pd.concat([train_data, test_data], ignore_index=True)

# Keep variable name as you requested: train_sample_ids
train_sample_ids = all_data['SampleID'].unique()

# ---------------------------
# (B) Main loop
# ---------------------------
for n_samples in samples:
    print(f"Number of samples: {n_samples}")

    # Initialize a set to store the unique combinations for the current number of samples
    unique_combinations[n_samples] = set()

    for i in range(repeat):
        print("-" * 60)
        print(f"[INFO] n_samples = {n_samples}, iteration = {i+1}/{repeat}")

        # Get the unique sample IDs and select the required number of samples
        while True:
            selected_sample_ids = tuple(sorted(np.random.choice(train_sample_ids, n_samples, replace=False)))

            # Check if the combination has already been used, and continue generating new combinations if it has
            if selected_sample_ids not in unique_combinations[n_samples]:
                break

        # Add the new combination to the set of unique combinations for the current number of samples
        unique_combinations[n_samples].add(selected_sample_ids)

        # Store the selected samples for this iteration
        iter_name = f"{n_samples}_samples_repeat_{i+1}"
        selected_samples['iteration'].append(iter_name)
        selected_samples['samples'].append(selected_sample_ids)

        # ============================================================
        # MODIFIED PART: training subset from ALL pool, test = remaining
        # ============================================================
        subset = all_data[all_data['SampleID'].isin(selected_sample_ids)].copy()

        remaining_sample_ids = [sid for sid in train_sample_ids if sid not in selected_sample_ids]
        test_subset = all_data[all_data['SampleID'].isin(remaining_sample_ids)].copy()

        # Optional guard for tiny test set
        if test_subset.shape[0] < 2:
            print(f"[WARN] ({iter_name}) Test set too small (n={test_subset.shape[0]}). Skipping.")
            continue

        # ---- patient TRAIN ----
        X_BM = subset.drop(labels=["SampleID", "HSA_score"], axis=1).values
        X_patient_BM_scaled = scaler.transform(X_BM)
        y = subset["HSA_score"].values

        # ---- patient TEST (use the SAME variable names as your original code) ----
        X_test_BM_patient = test_subset.drop(labels=["SampleID", "HSA_score"], axis=1).values
        X_test_BM = scaler.transform(X_test_BM_patient)
        y_test_BM = test_subset["HSA_score"].values

        # ============================================================
        # Keep your original logic for cell-line sampling for DANN
        # ============================================================
        num_rows = subset.shape[0]
        train_size = num_rows / (6829)

        X_train_use, X_test_no, y_train_use, y_test_no = train_test_split(
            X_train_scaled, y_train,
            train_size=train_size,
            random_state=42
        )

        X_combined = np.concatenate([X_patient_BM_scaled, X_train_use])
        y_combined = np.concatenate([y, y_train_use])

        domain_labels_patient = np.zeros(len(X_patient_BM_scaled))
        domain_labels_cell_line = np.ones(len(X_train_use))
        domain_labels = np.concatenate([domain_labels_patient, domain_labels_cell_line])

        # Split the combined data into training and testing sets
        #X_train_combined, X_test_combined, y_train_combined, y_test_combined, domain_labels_train, domain_labels_test = train_test_split(X_combined, y_combined, domain_labels, test_size=0.2, random_state=42)

        # Train and evaluate the XGBoostRegressor
        xgb = XGBRegressor(random_state=42,max_depth=5,n_estimators=50)
        xgb.fit(X_patient_BM_scaled, y)
        y_pred_xgb = xgb.predict(X_test_BM)
        r2_xgb = r2_score(y_test_BM, y_pred_xgb)
        r2_scores['xgb_CLL'].append(r2_xgb)
        spearman_corr_xgb, _ = spearmanr(y_test_BM, y_pred_xgb)
        corr_coeffs['xgb_CLL'].append(spearman_corr_xgb)
        mse_xgb = mean_squared_error(y_test_BM, y_pred_xgb)
        mse_scores['xgb_CLL'].append(mse_xgb)
        pearson_corr_xgb, _ = pearsonr(y_test_BM, y_pred_xgb)
        pearson_coeffs['xgb_CLL'].append(pearson_corr_xgb)
        
        
        y_pred_xgb_train = xgb.predict(X_patient_BM_scaled)
        r2_xgb_train = r2_score(y, y_pred_xgb_train)
        r2_scores['xgb_train'].append(r2_xgb_train)
        spearman_corr_xgb_train, _ = spearmanr(y, y_pred_xgb_train)
        corr_coeffs['xgb_train'].append(spearman_corr_xgb_train)
        mse_xgb_train = mean_squared_error(y, y_pred_xgb_train)
        mse_scores['xgb_train'].append(mse_xgb_train)
        pearson_corr_xgb_train, _ = pearsonr(y, y_pred_xgb_train)
        pearson_coeffs['xgb_train'].append(pearson_corr_xgb_train)
        print(f"[INFO] ({iter_name}) XGB done. R2_BM={r2_xgb:.3f}, mse={mse_xgb:.3f}")
        
        # Train and evaluate the RandomForestRegressor
        rf = RandomForestRegressor(random_state=42,max_depth=3,n_estimators = 30)
        rf.fit(X_patient_BM_scaled, y)
        y_pred_rf = rf.predict(X_test_BM)
        r2_rf = r2_score(y_test_BM, y_pred_rf)
        r2_scores['rf_CLL'].append(r2_rf)
        spearman_corr_rf, _ = spearmanr(y_test_BM, y_pred_rf)
        corr_coeffs['rf_CLL'].append(spearman_corr_rf)
        mse_rf = mean_squared_error(y_test_BM, y_pred_rf)
        mse_scores['rf_CLL'].append(mse_rf)
        pearson_corr_rf, _ = pearsonr(y_test_BM, y_pred_rf)
        pearson_coeffs['rf_CLL'].append(pearson_corr_rf)
    
        y_pred_rf_train = rf.predict(X_patient_BM_scaled)
        r2_rf_train = r2_score(y, y_pred_rf_train)
        r2_scores['rf_train'].append(r2_rf_train)
        spearman_corr_rf_train, _ = spearmanr(y, y_pred_rf_train)
        corr_coeffs['rf_train'].append(spearman_corr_rf_train)
        mse_rf_train = mean_squared_error(y, y_pred_rf_train)
        mse_scores['rf_train'].append(mse_rf_train)
        pearson_corr_rf_train, _ = pearsonr(y, y_pred_rf_train)
        pearson_coeffs['rf_train'].append(pearson_corr_rf_train)
        print(f"[INFO] ({iter_name}) RF done. R2_BM={r2_rf:.3f}, mse={mse_rf:.3f}")

    
        n_train_samples = X_patient_BM_scaled.shape[0]
        # Train and evaluate the KNN
        # 动态决定 n_neighbors，最多 10，且必须 <= 训练样本数
        if n_train_samples >= 2:
            n_neighbors = min(10, n_train_samples - 1)
        else:
        # 只有 1 个样本的极端情况，KNN 退化成 n_neighbors=1
            n_neighbors = 1
        knn = KNeighborsRegressor(n_neighbors = n_neighbors)
        knn.fit(X_patient_BM_scaled, y)                                                                                                                                
        y_pred_knn = knn.predict(X_test_BM)
        r2_knn = r2_score(y_test_BM, y_pred_knn)
        r2_scores['knn_CLL'].append(r2_knn)
        spearman_corr_knn, _ = spearmanr(y_test_BM, y_pred_knn)
        corr_coeffs['knn_CLL'].append(spearman_corr_knn)
        mse_knn = mean_squared_error(y_test_BM, y_pred_knn)
        mse_scores['knn_CLL'].append(mse_knn)
        pearson_corr_knn, _ = pearsonr(y_test_BM, y_pred_knn)
        pearson_coeffs['knn_CLL'].append(pearson_corr_knn)
        
        y_pred_knn_train = knn.predict(X_patient_BM_scaled)
        r2_knn_train = r2_score(y, y_pred_knn_train)
        r2_scores['knn_train'].append(r2_knn_train)
        spearman_corr_knn_train, _ = spearmanr(y, y_pred_knn_train)
        corr_coeffs['knn_train'].append(spearman_corr_knn_train)
        mse_knn_train = mean_squared_error(y, y_pred_knn_train)
        mse_scores['knn_train'].append(mse_knn_train)
        pearson_corr_knn_train, _ = pearsonr(y, y_pred_knn_train)
        pearson_coeffs['knn_train'].append(pearson_corr_knn_train)
        print(f"[INFO] ({iter_name}) KNN done. R2_BM={r2_knn:.3f}, mse={mse_knn:.3f}")
        
        
        # Train and evaluate the line regression                                                                                                                                
        #alphas = np.logspace(3, 9, 30)  # 1e-3 ... 1e3

        lr = Ridge(alpha=1e4, fit_intercept=True, random_state=42)
        lr.fit(X_patient_BM_scaled, y)
        y_pred_lr = lr.predict(X_test_BM)
        r2_lr = r2_score(y_test_BM, y_pred_lr)
        r2_scores['lr_CLL'].append(r2_lr)
        spearman_corr_lr, _ = spearmanr(y_test_BM, y_pred_lr)
        corr_coeffs['lr_CLL'].append(spearman_corr_lr)
        mse_lr = mean_squared_error(y_test_BM, y_pred_lr)
        mse_scores['lr_CLL'].append(mse_lr)
        pearson_corr_lr, _ = pearsonr(y_test_BM, y_pred_lr)
        pearson_coeffs['lr_CLL'].append(pearson_corr_lr)
        
        
        y_pred_lr_train = lr.predict(X_patient_BM_scaled)
        r2_lr_train = r2_score(y, y_pred_lr_train)
        r2_scores['lr_train'].append(r2_lr_train)
        spearman_corr_lr_train, _ = spearmanr(y, y_pred_lr_train)
        corr_coeffs['lr_train'].append(spearman_corr_lr_train)
        mse_lr_train = mean_squared_error(y, y_pred_lr_train)
        mse_scores['lr_train'].append(mse_lr_train)
        pearson_corr_lr_train, _ = pearsonr(y, y_pred_lr_train)
        pearson_coeffs['lr_train'].append(pearson_corr_lr_train)
        print(f"[INFO] ({iter_name}) LR done. R2_BM={r2_lr:.3f}, mse={mse_lr:.3f}")

        
        # ---------------- DANN ----------------
        print(f"[INFO] ({iter_name}) Training DANN model...")
        DANN = build_and_compile_model()
        DANN.set_weights(initial_weights)


        scheduler = DomainWeightScheduler(max_lambda=0.08, n_epochs=300)

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

        
        r2_DANN = r2_score(y_test_BM, y_pred_DANN)
        r2_scores['DANN_CLL'].append(r2_DANN)
        spearman_corr_DANN, _ = spearmanr(y_test_BM, y_pred_DANN)
        corr_coeffs['DANN_CLL'].append(spearman_corr_DANN)
        mse_DANN = mean_squared_error(y_test_BM, y_pred_DANN)
        mse_scores['DANN_CLL'].append(mse_DANN)
        pearson_corr_DANN, _ = pearsonr(y_test_BM, y_pred_DANN)
        pearson_coeffs['DANN_CLL'].append(pearson_corr_DANN)
        
        DANN_output_train  = DANN.predict(X_combined, verbose=0)
        y_pred_DANN_train = np.asarray(DANN_output_train[0], dtype=float).reshape(-1)
        y_combined_1d = np.asarray(y_combined, dtype=float).reshape(-1)  
        r2_DANN_train = r2_score(y_combined, y_pred_DANN_train)
        r2_scores['DANN_train'].append(r2_DANN_train)
        spearman_corr_DANN_train, _ = spearmanr(y_combined, y_pred_DANN_train)
        corr_coeffs['DANN_train'].append(spearman_corr_DANN_train)
        mse_DANN_train = mean_squared_error(y_combined, y_pred_DANN_train)
        mse_scores['DANN_train'].append(mse_DANN_train)
        pearson_corr_DANN_train, _ = pearsonr(y_combined, y_pred_DANN_train)
        pearson_coeffs['DANN_train'].append(pearson_corr_DANN_train)
        
        print(f"[INFO] ({iter_name}) DANN done. R2_BM={r2_DANN:.3f}, mse={mse_DANN:.3f}")

        
        
        for model in r2_scores.keys():
            if n_samples not in all_r2_scores[model]:
                all_r2_scores[model][n_samples] = []
            all_r2_scores[model][n_samples].append(r2_scores[model][-1])  # Store the last R2 score
            
            if n_samples not in all_corr_coeffs[model]:
                all_corr_coeffs[model][n_samples] = []
            all_corr_coeffs[model][n_samples].append(corr_coeffs[model][-1])
            
            if n_samples not in all_mse_scores[model]:
                 all_mse_scores[model][n_samples] = []
            all_mse_scores[model][n_samples].append(mse_scores[model][-1])
            
            
        # Store the last Pearson correlation for each model
        for model in pearson_coeffs.keys():
            if n_samples not in all_pearson_coeffs[model]:
                all_pearson_coeffs[model][n_samples] = []
            all_pearson_coeffs[model][n_samples].append(pearson_coeffs[model][-1])

                                                                                                                                         
        # Calculate average R2 scores for each model in each loop
    for model, scores in r2_scores.items():
        avg_r2 = np.mean(scores)
        avg_corr_coeff = np.mean(corr_coeffs[model])
        lower_ci, upper_ci = np.percentile(scores, [2.5, 97.5])
        avg_mse = np.mean(mse_scores[model])
        
        if model not in lower_ci_corr_coeffs:
            lower_ci_corr_coeffs[model] = {}
        if model not in upper_ci_corr_coeffs:
            upper_ci_corr_coeffs[model] = {}
            
        avg_r2_scores[model][n_samples] = avg_r2
        avg_corr_coeffs[model][n_samples] = avg_corr_coeff
        lower_ci_corr_coeffs[model][n_samples] = lower_ci
        upper_ci_corr_coeffs[model][n_samples] = upper_ci
        avg_mse_scores[model][n_samples] = avg_mse
        print(f"Average R2 score for {model} with {n_samples} samples: {avg_r2:.2f}")
        print(f"Average Spearman correlation for {model} with {n_samples} samples: {avg_corr_coeff:.2f}")
    # Calculate average Pearson correlation for each model
    for model, scores in pearson_coeffs.items():
        avg_pearson = np.mean(scores)
        lower_ci, upper_ci = np.percentile(scores, [2.5, 97.5])
    
        avg_pearson_coeffs[model][n_samples] = avg_pearson
        lower_ci_pearson_coeffs[model][n_samples] = lower_ci
        upper_ci_pearson_coeffs[model][n_samples] = upper_ci
        print(f"Average Pearson correlation for {model} with {n_samples} samples: {avg_pearson:.2f}")
                                                                                                                                           
    # Clear the R2 scores for the next iteration
    for model in r2_scores:
        r2_scores[model] = []
        corr_coeffs[model] = []
        mse_scores[model] = []    
    for model in pearson_coeffs:
        pearson_coeffs[model] = []
# Plot the average correlation coefficients for BM
plt.figure()
for model, scores in avg_corr_coeffs.items():
    if '_CLL' in model:
        x = list(scores.keys())
        y = list(scores.values())
        plt.plot(x, y, label=model)

plt.xlabel('Number of samples')
plt.ylabel('Average Correlation Coefficient (CLL)')
plt.legend()

# Save the plot as a high-quality PDF
plt.savefig('avg_corr_coeff_plot_CLL_5times_blood.pdf', dpi=900, format='pdf')
plt.show()

##save the model
DANN.save("DANN_50_samples_CLL.keras")
##save the summry files
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
write_all_results_to_csv('all_r2_scores_5times_CLL_blood.csv', all_r2_scores)
write_all_results_to_csv('all_corr_coeffs_5times_CLL_blood.csv', all_corr_coeffs)
write_all_results_to_csv('all_mse_scores_5times_CLL_blood.csv', all_mse_scores)
write_all_results_to_csv('all_pearson_coeffs_5times_CLL_blood.csv', all_pearson_coeffs)

