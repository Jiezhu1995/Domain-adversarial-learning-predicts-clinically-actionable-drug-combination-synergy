# Domain-adversarial-learning-predicts-clinically-actionable-drug-combination-synergy
This repository contains the implementation and data for a Domain-Adversarial Neural Network (DANN) designed to predict drug combination synergy in primary leukemia samples.

## Data Availability

All training and test datasets for DANN model and error precition model are publicly available on Zenodo:

👉 https://zenodo.org/records/19749487

The dataset includes:
- Gene expression profiles (ssGSEA features, including BM AML and PB patietns, and CLL patients)
- Drug features (LINCS L1000 + chemical fingerprints)
- Synergy score (HSA synerygy score including BM, PB and CLL)

For the raw data used on this study are avaiable from the following resources:
- Patient drug combination data (AML & CLL): AML_CLL_patient_RNA-seq_data_source    

- **Patient RNA-seq data (AML & CLL)** 
  Obtained from previously published studies: AML_CLL_patient_RNA-seq_data_source  

- **Cell line transcriptomic responses (LINCS L1000)**
  LINCS_cell_line_transcriptomic_responses_source  

- **Cell line drug combination data (DrugComb)** 
  Cell_line_drug_combination_data_source  


## Usage

### 1. Data preparation
Download data from Zenodo:
https://zenodo.org/records/19749487

Unzip and place the data into the following directory structure:
data/
├── Cell_line/
├── BM/
├── PB/
├── CLL/

### 2. Model training
Run the following scripts to train the DANN model and the error prediction model:

```bash
python Beat_DANN_error_prediction_model.py   # for AML (BeatAML dataset)
python CLL_DANN_error_prediction_model.py    # for CLL dataset

Ablation study
To perform ablation analysis, run:
python Ablation.py

