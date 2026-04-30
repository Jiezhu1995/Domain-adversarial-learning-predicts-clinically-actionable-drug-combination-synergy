# Domain-adversarial-learning-predicts-clinically-actionable-drug-combination-synergy
This repository contains the implementation and data for a Domain-Adversarial Neural Network (DANN) designed to predict drug combination synergy in primary leukemia samples.

## Data Availability

All training and test datasets for DANN model and error prediction model are publicly available on Zenodo:

👉 https://zenodo.org/records/19749487

The dataset includes:
- Patient-level gene expression features, represented as ssGSEA pathway scores for AML bone marrow (BM)，AML peripheral blood (PB), and CLL PB samples
- Drug-level features, including LINCS L1000 transcriptional signatures and chemical fingerprints
- Drug combination synergy labels, including HSA synergy scores for AML BM, AML PB, and CLL samples

The original data sources used on this study are listed below: 

- **Patient drug combination screening data for AML and CLL** 
  Source information and data links are provided in:
  https://github.com/Jiezhu1995/Domain-adversarial-learning-predicts-clinically-actionable-drug-combination-synergy/tree/main/AML_CLL_patient_RNA-seq_data_source

- **Cell line transcriptomic responses (LINCS L1000)**
 Source information is provided in:  
  https://github.com/Jiezhu1995/Domain-adversarial-learning-predicts-clinically-actionable-drug-combination-synergy/tree/main/LINCS_cell_line_transcriptomic_responses_source  

- **Cell line drug combination data (DrugComb)** 
  Source information is provided in:  
  https://github.com/Jiezhu1995/Domain-adversarial-learning-predicts-clinically-actionable-drug-combination-synergy/tree/main/Cell_line_drug_combination_data_source  

## Usage

### 1. Data preparation
Download data from Zenodo:
https://zenodo.org/records/19749487

After downloading, place the files into the following directory structure:
data/
├── Cell line/
├── BM/
├── PB/
├── CLL/
Please ensure taht the file paths in the Python script match your local directory structure before running the code.
### 2. Model training
To train the DANN model, run:

python Beat_DANN_model.py
for the AML BeatAML dataset, and:

python CLL_DANN_model.py
for the CLL dataset.
These scripts include model training, domain-adversarial adaptation and plotting.
### 3. Error prediction model
To obtain the error prediction model, run:
python Error_prediction_model.py
for the AML BeatAML dataset and CLL dataset.

### 4. Ablation analysis
To perform ablation analysis, run:
python Ablation.py
This script evaluates the contribution of domain-adversarial learning by comparing the full DANN model with reduced model variants.

