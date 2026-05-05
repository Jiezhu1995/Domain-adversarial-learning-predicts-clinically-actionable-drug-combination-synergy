# Domain-Adversarial Learning Predicts Clinically Actionable Drug Combination Synergy

This repository provides the implementation and data resources for a Domain-Adversarial Neural Network (DANN) framework designed to predict drug combination synergy in primary leukemia samples.

---

## Data Availability

All processed datasets used for training and evaluation are publicly available on Zenodo:

👉 https://zenodo.org/records/19749487

The dataset includes:

* **Patient-level gene expression features (ssGSEA pathway scores derived from gene expression)**

  * AML bone marrow (BM)
  * AML peripheral blood (PB)
  * CLL peripheral blood (PB)

* **Drug-level features**

  * LINCS L1000 transcriptional signatures
  * MACCS chemical fingerprints

* **Drug combination synergy labels**

  * HSA scores for AML BM, AML PB, and CLL samples

---

## Repository Structure

```bash
data/
├── patient/
│   ├── AML/
│   │   ├── rna_seq.csv
│   │   └── drug_response.csv
│   ├── CLL/
│   │   ├── rna_seq.csv
│   │   └── drug_response.csv
│
├── cell_line/
│   ├── drug_response.csv
│   └── maccs_fingerprint.csv
│
├── L1000/
│   ├── signatures.csv
│   └── zscore_1006gene.csv

scripts/
```

---

## Data Sources

The datasets used in this study are organized in the `data/` directory as follows:

* **Patient data (`data/patient/`)**

  * AML and CLL patient datasets including:

    * bulk RNA-seq data (`rna_seq.csv`)
    * Drug combination response data (`drug_combination.csv`)

* **Cell line data (`data/cell_line/`)**

  * Drug combination response data (`drug_combination.csv`)
  * Chemical structure features represented as MACCS fingerprints (`maccs_fingerprint.csv`)

* **LINCS L1000 data (`data/L1000/`)**

  * Drug-induced transcriptional signatures (`signatures.csv`)
  * Processed gene-level features (z-scored expression of selected genes; `zscore_1006gene.csv`)

---

## Usage

### 1. Data Preparation

Download the dataset from Zenodo and place the files into the corresponding directories under `data/`.

Ensure that file paths in the scripts match your local directory structure before execution.

---

### 2. Model Training

For AML (BeatAML dataset):

```bash
python scripts/Beat_DANN_model.py
```

For CLL dataset:

```bash
python scripts/CLL_DANN_model.py
```

These scripts include:

* Model training
* Domain-adversarial adaptation
* Performance evaluation and plotting

---

### 3. Error Prediction Model

```bash
python scripts/Error_prediction_model.py
```

This script builds a post hoc model to estimate prediction uncertainty.

---

### 4. Ablation Analysis

```bash
python scripts/Ablation.py
```

This evaluates the contribution of domain-adversarial learning by comparing the full DANN model with reduced variants.

---

## Notes

* Ensure consistent preprocessing between training and test datasets
* All features are precomputed and provided in the Zenodo dataset
* The framework supports both AML and CLL datasets

---

## Contact

For questions or collaborations, please contact the repository owner.
