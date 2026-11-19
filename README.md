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

## Patient drug combination data

The patient tables provide experimentally derived HSA synergy scores and the corresponding gene expression matrices for reanalysis and prediction benchmarking. Each row in a `drug_combination.csv` file is one drug-pair measurement in one sample. These are calculated synergy-score tables; they do not contain the individual drug concentrations or viability measurements used to calculate the scores.

### Cohorts and direct downloads

| Cohort | Sample source | Samples | Drug-pair identities, ignoring order | HSA measurements | Download scores |
| --- | --- | ---: | ---: | ---: | --- |
| AML | Bone marrow (BM) | 72 | 38 | 1,724 | [AML CSV](https://raw.githubusercontent.com/Jiezhu1995/Domain-adversarial-learning-predicts-clinically-actionable-drug-combination-synergy/main/data/patient/AML/drug_combination.csv) |
| AML | Peripheral blood (PB) | 74 | 38 | 1,952 | [AML CSV](https://raw.githubusercontent.com/Jiezhu1995/Domain-adversarial-learning-predicts-clinically-actionable-drug-combination-synergy/main/data/patient/AML/drug_combination.csv) |
| CLL | Peripheral blood (PB) | 62 | 90 | 5,952 | [CLL CSV](https://raw.githubusercontent.com/Jiezhu1995/Domain-adversarial-learning-predicts-clinically-actionable-drug-combination-synergy/main/data/patient/CLL/drug_combination.csv) |

The AML download contains both BM and PB samples, distinguished by `SampleSource`. It includes the 38-pair subset used in the AML modeling analyses, rather than the full original screening library. The sample counts refer to specimens, not necessarily distinct patients.

The CLL score table is retained unchanged. It contains 96 ordered entries per sample, corresponding to 90 drug-pair identities when order is ignored. Reversed drug orientations and same-drug entries are retained as supplied. Reproducing the original distribution analyses therefore requires retaining these entries; any filtering or aggregation for a new analysis should be stated explicitly.

The expression matrices are available separately as [AML RNA-seq CSV](https://raw.githubusercontent.com/Jiezhu1995/Domain-adversarial-learning-predicts-clinically-actionable-drug-combination-synergy/main/data/patient/AML/rna_seq.csv) and [CLL RNA-seq CSV](https://raw.githubusercontent.com/Jiezhu1995/Domain-adversarial-learning-predicts-clinically-actionable-drug-combination-synergy/main/data/patient/CLL/rna_seq.csv). In both matrices, rows are genes, the first column (`ID`) contains gene identifiers, and the remaining column names are sample identifiers. **The AML expression file is semicolon-separated; the CLL expression file and both synergy tables are comma-separated.** The expression files are distinct from the ssGSEA pathway-feature matrices used by the models.

### Score-table columns

| Column | Meaning |
| --- | --- |
| `block_id` | Identifier of a drug-pair–sample measurement. For AML, this is the existing key shared by the HSA labels and processed feature matrices. It is not a patient identifier or a drug concentration. Use it within the relevant cohort. |
| `Row_drug` | Name of the first drug, preserving the original input orientation. |
| `Col_drug` | Name of the second drug, preserving the original input orientation. |
| `HSA_score` | Experimentally derived Highest Single Agent synergy score on the original percentage-point scale, without rescaling. Positive scores indicate additional inhibition relative to the more active single agent; zero indicates an equal effect and negative scores indicate less inhibition. These are measured synergy labels, not model predictions. |
| `SampleID` | Existing pseudonymous specimen identifier: `AML1`–`AML146` or `CLL1`–`CLL62`. This value matches a column name in the corresponding public expression matrix. |
| `SampleSource` | AML only: `BM` for bone marrow or `PB` for peripheral blood. All samples in the CLL table are PB. |

The original-to-public sample identifier mappings are not distributed. The existing pseudonyms are preserved; no sample identifiers have been reassigned. The AML table preserves the original `block_id`, drug orientation and HSA values, with BM measurements followed by PB measurements.

### Minimal loading example

Run the following from the repository root after downloading or cloning the repository. Only `pandas` is needed to load these tables; the processed model archives are not needed for this example.

```python
from pathlib import Path
import pandas as pd

data = Path("data/patient")
aml = pd.read_csv(data / "AML/drug_combination.csv", dtype={"block_id": str})
cll = pd.read_csv(data / "CLL/drug_combination.csv", dtype={"block_id": str})
aml_expression = pd.read_csv(data / "AML/rna_seq.csv", sep=";", index_col="ID")
cll_expression = pd.read_csv(data / "CLL/rna_seq.csv", index_col="ID")

assert set(aml["SampleID"]) == set(aml_expression.columns)
assert set(cll["SampleID"]) == set(cll_expression.columns)
assert aml["block_id"].is_unique and cll["block_id"].is_unique

aml_bm = aml.loc[aml["SampleSource"].eq("BM")].copy()
aml_pb = aml.loc[aml["SampleSource"].eq("PB")].copy()

# One expression profile per sample, in the order encountered in the score table.
aml_profiles = aml_expression.loc[:, aml["SampleID"].drop_duplicates()].T
cll_profiles = cll_expression.loc[:, cll["SampleID"].drop_duplicates()].T
print(aml.groupby("SampleSource").agg(
    samples=("SampleID", "nunique"), measurements=("block_id", "size")
))
```

### Matching AML scores to processed model inputs

Download the processed inputs from [Zenodo](https://doi.org/10.5281/zenodo.19749487). The AML score table uses the same `block_id` values as `HSA_label_BM.csv`, `HSA_label_PB.csv`, the corresponding `cellExp_ssGSEA_input_*` matrices, and the `rowDrug_*` and `colDrug_*` feature matrices. Join these files by `block_id` within BM or PB. `Row_drug` corresponds to `drug_row` in the row-drug matrices, and `Col_drug` corresponds to `drug_col` in the column-drug matrices. Match the public gene expression columns using `SampleID`; these pseudonyms also agree with the `SampleID` metadata in the released AML ssGSEA matrices.

For example, after placing the downloaded BM label file in `data/`, the following verifies that the public scores match the model labels. Adjust only the file path if the downloaded inputs are in a different directory.

```python
labels_bm = pd.read_csv("data/HSA_label_BM.csv", dtype={"block_id": str})
matched = aml_bm.merge(labels_bm, on="block_id", how="outer",
                       validate="one_to_one", indicator=True)
assert matched["_merge"].eq("both").all()
assert (matched["HSA_score"] - matched["synergy_hsa"]).abs().le(1e-12).all()
```

Use the identifiers for matching rather than assuming that separately downloaded files have the same row order. For sample-held-out benchmarks, keep all drug-pair measurements from a given `SampleID` in the same partition. The exact files needed to run each existing model script are listed in [required_inputs.json](required_inputs.json).

### Versions and citation

The AML score table was restored on 19 November 2026, retaining the existing public sample identifiers and model-input keys. The CLL score table and both public expression matrices were unchanged by this update. For reproducibility, record the repository URL and commit used to download these GitHub tables (`git rev-parse HEAD` in a cloned repository).

Please cite the associated study when reusing the patient data. When using the processed model inputs, also cite their versioned data record: Zhu, J. (2026). *Domain-adversarial learning predicts clinically actionable drug combination synergy using bulk transcriptomics data in leukemia* (v2). Zenodo. [doi:10.5281/zenodo.19749487](https://doi.org/10.5281/zenodo.19749487). The Zenodo record describes the processed model inputs; the restored score table is versioned separately in this GitHub repository. For questions about additional experimental measurements or data-use terms, contact the repository owner.

---

## Repository Structure

```bash
data/
├── patient/
│   ├── AML/
│   │   ├── rna_seq.csv
│   │   └── drug_combination.csv
│   ├── CLL/
│   │   ├── rna_seq.csv
│   │   └── drug_combination.csv
│
├── cell_line/
│   ├── drug_combination.csv
│   └── maccs_fingerprint.csv
│
├── L1000/
│   ├── signatures.csv
│   └── zscore_1006gene.csv

scripts/
├── BeatAML_DANN_model.py
├── CLL_DANN_model.py
├── Ablation.py
├── Error_prediction_model.py
└── data_paths.py
required_inputs.json
```

---

## Data Sources

The datasets used in this study are organized in the `data/` directory as follows:

* **Patient data (`data/patient/`)**

  * AML and CLL patient datasets including:

    * bulk RNA-seq data (`rna_seq.csv`)
    * Calculated drug combination HSA scores (`drug_combination.csv`; see the patient data descriptions above)

* **Cell line data (`data/cell_line/`)**

  * Drug combination response data (`drug_combination.csv`)
  * Chemical structure features represented as MACCS fingerprints (`maccs_fingerprint.csv`)

* **LINCS L1000 data (`data/L1000/`)**

  * Drug-induced transcriptional signatures (`signatures.csv`)
  * Processed gene-level features (z-scored expression of selected genes; `zscore_1006gene.csv`)

---

## Usage

### 1. Data Preparation

Download and extract the processed model inputs from Zenodo into `data/` beside this README. The source tables listed above are distinct from the processed matrices and saved models required by the scripts. See [required_inputs.json](required_inputs.json) for the exact filenames required by each entry point; preserve filenames when extracting archives.

The scripts resolve inputs relative to the repository's `data/` directory, independently of the working directory. Flat and nested layouts are supported. To use an existing data directory, set:

```bash
export DANN_DATA_DIR="/absolute/path/to/processed_inputs"
```

Keep one authoritative copy of each required filename under that directory. Missing files and duplicate filenames raise explicit errors; no script path edits are needed. The default directory is `data/` (lowercase).

Run the commands below from the repository root. Outputs retain their original filenames and are written relative to the working directory; use a separate output directory and an absolute script path to preserve earlier results.

---

### 2. Model Training

For AML (BeatAML dataset):

```bash
python scripts/BeatAML_DANN_model.py
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
## Environment

The code was tested with:

```text
Python 3.10.18

numpy==1.24.4
pandas==2.3.3
scikit-optimize==0.10.2
matplotlib==3.10.1
joblib==1.5.2
tensorflow==2.15.0
h5py==3.15.1
scikit-learn==1.3.2
xgboost==3.1.2
scipy==1.10.1
keras==2.15.0
concrete-autoencoder==0.0.1
```
---
## Code maintenance validation

The Chinese comments in the public Python scripts have been translated into English, and the AML entry point is consistently named `BeatAML_DANN_model.py`. Model definitions, training settings, random seeds, and statistical calculations are unchanged by these maintenance edits.

Syntax checks, comparison of executable operations apart from input-path resolution, and path-resolution checks have been performed. These checks do not constitute a full training or end-to-end reproduction run.

## Notes

* Ensure consistent preprocessing between training and test datasets
* All features are precomputed and provided in the Zenodo dataset
* The framework supports both AML and CLL datasets

---

## Contact

For questions or collaborations, please contact the repository owner.
