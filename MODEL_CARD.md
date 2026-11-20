# Model Card: DANN

## Model and intended use

The domain-adversarial neural network (DANN) predicts drug combination synergy in acute myeloid leukemia (AML) and chronic lymphocytic leukemia (CLL) specimens. HSA is the main prediction endpoint, with separate analyses for Bliss and ZIP. The intended research use is to prioritize two-drug small-molecule combinations for experimental follow-up.

## Data and inputs

| Dataset | Coverage | Role |
| --- | --- | --- |
| DrugComb cell-line data | 9 blood cancer cell lines; 2,150 drug combinations | Pre-training |
| AML | 72 bone marrow (BM) and 74 peripheral blood (PB) specimens; 38 combinations | BM adaptation and held-out BM/PB evaluation |
| CLL | 62 PB specimens; 90 unique combinations | Separate adaptation and held-out CLL evaluation |

Specimen counts do not necessarily equal unique patient counts. Inputs combine MACCS chemical fingerprints and LINCS L1000 perturbation signatures for both drugs with specimen-level ssGSEA pathway profiles. Dataset descriptions, pseudonymous sample identifiers and matching instructions are provided in the [README](README.md#patient-drug-combination-data).

## Architecture and training

DANN combines a shared neural feature extractor, a synergy regression head and a domain classifier with a gradient-reversal layer. Cell-line pre-training is followed by separate adaptation to AML and CLL data. The released scripts specify Adam with a learning rate of 0.0005, 300 epochs and a batch size of 32. Full configurations are provided in the [AML](scripts/BeatAML_DANN_model.py) and [CLL](scripts/CLL_DANN_model.py) scripts.

## Evaluation and results

Main comparisons use five repeated specimen-level training/test splits and report Pearson correlation, Spearman correlation and mean squared error. AML PB specimens are held out from BM adaptation. CLL evaluation uses specimens excluded from CLL fine-tuning. The auxiliary [error prediction model](scripts/Error_prediction_model.py) uses three-fold cross-validation within calibration data and separate specimens for final evaluation.

The associated manuscript reports improved predictive performance relative to RF, XGBoost, KNN and linear regression on matched inputs and sample splits. In CLL, Pearson correlation is approximately 0.50 with 16-20 training specimens. Results across training sizes are reported in manuscript Figures 2 and 3. Patient-only and ssGSEA-shuffling ablations assess the contributions of the training framework and patient-specific features.

## Limitations

Performance depends on cohort composition, training size and experimental conditions. The evaluated cohorts are modest, and applicability beyond AML/CLL, two-drug small-molecule combinations or the represented assay settings requires further evaluation. Predicted synergy is distinct from absolute drug activity or clinical benefit. The error model supports ranking predictions by estimated reliability but does not provide calibrated probability guarantees.

## Access and reproducibility

Installation and execution instructions are in the [README](README.md#usage), and required input files are listed in [required_inputs.json](required_inputs.json). Processed inputs and trained models are available on [Zenodo](https://doi.org/10.5281/zenodo.19749487). Preserve the supplied feature order, preprocessing conventions and sample identifiers when reproducing analyses. Raw patient sequencing data remain subject to the source repositories' controlled-access conditions.
