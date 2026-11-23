# Car Insurance Claim Cost Prediction

Final project for the Machine Learning course.

The goal is to predict `annual_claim_cost`: the total cost of claims a driver will cost the insurance company during the policy year (0 if there was no claim, about 65% of drivers). The model uses only fields that are known **before the contract is signed**.

<!-- TODO (block 3): one or two sentences with the final CV results of both models -->

## Repository structure

| File | Description |
|---|---|
| `notebook.ipynb` | Main notebook: all final code, in the order of the assignment sections |
| `dataset.csv` | Training data: 45,000 drivers, 30 columns |
| `model.pkl` | Trained ElasticNet pipeline (used for the held-out test) <!-- TODO (block 4) --> |
| `requirements.txt` | Libraries and exact versions |
| `report.pdf` | Project report <!-- TODO (block 6) --> |
| `cleaning_driver.ipynb`, `daniel_prepare.ipynb`, `target_eda.ipynb` | Exploration notebooks (cleaning of columns A–M and N–X, target exploration) |
| `AI usage log.ipynb` | Log of AI consultations (also in the report appendix) |

## Setup

Python 3.10 or newer.

```bash
git clone <repo-url>
cd car-insurance-claim-prediction
pip install -r requirements.txt
```

The library versions in `requirements.txt` must be installed exactly as listed, because `model.pkl` can only be loaded with the same scikit-learn version it was saved with.

## How to run

Open `notebook.ipynb` and run all cells (**Restart & Run All**). The notebook reads `dataset.csv` from the repository root and runs end-to-end. All randomness uses `random_state=42`.

## Test-time usage

```python
import pandas as pd
import joblib

# prepare_data is defined in notebook.ipynb
model = joblib.load("model.pkl")
df_test = pd.read_csv("test.csv")
X = prepare_data(df_test)
y_pred = model.predict(X)
```

`prepare_data(df)` receives a DataFrame in the raw structure of `dataset.csv` (with or without the target column) and returns the 22 model columns in a fixed order. It applies fixed cleaning rules only and learns nothing from the data, so it works on a single row as well as on thousands. It was tested on 12 types of edge-case input (single rows, empty rows, dirty values, shuffled and extra columns, missing target).

<!-- TODO (block 2): update the column count after adding the engineered features -->

## Model

<!-- TODO (block 3): describe the final pipeline -->
- **ElasticNet** (submitted in `model.pkl`): a single scikit-learn Pipeline with imputation, scaling and one-hot encoding, tuned with 10-fold cross validation.
- **Random Forest**: for comparison and for the complexity curve only.

All preprocessing is fitted inside the Pipeline, on the training fold only.

## Data leakage

Five columns are created during or at the end of the policy year and are removed: `claim_count`, `at_fault`, `days_to_settlement`, `police_report_filed`, `current_premium`. Two identifier columns are also removed: `customer_id_hash`, `agent_code`. See section 2.2 in the notebook for the evidence.

## Authors

<!-- TODO: names -->
