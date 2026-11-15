# Running Who Wrote It?

For the research background, architecture, and experimental results, see the [main README](README.md). The [online demo](https://who-wrote-it.streamlit.app/) can be used without installing anything.

## 1. Run the demo locally

Clone the repository and enter its root directory:

```bash
git clone https://github.com/amnonj/WhoWroteIt.git
cd WhoWroteIt
```

Use Python 3.12 and preferably a virtual environment. Install the application dependencies and start Streamlit:

```bash
python -m pip install -r streamlit_app/requirements.txt
python -m streamlit run streamlit_app/app.py
```

The selected trained model and its configuration are included in the repository. No training data or Project Ben-Yehuda API key is required to run the demo.

## 2. Research environment

From the repository root, install the research dependencies:

```bash
python -m pip install -r requirements-research.txt
```

The training and evaluation scripts use paths relative to the `poets/` directory and imports from the repository root. Set `PYTHONPATH` to the repository root before changing the working directory.

Windows (PowerShell):

```powershell
$env:PYTHONPATH = (Get-Location).Path
Set-Location poets
```

Linux/macOS (bash):

```bash
export PYTHONPATH="$PWD"
cd poets
```

The following research commands assume that `poets/` is the current working directory and that the research dependencies have been installed. The core package versions are those used by the published demo; versions of the additional research libraries are not fully pinned.

## 3. Obtain the poetry dataset

The original poetry corpus is not distributed with this repository. The collection script downloads metadata and accessible text files from the Project Ben-Yehuda API, subject to its applicable terms of use and the rights of the individual works.

Obtain an API key from [Project Ben-Yehuda](https://benyehuda.org/) and create a local file named `poets/api_secrets.py` containing:

```python
API_KEY = "YOUR_API_KEY"
```

Do not commit this file. It is excluded by `.gitignore`.

From `poets/`, run:

```bash
python collect_benyehuda_poetry.py
```

The script writes `poetry_data/poetry_index.csv` and `poetry_data/poetry_dataset.csv`, supports resuming interrupted downloads, and deliberately throttles its requests. A full download may take a considerable amount of time.

If you already have the original dataset CSV, place it at `poets/poetry_data/poetry_dataset.csv` instead.

To inspect the filtered dataset and the train/validation/test distribution:

```bash
python analyze_db.py
```

Filtering, duplicate removal, and stratified splitting are performed by `data.py`. Exact reproduction of the published data split requires the original corpus snapshot and ordering, as the source collection may change over time.

## 4. Train a model

Set the desired architecture and training parameters in `nn_config.py`, then run:

```bash
python main.py
```

The script trains one model for each seed in `nn_config.SEED`. To train using the seed of the published model, set `SEED = [200]` with the selected architecture configuration.

Each run saves its weights, configuration, and validation predictions in a timestamped subdirectory of `experiments/`, and appends a summary to `experiments/results.csv`.

## 5. Evaluate the published model

With the dataset available, run:

```bash
python evaluate.py
```

The default configuration evaluates the published model (`20261111-141613`) on the test set. Change `DATA_SET` in the script to `val` to evaluate the validation set instead. The script prints evaluation metrics and a confusion matrix and saves prediction arrays. Test evaluation writes `test.npz`; validation evaluation writes `validation_recheck.npz` and compares against the original saved validation predictions. If preserving the published files, back them up before rerunning evaluation on a different corpus snapshot.

## 6. Reproduce the ensemble analysis

The repository includes prediction files and configurations for the four selected ensemble members. Their trained weights are not all distributed. Ensemble analysis operates on the saved predictions and does not require downloading the poems or retraining the networks.

From `poets/`, run:

```bash
python ensemble.py
```

The default configuration combines all four members (`COMB = 15`) on the test set using mean logits (`SELECTION = 'MUL'`). For the validation results reported in the main README, use the prediction files produced with the deterministic cropping procedure (`DATA_SET = 'validation_recheck'`). These re-evaluated prediction files are distinct from the original `validation.npz` files.

The separate weight-optimization experiment can be run using:

```bash
python ensemble_weight_optimization.py
```

It uses the original validation predictions to compare uniform and optimized ensemble weights through five-fold cross-validation. With the published four-member `FOLDERS` list, the selected members are numbered `(1, 2, 3, 4)`.

The published predictions allow analysis of the selected ensemble. The original exhaustive search over 20 candidate models requires additional prediction files that are not included in the repository.

## 7. Repository structure

| Path | Purpose |
|---|---|
| `common/text_utils.py` | Shared character encoding and decoding utilities. |
| `poets/collect_benyehuda_poetry.py` | Download poetry metadata and accessible text files. |
| `poets/config.py` | Dataset selection and preprocessing parameters. |
| `poets/data.py` | Filter and clean poems, remove near-duplicates, and create stratified splits. |
| `poets/analyze_db.py` | Inspect dataset statistics and sampling regions. |
| `poets/datasets.py` | PyTorch dataset with character- and line-aware sampling and padding. |
| `poets/AuthorNet.py` | Configurable 1D, 2D, and dual-branch CNN architecture. |
| `poets/nn_config.py` | Architecture and training hyperparameters. |
| `poets/main.py` | Model training, validation, and experiment logging. |
| `poets/evaluate.py` | Evaluation of saved models on validation or test data. |
| `poets/ensemble.py` | Ensemble aggregation and error analysis. |
| `poets/ensemble_weight_optimization.py` | Ensemble weight optimization and cross-validation. |
| `poets/who_wrote_it.py` | Earlier local Tkinter inference interface. |
| `poets/experiments/` | Experiment summaries, published prediction arrays, configurations, and selected model weights. |
| `streamlit_app/` | Public web application and its dependencies. |
| `images/` | Network architecture diagram. |
| `politicians/` | [Earlier exploratory experiments](politicians/README.md) on political social media posts. |

The files in `politicians/` are retained for reference and are not required by the poetry classifier.
