# mcreweight

`mcreweight` is a Python package for Monte Carlo event reweighting to match data distributions in multiplicity and kinematic variables. It provides multiple reweighting backends, including `hep_ml`-based gradient boosting, XGBoost, neural-network approaches, folding variants, and bin-based reweighting, with optional Optuna hyperparameter tuning and integrated plotting/validation utilities.

> [!WARNING]
Bins reweighting works fine for one or two dimensional histograms, but it is unstable and inaccurate for higher dimenstions 

## Documentation

Project documentation is currently intended to be available at:

- CERN Pages: `http://mcreweight.docs.cern.ch`
- Read the Docs: `https://mcreweight.readthedocs.io/` once the project is imported and built there

## Setup

Run in a `lb-conda` environment, as
```bash
lb-conda mcreweight
```

## Installation

If you don't run in a `lb-conda` environment, consider installing the python package from `PyPI` or cloning it from `GitLab`.

### From PyPI

```bash
pip install mcreweight
```

### From GitLab

Requires [pixi](https://pixi.sh/latest/#installation).

```bash
git clone https://gitlab.cern.ch/lhcb-dpa/tools/mcreweight.git
cd mcreweight
```

To run the CLI tools you can prefix them with `pixi run`, i.e.

```bash
pixi run run-reweight --help
pixi run apply-weights --help
```

To run the verification checks used in CI:

```bash
pixi run -e lint quality
pixi run test
pixi run -e docs build-docs
```

To auto-format the code before rerunning the checks:

```bash
pixi run -e lint black .
pixi run -e lint quality
```

This repository currently uses these Pixi environments:

- `default`: package runtime and CLI usage
- `lint`: `black` and `ruff`
- `test`: `pytest`
- `docs`: Sphinx documentation build

Useful Pixi commands:

```bash
pixi run -e lint ruff check .
pixi run -e lint black --check .
pixi run -e docs sphinx-build -b html -n -W --keep-going docs docs/_build/html
pixi run -e test pytest tests/test_cli.py::test_run_reweight -v
pixi run -e test pytest tests/test_cli.py::test_apply_weights -v
```

The `pixi.lock` file pins all dependencies for reproducibility. To update them, run `pixi update` and commit the updated lock file.

## Usage

Both CLIs support two usage modes:

- Config-driven mode (recommended): pass `--config <file.yaml>`
- Direct CLI mode: pass all options on the command line

Use `--dry-run` to validate the merged configuration without running.

To run reweighting do it via configuration file:

```bash
run-reweight --config <path_to_config.yaml>
```
 
or passing all options to the command line:

```bash
run-reweight --path-data <path_to_data.root> \
             --path-mc <path_to_mc.root> \
             --training-vars <variable_list> \
             --monitoring-vars <monitoring_variable_list> \
             --sample <sample> \
             --n_trials <optuna_tests> \
             --test_size <test_sample_size> \
             --weightsdir <weights_directory>
```

To apply saved weights to an MC sample with configuration file:
 
```bash
apply-weights --config <path_to_config.yaml>
```

or passing all options to the command line:

```bash
apply-weights --path-mc <path_to_mc.root> \
              --vars <variable_list> \
              --training-sample <training_sample> \
              --application-sample <application_sample> \
              --method <method_for_reweighter> \
              --monitoring-vars <monitoring_variable_list> \
              --output-path <output_file.root> \
              --weightsdir <weights_directory>
```

### Options

#### For the reweighting (`run-reweight`):

General:
- `--config`: YAML configuration file
- `--dry-run`: Validate config and print effective settings only
- `--verbosity`: Logging level (`1`, `2`, `3`, `4`), default `1`

Inputs:
- `--path-mc`: MC ROOT file path(s)
- `--tree-mc`: MC tree name (default: `DecayTree`)
- `--mcweights-name`: MC input weight branch
- `--mc-label`: MC sample label (default: `MC`)
- `--path-data`: Data ROOT file path(s)
- `--tree-data`: Data tree name (default: `DecayTree`)
- `--sweights-name`: Data sWeights branch (default: `sweight_sig`)
- `--data-label`: Data sample label (default: `Data`)
- `--path-xlabels`: YAML mapping for axis labels (default: `utils/default_xlabels.yaml`)

Variables:
- `--training-vars`: Reweighting training variables
- `--monitoring-vars`: Variables used for monitoring plots

Reweighting:
- `--sample`: Sample tag (default: `bd_jpsikst_ee`)
- `--methods`: Methods list (defaults from YAML or internal defaults)
- `--transform`: Feature transform (`quantile`, `yeo-johnson`, `signed-log`, `scaler`)
- `--n_trials`: Optuna trials (default: `10`)
- `--test_size`: Test split fraction (default: `0.3`)
- `--n_folds`: Folding count (default: `10`)
- `--n_bins`: Bin count for `Bins` method (default: `10`)
- `--n_neighs`: Neighbors for `Bins` method (default: `3`)
- `--reweight-validation-fraction`: Validation split used by iterative ONNX reweighters
- `--reweight-early-stopping-rounds`: Early-stopping patience for iterative ONNX reweighters
- `--reweight-metric-every`: Evaluate ONNX validation metric every N stages
- `--folding-aggregation`: Folding aggregation (`weighted_geometric`, `geometric`, `median`)
- `--shap`: Enable SHAP computation

Output:
- `--weightsdir`: Weights root directory; if omitted, `MCREWEIGHTS_DATA_ROOT` is used
- `--plotdir`: Plot output directory (default: `plots`)

Additional options can be found by running:
```bash
run-reweight --help
```

#### For the application of the weights (`apply-weights`):

General:
- `--config`: YAML configuration file
- `--dry-run`: Validate config and print effective settings only
- `--verbosity`: Logging level (`1`, `2`, `3`, `4`), default `1`

Inputs:
- `--path-mc`: MC file path(s)
- `--tree-mc`: MC tree name (default: `DecayTree`)
- `--mcweights-name`: Existing MC input weight branch
- `--path-data`: Optional data file path(s), for comparison plots
- `--tree-data`: Data tree name (default: `DecayTree`)
- `--sweights-name`: Data sWeights branch (default: `sweight_sig`)
- `--path-xlabels`: YAML mapping for axis labels

Variables:
- `--vars`: Application variables
- `--training-vars`: Variables used when training the saved model
- `--monitoring-vars`: Variables used for monitoring plots

Reweighting:
- `--method`: Method (`GB`, `XGB`, `NN`, `Folding`, `XGBFolding`, `NNFolding`, `Bins`), default `XGB`
- `--training-sample`: Training sample tag (default: `bd_jpsikst_ee`)
- `--application-sample`: Application sample tag (default: `bd_jpsikst_ee`)
- `--weightsdir`: Weights root directory; if omitted, `MCREWEIGHTS_DATA_ROOT` is used
- `--plotdir`: Plot output directory (default: `plots`)

Output:
- `--output-path`: Output ROOT file path
- `--output-ntuple`: Output ntuple type (default: `TTree`)
- `--output-tree`: Output tree name (default: `DecayTree`)
- `--weights-name`: Name of produced weights branch (default: `weights`)

Additional options can be found by running:
```bash
apply-weights --help
```

## Example

Reweighting:
```bash
pixi run run-reweight \
  --config tests_run/run_reweighting_config.yaml \
  --verbosity 2
```

Application of the weights:
```bash
pixi run apply-weights \
  --config tests_run/apply_weights_config.yaml \
  --method XGB \
  --output-path test_applied_weights.root
```

Documentation build:
```bash
pixi run -e docs build-docs
```

The generated HTML documentation is written to `docs/_build/html/`.

## Contact

For questions, please contact the repository maintainer.
