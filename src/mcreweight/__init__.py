"""mcreweight: MC event reweighting based on multiplicity and kinematic variables."""

from importlib.metadata import version

__version__ = version("mcreweight")

from mcreweight.config import X_LABELS
from mcreweight.core import apply_weights_pipeline, run_reweighting_pipeline
from mcreweight.io import def_aliases, load_data, save_data
from mcreweight.optuna import optuna_tuning
from mcreweight.train import (
    binning_reweight,
    gbfolding,
    gbreweight,
    nnfolding,
    nnreweight,
    onnxgbreweight,
    onnxfolding,
    xgbfolding,
    xgbreweight,
    train_and_test,
)

kfolding = gbfolding

__all__ = [
    "__version__",
    "X_LABELS",
    "apply_weights_pipeline",
    "binning_reweight",
    "def_aliases",
    "gbfolding",
    "gbreweight",
    "kfolding",
    "load_data",
    "nnfolding",
    "nnreweight",
    "onnxfolding",
    "onnxgbreweight",
    "optuna_tuning",
    "run_reweighting_pipeline",
    "save_data",
    "train_and_test",
    "xgbfolding",
    "xgbreweight",
]
