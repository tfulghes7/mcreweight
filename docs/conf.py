import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "mcreweight"
author = "Tommaso Fulghesu"

extensions = ["sphinx.ext.autodoc", "sphinx.ext.mathjax"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_static_path = ["_static"]

autodoc_member_order = "bysource"
autodoc_mock_imports = [
    "awkward",
    "awkward_pandas",
    "hep_ml",
    "joblib",
    "matplotlib",
    "matplotlib.pyplot",
    "numexpr",
    "onnx",
    "onnxruntime",
    "onnxmltools",
    "optuna",
    "pandas",
    "seaborn",
    "shap",
    "skl2onnx",
    "sklearn",
    "uproot",
    "xgboost",
    "yaml",
]
