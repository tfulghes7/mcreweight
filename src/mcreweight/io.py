from importlib import resources
from pathlib import Path
import uproot
import awkward as ak
import numpy as np
import pandas as pd
import numexpr as ne
import re
import yaml


def load_config(path):
    """
    Load YAML configuration file.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_x_labels_path(path=None):
    """
    Resolve the labels YAML from an explicit path or the installed package data.
    """
    if path:
        return path
    return resources.files("mcreweight.utils").joinpath("default_xlabels.yaml")


def get_from_cfg(cfg, keys, default=None):
    """
    Safely extract nested YAML value.
    """
    out = cfg
    for k in keys:
        if out is None or k not in out:
            return default
        out = out[k]
    return out


def extract_variables_from_expression(expr):
    """
    Extract variable names from a mathematical expression string.
    Assumes variable names contain letters, numbers, or underscores.
    Ignores function names like 'log'.
    """
    # Remove function calls like log(...), sin(...), etc.
    expr_no_funcs = re.sub(r"\b[a-zA-Z_][a-zA-Z0-9_]*\s*\(", "", expr)
    # Extract variable-like tokens
    tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", expr_no_funcs)
    # Filter out math functions
    known_funcs = {"log", "exp", "sqrt", "sin", "cos", "tan", "abs"}
    return {tok for tok in tokens if tok not in known_funcs}


def load_data(path, tree, columns, weights_col=None, return_mask=False):
    """
    Load data from a ROOT file and return a DataFrame with computed expressions and optional weights.

    Args:
        path (list): List of paths to the ROOT files.
        tree (str): Name of the tree to read from.
        columns (list): List of column names or expressions (e.g. ["pt", "log(pt)", "pt/eta"]).
        weights_col (str, optional): Name of the column containing weights, or a
            mathematical expression built from branch names (e.g. ``"w1*w2"``,
            ``"w1/w2"``, ``"log(w1)"``).

    Returns:
        df (pd.DataFrame): DataFrame with evaluated columns.
        weights (np.ndarray): Array of weights.
        mask (np.ndarray, optional): Boolean mask mapping kept rows onto the full concatenated input.
    """
    columns = list(columns)  # in case it's a tuple or other iterable
    expr_map = {}  # final_name -> expression
    needed_vars = set()

    for col in columns:
        if any(op in col for op in "+-*/()") or "log" in col or "exp" in col:
            expr_map[col] = col  # column is an expression
            needed_vars.update(extract_variables_from_expression(col))
        else:
            expr_map[col] = col
            needed_vars.add(col)

    if weights_col:
        if any(op in weights_col for op in "+-*/") or re.search(
            r"\b[a-zA-Z_]\w*\s*\(", weights_col
        ):
            needed_vars.update(extract_variables_from_expression(weights_col))
        else:
            needed_vars.add(weights_col)

    def _to_dataframe(obj):
        if isinstance(obj, pd.DataFrame):
            return obj
        if isinstance(obj, ak.Array):
            return ak.to_dataframe(obj).reset_index(drop=True)
        raise TypeError(f"Unsupported array type returned by uproot: {type(obj)!r}")

    dfs = []
    for p in path:
        with uproot.open(p) as f:
            arrays = f[tree].arrays(list(needed_vars), library="pd")
            dfs.append(_to_dataframe(arrays))
    df = pd.concat(dfs, ignore_index=True)

    out_df = pd.DataFrame()
    for name, expr in expr_map.items():
        try:
            out_df[name] = ne.evaluate(expr, local_dict=df)
        except Exception as e:
            raise ValueError(
                f"Failed to evaluate expression '{expr}' for column '{name}': {e}"
            )

    if weights_col:
        if any(op in weights_col for op in "+-*/") or re.search(
            r"\b[a-zA-Z_]\w*\s*\(", weights_col
        ):
            weights = ne.evaluate(weights_col, local_dict=df)
        else:
            weights = df[weights_col].values
    else:
        weights = np.ones(len(df))
    out_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    mask = out_df.notna().all(axis=1) & np.isfinite(weights)
    out_df = out_df.loc[mask].reset_index(drop=True)
    weights = weights[mask]
    if return_mask:
        return out_df, weights, mask.to_numpy(dtype=bool)
    return out_df, weights


def save_data(
    input_path,
    tree,
    output_path,
    output_tree,
    branch,
    weights,
    ntuple_type,
    row_mask=None,
):
    """
    Save weights to a ROOT file.

    Args:
        input_path (list): List of paths to the input ROOT files.
        tree (str): Name of the tree to read from.
        output_path (str): Path to the output ROOT file.
        output_tree (str): Name of the tree to write to.
        branch (str): Name of the branch to save weights under.
        weights (np.ndarray): Weights to save.
        ntuple_type (str): Type of ntuple to write ("RNTuple" or "TTree").
        row_mask (np.ndarray, optional): Boolean mask of selected rows in the full input tree.
    """
    arrays_list = []
    for p in input_path:
        with uproot.open(p) as f:
            arrays_list.append(f[tree].arrays(library="ak"))

    if not arrays_list:
        raise ValueError("No input ROOT files were provided.")

    data = (
        arrays_list[0] if len(arrays_list) == 1 else ak.concatenate(arrays_list, axis=0)
    )
    n_rows = len(data)
    weights = np.asarray(weights)
    if row_mask is not None:
        row_mask = np.asarray(row_mask, dtype=bool)
        if len(row_mask) != n_rows:
            raise ValueError(
                f"Row mask length ({len(row_mask)}) does not match input rows ({n_rows})."
            )
        if int(np.count_nonzero(row_mask)) != len(weights):
            raise ValueError(
                f"Output weights length ({len(weights)}) does not match selected rows "
                f"({int(np.count_nonzero(row_mask))})."
            )
        full_weights = np.full(n_rows, np.nan, dtype=np.float32)
        full_weights[row_mask] = weights.astype(np.float32)
        data = ak.with_field(data, ak.Array(full_weights), branch)
    else:
        if n_rows != len(weights):
            raise ValueError(
                f"Output weights length ({len(weights)}) does not match input rows ({n_rows})."
            )
        data = ak.with_field(data, ak.Array(weights.astype(np.float32)), branch)

    branch_dict = {field: data[field] for field in ak.fields(data)}
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with uproot.recreate(output_path) as f:
        if ntuple_type == "RNTuple":
            writer = f.mkrntuple(output_tree, branch_dict)
            writer.extend(branch_dict)
        elif ntuple_type == "TTree":
            f[output_tree] = branch_dict
        else:
            raise ValueError(f"Unsupported ntuple type: {ntuple_type}")


def def_aliases(df, aliases):
    """
    Apply aliases to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to apply aliases to.
        aliases (dict): Dictionary of aliases where keys are new names and values are expressions.

    Returns:
        pd.DataFrame: DataFrame with applied aliases.
    """
    for new_name, expr in aliases.items():
        try:
            if expr in df.columns:
                df[new_name] = df[expr]
            else:
                df[new_name] = df.eval(expr)
        except Exception as e:
            print(f"Error applying alias '{new_name}': {e}")
    return df


def flatten_vars(lst):
    """
    Flatten a names list

    Args:
        lst (list): List of lists or a single list.
    """

    def transform(x):
        # Replace operators
        x = x.replace("/", "over")
        x = x.replace("+", "plus")
        x = x.replace("*", "times")
        x = x.replace("-", "minus")
        # Flatten nested function calls like sqrt(log(pt)) -> sqrtlogpt
        while True:
            new_x = re.sub(r"(\w+)\(([^()]+)\)", r"\1\2", x)
            if new_x == x:
                break
            x = new_x
        x = x.replace("(", "").replace(")", "")
        return x

    return [transform(x) for x in lst if isinstance(x, str) and x.strip() != ""] + [
        x for x in lst if not isinstance(x, str) or x.strip() == ""
    ]


def read_x_labels(path):
    """
    Read x-axis labels from a YAML file.
    """
    with open(Path(path), "r") as f:
        return yaml.safe_load(f) or {}
