import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler
import shap


def fit_transform(X, transform):
    """
    Fit a transformer to the data and return the transformed data and the fitted transformer.

    Args:
        X (pd.DataFrame): Input data to fit the transformer on.
        transform (str): Type of transformation to apply. If a string, it specifies a predefined transformation.
    """
    t = transform.lower()

    if "quantile" in t:
        tr = QuantileTransformer(output_distribution="uniform", random_state=42)
        tr.fit(X)
        return ("quantile", tr)
    elif "yeo" in t:
        tr = PowerTransformer(method="yeo-johnson")
        tr.fit(X)
        return ("yeo", tr)
    elif "scaler" in t:
        tr = StandardScaler()
        tr.fit(X)
        return ("scaler", tr)
    elif "log" in t:
        return ("log", None)
    raise ValueError(f"Unknown transform: {transform}")


def apply_transform(X, transformer):
    """
    Apply a fitted transformer to the data.

    Args:
        X (pd.DataFrame): Input data to transform.
        transformer (tuple): A tuple containing the type of transformation and the fitted transformer object.
    """
    return (
        transformer[1].transform(X)
        if transformer[0] != "log"
        else np.sign(X) * np.log1p(np.abs(X))
    )


def evaluate_reweighting(
    mc, data, weights_mc, weights_data, label, ax, score_dict=None
):
    """
    Evaluate the reweighting performance using ROC curve and AUC.

    Args:
        mc (np.ndarray): MC data features.
        data (np.ndarray): Data features.
        weights_mc (np.ndarray): Weights for MC data.
        weights_data (np.ndarray): Weights for Data.
        label (str): Label for the plot.
        ax (matplotlib.axes.Axes): Axes to plot the ROC curve on.
        score_dict (dict, optional): Dictionary to store scores for MC and Data.

    Returns:
        float: AUC score for the reweighting performance.
    """
    X = np.vstack([mc, data])
    y = np.hstack([np.zeros(len(mc)), np.ones(len(data))])
    sample_weight = np.hstack([weights_mc, weights_data])

    clf = GradientBoostingClassifier(n_estimators=50, max_depth=3)
    clf.fit(X, y, sample_weight=sample_weight)
    y_scores = clf.predict_proba(X)[:, 1]

    if score_dict is not None:
        score_dict["MC"] = y_scores[: len(mc)]
        score_dict["Data"] = y_scores[len(mc) :]

    fpr, tpr, _ = roc_curve(y, y_scores, sample_weight=sample_weight)
    auc_val = auc(fpr, tpr)
    if ax is not None:
        ax.plot(fpr, tpr, label=f"{label} (AUC = {auc_val:.3f})")
    return auc_val


def get_scores(sample, weights, methods, columns, ax=None):
    """
    Get classifier scores for different reweighting methods.

    Args:
        sample (dict): Dictionary containing MC and Data samples.
        weights (dict): Dictionary containing weights for each method.
        methods (list): List of reweighting methods to evaluate.
        columns (list): List of feature columns to use for evaluation.
        ax (matplotlib.axes.Axes, optional): Axes to plot the ROC curves on. If None, no plotting is done.

    Returns:
        dict: Dictionary of scores for each method.
    """
    scores = {}
    for method in methods:
        scores[method] = {"MC": None, "Data": None}
        method_weights = weights.get(method)
        if method_weights is not None:
            if method in ("Folding", "XGBFolding", "NNFolding", "ONNXFolding"):
                mc_sample = sample["mc"]
                data_sample = sample["data"]
                w_data = sample["w_data"]
            else:
                mc_sample = sample["mc_test"]
                data_sample = sample["data_test"]
                w_data = sample["w_data_test"]
            auc_val = evaluate_reweighting(
                mc_sample[columns].values,
                data_sample[columns].values,
                method_weights,
                w_data,
                method,
                ax,
                scores[method],
            )
            print(f"[INFO] {method} AUC: {auc_val:.3f}")
    return scores


def shap_importance_reweighter(
    model,
    X_mc,
    w_mc,
    columns,
    max_events=100,
    random_state=42,
):
    """
    Compute SHAP values for a reweighter.

    Args:
        model: Reweighter model with a predict_weights method.
        X_mc (pd.DataFrame): MC data features.
        w_mc (np.ndarray): Weights for the MC data.
        columns (list): List of feature columns to compute SHAP values for.
        max_events (int): Maximum number of events to use for SHAP computation.
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: Dictionary of SHAP values for each feature.
    """
    rng = np.random.default_rng(random_state)

    if len(X_mc) > max_events:
        idx = rng.choice(len(X_mc), max_events, replace=False)
        X_eval = X_mc.iloc[idx]
        w_eval = w_mc[idx]
    else:
        X_eval = X_mc
        w_eval = w_mc

    def model_fn(X_subset):
        """
        SHAP may pass a subset of X. Only use the weights corresponding to this subset.
        """
        # Use positional slicing relative to X_eval
        if isinstance(X_subset, pd.DataFrame):
            # Get the positions of X_subset within X_eval
            pos = X_eval.index.get_indexer(X_subset.index)
            w_subset = w_eval[pos]
        else:
            # numpy array: assume rows are in the same order
            w_subset = w_eval[: len(X_subset)]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="hep_ml")
            w = model.predict_weights(X_subset, w_subset)
        return np.log(w + 1e-12)

    explainer = shap.Explainer(model_fn, X_eval)
    shap_values = explainer(X_eval).values  # (N, F)

    out = {}
    for i, name in enumerate(columns):
        out[name] = shap_values[:, i]

    return out, X_eval


def update_scores_with_importance(args, models, scores, sample, methods, columns):
    """
    Update the scores dictionary with SHAP importance values for each method.

    Args:
        args (argparse.Namespace): Command line arguments containing verbosity flag.
        models (dict): Dictionary of trained reweighter models.
        scores (dict): Dictionary of scores to update with importance values.
        sample (dict): Dictionary containing MC and Data samples.
        methods (list): List of reweighting methods to compute importance for.
        columns (list): List of feature columns to compute SHAP values for.

    Returns:
        dict: Updated scores dictionary with SHAP importance values.
    """
    if args.verbosity >= 3:
        print("[INFO] Computing SHAP importance values for each method...")
        print(
            "mc_test shape:",
            sample["mc_test"].shape,
            "w_mc_test shape:",
            sample["w_mc_test"].shape,
        )
        print("mc shape:", sample["mc"].shape, "w_mc shape:", sample["w_mc"].shape)
    if "GB" in methods:
        shap_vals, X_eval = shap_importance_reweighter(
            models["GB"],
            X_mc=sample["mc_test"][columns],
            w_mc=sample["w_mc_test"],
            columns=columns,
        )
        scores["GB_importances"] = (shap_vals, X_eval)
    if "XGB" in methods:
        shap_vals, X_eval = shap_importance_reweighter(
            models["XGB"],
            X_mc=sample["mc_test"][columns],
            w_mc=sample["w_mc_test"],
            columns=columns,
        )
        scores["XGB_importances"] = (shap_vals, X_eval)
    if "ONNXGB" in methods:
        shap_vals, X_eval = shap_importance_reweighter(
            models["ONNXGB"],
            X_mc=sample["mc_test"][columns],
            w_mc=sample["w_mc_test"],
            columns=columns,
        )
        scores["ONNXGB_importances"] = (shap_vals, X_eval)
    if "NN" in methods:
        shap_vals, X_eval = shap_importance_reweighter(
            models["NN"],
            X_mc=sample["mc_test"][columns],
            w_mc=sample["w_mc_test"],
            columns=columns,
        )
        scores["NN_importances"] = (shap_vals, X_eval)
    if "Bins" in methods:
        shap_vals, X_eval = shap_importance_reweighter(
            models["Bins"],
            X_mc=sample["mc_test"][columns],
            w_mc=sample["w_mc_test"],
            columns=columns,
        )
        scores["Bins_importances"] = (shap_vals, X_eval)
    if (
        "Folding" in methods
        or "XGBFolding" in methods
        or "NNFolding" in methods
        or "ONNXFolding" in methods
    ):
        if args.verbosity >= 2:
            print("[INFO] Skipping SHAP importance for folding methods.")

    return scores


def weighted_corr_matrix(df, columns, weights, allow_signed_fallback=True):
    """Compute (weighted) correlation matrix for a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        columns (list): List of column names to include in the correlation matrix.
        weights (np.ndarray, optional): Weights for the correlation calculation.

    Returns:
        tuple[pd.DataFrame, str]: Weighted correlation matrix and the weight mode used.
    """
    data = df[columns].values
    weights = np.asarray(weights, dtype=np.float64)

    def _corr_from_weights(active_weights):
        mean = np.average(data, axis=0, weights=active_weights)
        xm = data - mean
        cov = np.dot(active_weights * xm.T, xm) / np.sum(active_weights)
        diag = np.diag(cov)
        if np.any(diag < -1e-12):
            raise ValueError("Signed weights produced a non-positive variance.")
        diag = np.clip(diag, 0.0, np.inf)
        stddev = np.sqrt(diag)
        denom = np.outer(stddev, stddev)
        corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)
        corr = np.clip(corr, -1, 1)
        return pd.DataFrame(corr, index=columns, columns=columns)

    try:
        return _corr_from_weights(weights), "signed"
    except Exception:
        if not allow_signed_fallback:
            raise
        abs_weights = np.abs(weights)
        if np.sum(abs_weights) <= 0:
            return df[columns].corr(), "unweighted"
        return _corr_from_weights(abs_weights), "absolute"


def weighted_ks_statistic(x1, x2, w1=None, w2=None):
    """
    Compute weighted two-sample KS statistic.

    Args:
        x1 (array-like): Sample 1 data.
        x2 (array-like): Sample 2 data.
        w1 (array-like, optional): Weights for sample 1. If None, uniform weights are assumed.
        w2 (array-like, optional): Weights for sample 2. If None, uniform weights are assumed.

    Returns:
        float: KS distance between the two samples.
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    if w1 is None:
        w1 = np.ones_like(x1)
    if w2 is None:
        w2 = np.ones_like(x2)

    # Sort samples
    idx1 = np.argsort(x1)
    idx2 = np.argsort(x2)

    x1 = x1[idx1]
    x2 = x2[idx2]
    w1 = w1[idx1]
    w2 = w2[idx2]

    # Normalize weights
    w1 = w1 / np.sum(w1)
    w2 = w2 / np.sum(w2)

    # Build combined grid
    x_all = np.concatenate([x1, x2])
    x_all.sort()

    # CDFs
    cdf1 = np.searchsorted(x1, x_all, side="right")
    cdf2 = np.searchsorted(x2, x_all, side="right")

    cdf1 = np.array([np.sum(w1[:i]) for i in cdf1])
    cdf2 = np.array([np.sum(w2[:i]) for i in cdf2])

    return np.max(np.abs(cdf1 - cdf2))
