from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import time
from mcreweight.io import flatten_vars
from mcreweight.models.onnxreweighter import (
    ONNXGBReweighter,
    ONNXINNReweighter,
    ONNXIXGBReweighter,
    ONNXBinsReweighter,
)
from mcreweight.models.onnxfolding import (
    ONNXFoldingReweighter,
    ONNXINNFoldingReweighter,
    ONNXIXGBFoldingReweighter,
)
from hep_ml.reweight import GBReweighter, FoldingReweighter


def _build_training_metrics(
    method, n_original, n_target, fit_seconds, stage_repetitions=1, outer_repetitions=1
):
    dataset_events = int(n_original + n_target)
    stage_repetitions = max(int(stage_repetitions), 1)
    outer_repetitions = max(int(outer_repetitions), 1)
    effective_events = int(dataset_events * stage_repetitions * outer_repetitions)
    fit_seconds = float(max(fit_seconds, 1e-12))
    return {
        "method": method,
        "n_original": int(n_original),
        "n_target": int(n_target),
        "dataset_events": dataset_events,
        "stage_repetitions": stage_repetitions,
        "outer_repetitions": outer_repetitions,
        "effective_events": effective_events,
        "fit_seconds": fit_seconds,
        "dataset_events_per_second": float(dataset_events / fit_seconds),
        "effective_events_per_second": float(effective_events / fit_seconds),
    }


def _attach_training_metrics(model, **metrics):
    model.training_metrics_ = metrics
    return model


def _tag(columns):
    return "_".join(flatten_vars(columns))


def _fit_with_metrics(
    model,
    fit_callable,
    method,
    n_original,
    n_target,
    stage_repetitions=1,
    outer_repetitions=1,
):
    t0 = time.perf_counter()
    fit_callable()
    fit_seconds = time.perf_counter() - t0
    _attach_training_metrics(
        model,
        **_build_training_metrics(
            method,
            n_original,
            n_target,
            fit_seconds,
            stage_repetitions=stage_repetitions,
            outer_repetitions=outer_repetitions,
        ),
    )
    return model


def _dump_weights(weightsdir, prefix, tag, weights):
    joblib.dump(weights, f"{weightsdir}/{prefix}_{tag}.pkl")


def _persist_joblib(model, prefix):
    joblib.dump(model, prefix + ".pkl")
    return model


def _persist_onnx(model, prefix):
    model.save(prefix)
    return model


def _clip_predicted_weights(args, weights):
    weights = np.asarray(weights, dtype=np.float64)
    clipped = np.clip(weights, 0.0, np.inf)
    finite = clipped[np.isfinite(clipped)]

    if finite.size == 0:
        return clipped

    mean = float(np.mean(finite))
    std = float(np.std(finite))

    if not np.isfinite(std) or std <= 0.0:
        return clipped

    upper = mean + 5.0 * std
    return np.clip(clipped, 0.0, upper)


def _predict_test_weights(model, sample, columns, use_onnx_api):
    X_mc_test = sample["mc_test"][columns]
    if use_onnx_api:
        return model.predict_weights(X_mc_test.to_numpy(), ow=sample["w_mc_test"])
    return model.predict_weights(X_mc_test, original_weight=sample["w_mc_test"])


def _predict_full_weights(model, sample, columns, use_onnx_api, out_of_fold=False):
    X_mc = sample["mc"][columns].to_numpy()
    if use_onnx_api:
        if out_of_fold:
            return model.predict_oof_weights(X_mc, ow=sample["w_mc"])
        return model.predict_weights(X_mc, ow=sample["w_mc"])
    return model.predict_weights(X_mc, original_weight=sample["w_mc"])


def train_and_test(mc, data, mcweights, sweights, columns, test_size):
    """
    Split the data into training and testing sets.

    Args:
        mc (pd.DataFrame): MC data.
        data (pd.DataFrame): Data to reweight to.
        mcweights (np.ndarray): Weights for the MC data.
        sweights (np.ndarray): Sweights for the data.
        columns (list): List of column names to use for training.
        test_size (float): Proportion of the dataset to include in the test split.

    Returns:
       sample (dict): Dictionary containing the training and testing splits for MC and data, along with their weights.
    """
    mc_train, mc_test, w_mc_train, w_mc_test = train_test_split(
        mc[columns], mcweights, test_size=test_size, random_state=42
    )
    data_train, data_test, w_data_train, w_data_test = train_test_split(
        data[columns], sweights, test_size=test_size, random_state=42
    )

    sample = {
        "mc": mc,
        "data": data,
        "w_mc": mcweights,
        "w_data": sweights,
        "mc_train": mc_train,
        "mc_test": mc_test,
        "w_mc_train": w_mc_train,
        "w_mc_test": w_mc_test,
        "data_train": data_train,
        "data_test": data_test,
        "w_data_train": w_data_train,
        "w_data_test": w_data_test,
    }

    return sample


def gbreweight(args, sample, columns, study, weightsdir):
    """
    Train a GradientBoostingClassifier reweighter and predict weights for the test MC data.

    Args:
        args: Command-line arguments containing configuration options.
        sample (dict): Dictionary containing training and test data along with their weights.
        columns (list): List of column names to use for training.
        study (optuna.study.Study): Optuna study object containing the best hyperparameters.
        weightsdir (str): Directory to save the model and weights.
    """
    best = study.best_params if study is not None else {}
    tag = _tag(columns)

    gbr = GBReweighter(
        n_estimators=best.get("gb_n_estimators", best.get("n_estimators", 100)),
        learning_rate=best.get("gb_learning_rate", best.get("learning_rate", 0.1)),
        max_depth=best.get("gb_max_depth", best.get("max_depth", 4)),
    )

    print(f"Training GBReweighter with columns: {columns}")

    _fit_with_metrics(
        gbr,
        lambda: gbr.fit(
            sample["mc_train"][columns],
            sample["data_train"][columns],
            original_weight=sample["w_mc_train"],
            target_weight=sample["w_data_train"],
        ),
        "GB",
        len(sample["mc_train"]),
        len(sample["data_train"]),
        stage_repetitions=gbr.n_estimators,
    )
    gbr = _persist_joblib(gbr, f"{weightsdir}/gbr_model_{tag}")
    new_mc_weights = _clip_predicted_weights(
        args, _predict_test_weights(gbr, sample, columns, use_onnx_api=False)
    )
    _dump_weights(weightsdir, "gbr_weights", tag, new_mc_weights)
    return gbr, new_mc_weights


def xgbreweight(args, sample, columns, weightsdir, study=None):
    """
    Train an iterative XGBoost reweighter using best hyperparameters from Optuna, then predict weights on mc_test.

    Args:
        args: Command-line arguments containing configuration options.
        sample (dict): Dictionary containing training and test data along with their weights.
        columns (list): List of column names to use for training.
        study (optuna.study.Study): Optuna study object containing the best hyperparameters.
        weightsdir (str): Directory to save the model and weights.
    """
    best = study.best_params if study is not None else {}
    tag = _tag(columns)

    # ---- Map Optuna params -> your iterative reweighter params ----
    # Iterative (stage-wise) knobs
    iter_kwargs = dict(
        n_iterations=best.get("n_iterations", 25),  # <-- number of iterations/stages
        mixing_learning_rate=best.get(
            "mixing_learning_rate", 0.05
        ),  # <-- iterative step size
        verbosity=args.verbosity,
        transform=args.transform,
        reweight_validation_fraction=args.reweight_validation_fraction,
        reweight_early_stopping_rounds=args.reweight_early_stopping_rounds,
        reweight_metric_every=args.reweight_metric_every,
    )

    # Base XGB (per-stage classifier) knobs
    xgb_kwargs = dict(
        learning_rate=best.get("xgb_learning_rate", 0.1),  # <-- XGB internal LR
        max_depth=best.get("max_depth", 4),
        subsample=best.get("subsample", 0.8),
        reg_alpha=best.get("reg_alpha", 2.0),
        reg_lambda=best.get("reg_lambda", 5.0),
    )

    xgb = ONNXIXGBReweighter(**iter_kwargs, **xgb_kwargs)

    print(f"Training ONNXIXGBReweighter with Optuna params:\n{best}")

    # IMPORTANT: ensure numpy arrays (your code supports both, but be consistent)
    X_mc_train = sample["mc_train"][columns].to_numpy()
    X_data_train = sample["data_train"][columns].to_numpy()

    _fit_with_metrics(
        xgb,
        lambda: xgb.fit(
            X_mc_train,
            X_data_train,
            ow=sample["w_mc_train"],
            tw=sample["w_data_train"],
        ),
        "XGB",
        len(X_mc_train),
        len(X_data_train),
        stage_repetitions=xgb.n_iterations,
    )

    xgb = _persist_onnx(xgb, f"{weightsdir}/ixgb_model_{tag}")
    new_mc_weights = _clip_predicted_weights(
        args, _predict_test_weights(xgb, sample, columns, use_onnx_api=True)
    )
    _dump_weights(weightsdir, "ixgb_weights", tag, new_mc_weights)
    return xgb, new_mc_weights


def onnxgbreweight(args, sample, columns, weightsdir, study=None):
    """
    Train an ONNX-exportable GB reweighter that mirrors hep_ml's signed-weight loss.
    """
    best = study.best_params if study is not None else {}
    tag = _tag(columns)

    gb_kwargs = dict(
        n_estimators=best.get("gb_n_estimators", best.get("n_estimators", 100)),
        learning_rate=best.get("gb_learning_rate", best.get("learning_rate", 0.1)),
        max_depth=best.get("gb_max_depth", best.get("max_depth", 4)),
        min_samples_leaf=best.get("min_samples_leaf", 200),
        loss_regularization=best.get("loss_regularization", 5.0),
        subsample=best.get("subsample", 1.0),
        verbosity=args.verbosity,
        transform=args.transform,
    )

    onnxgb = ONNXGBReweighter(**gb_kwargs)

    print(f"Training ONNXGBReweighter with params:\n{gb_kwargs}")

    X_mc_train = sample["mc_train"][columns].to_numpy()
    X_data_train = sample["data_train"][columns].to_numpy()

    _fit_with_metrics(
        onnxgb,
        lambda: onnxgb.fit(
            X_mc_train,
            X_data_train,
            ow=sample["w_mc_train"],
            tw=sample["w_data_train"],
        ),
        "ONNXGB",
        len(X_mc_train),
        len(X_data_train),
        stage_repetitions=onnxgb.n_estimators,
    )

    onnxgb = _persist_onnx(onnxgb, f"{weightsdir}/onnxgb_model_{tag}")
    new_mc_weights = _clip_predicted_weights(
        args, _predict_test_weights(onnxgb, sample, columns, use_onnx_api=True)
    )
    _dump_weights(weightsdir, "onnxgb_weights", tag, new_mc_weights)

    return onnxgb, new_mc_weights


def nnreweight(args, sample, columns, weightsdir, study=None):
    """
    Train an iterative NN reweighter using best hyperparameters from Optuna, then predict weights on mc_test.

    Args:
        args: Command-line arguments containing configuration options.
        sample (dict): Dictionary containing training and test data along with their weights.
        columns (list): List of column names to use for training.
        study (optuna.study.Study): Optuna study object containing the best hyperparameters.
        weightsdir (str): Directory to save the model and weights.

    Returns:
        nn (ONNXINNReweighter): Trained NN reweighter model.
        new_mc_weights (np.ndarray): Predicted weights for the MC data using the trained NN reweighter.
    """
    tag = _tag(columns)

    best = study.best_params if study is not None else {}

    iter_kwargs = dict(
        n_iterations=best.get("n_iterations", 5),
        mixing_learning_rate=best.get("mixing_learning_rate", 0.05),
        verbosity=args.verbosity,
        transform=args.transform,
        reweight_validation_fraction=args.reweight_validation_fraction,
        reweight_early_stopping_rounds=args.reweight_early_stopping_rounds,
        reweight_metric_every=args.reweight_metric_every,
    )

    # NN base (per-stage MLP) knobs
    hidden1 = best.get("hidden1", 64)
    hidden2 = best.get("hidden2", 32)

    nn_kwargs = dict(
        hidden_layer_sizes=(hidden1, hidden2),
        alpha=best.get("alpha", 1e-4),
        learning_rate_init=best.get("nn_learning_rate_init", 1e-3),
        batch_size=best.get("batch_size", 1024),
    )

    print(f"Training ONNXINNReweighter with Optuna params:\n{best}")

    nn = ONNXINNReweighter(**iter_kwargs, **nn_kwargs)

    X_mc_train = sample["mc_train"][columns].to_numpy()
    X_data_train = sample["data_train"][columns].to_numpy()

    _fit_with_metrics(
        nn,
        lambda: nn.fit(
            X_mc_train,
            X_data_train,
            ow=sample["w_mc_train"],
            tw=sample["w_data_train"],
        ),
        "NN",
        len(X_mc_train),
        len(X_data_train),
        stage_repetitions=nn.n_iterations,
    )

    nn = _persist_onnx(nn, f"{weightsdir}/inn_model_{tag}")
    new_mc_weights = _clip_predicted_weights(
        args, _predict_test_weights(nn, sample, columns, use_onnx_api=True)
    )
    _dump_weights(weightsdir, "inn_weights", tag, new_mc_weights)

    return nn, new_mc_weights


def gbfolding(args, gb, sample, columns, n_folds, weightsdir):
    """
    Train a folding reweighter using the base GB model, then predict weights on mc_test.

    Args:
        args: Command-line arguments containing configuration options.
        gb (GBReweighter): Base GB reweighter model to use for folding.
        sample (dict): Dictionary containing training and test data along with their weights.
        columns (list): List of column names to use for training.
        n_folds (int): Number of folds for k-folding.
        weightsdir (str): Directory to save the model and weights.
    """
    folding = FoldingReweighter(base_reweighter=gb, n_folds=n_folds)

    _fit_with_metrics(
        folding,
        lambda: folding.fit(
            sample["mc"][columns].to_numpy(),
            sample["data"][columns].to_numpy(),
            original_weight=sample["w_mc"],
            target_weight=sample["w_data"],
        ),
        "Folding",
        len(sample["mc"]),
        len(sample["data"]),
        stage_repetitions=gb.n_estimators,
        outer_repetitions=max(n_folds - 1, 1),
    )

    tag = _tag(columns)
    folding = _persist_joblib(folding, f"{weightsdir}/folding_model_{tag}")
    new_mc_weights = _clip_predicted_weights(
        args, _predict_full_weights(folding, sample, columns, use_onnx_api=False)
    )
    _dump_weights(weightsdir, "folding_weights", tag, new_mc_weights)

    return folding, new_mc_weights


def xgbfolding(args, xgb, sample, columns, n_folds, weightsdir, n_iterations=15):
    """
    Train an ONNXIXGBFoldingReweighter using the base XGB model, then predict weights on mc_test.

    Args:
        args: Command-line arguments containing configuration options.
        xgb (ONNXIXGBReweighter): Base XGB reweighter model to use for folding.
        sample (dict): Dictionary containing training and test data along with their weights.
        columns (list): List of column names to use for training.
        n_folds (int): Number of folds for k-folding.
        weightsdir (str): Directory to save the model and weights.
        n_iterations (int): Number of iterations for the base XGB model.
    """
    base_params = xgb.get_params()
    base_params["n_iterations"] = (
        n_iterations  # Ensure consistent number of iterations for folding
    )
    folding = ONNXIXGBFoldingReweighter(
        n_folds=n_folds,
        shuffle=True,
        aggregation=args.folding_aggregation,
        **base_params,
    )

    _fit_with_metrics(
        folding,
        lambda: folding.fit(
            sample["mc"][columns].to_numpy(),
            sample["data"][columns].to_numpy(),
            ow=sample["w_mc"],
            tw=sample["w_data"],
        ),
        "XGBFolding",
        len(sample["mc"]),
        len(sample["data"]),
        stage_repetitions=base_params.get("n_iterations", n_iterations),
        outer_repetitions=max(n_folds - 1, 1),
    )

    tag = _tag(columns)
    folding = _persist_onnx(folding, f"{weightsdir}/xgbfolding_model_{tag}")
    new_mc_weights = _clip_predicted_weights(
        args,
        _predict_full_weights(
            folding, sample, columns, use_onnx_api=True, out_of_fold=True
        ),
    )
    _dump_weights(weightsdir, "xgbfolding_weights", tag, new_mc_weights)

    return folding, new_mc_weights


def onnxfolding(args, onnxgb, sample, columns, n_folds, weightsdir):
    """
    Train an ONNXFoldingReweighter using the ONNXGB base model.
    """
    base_params = onnxgb.get_params()
    folding = ONNXFoldingReweighter(
        n_folds=n_folds,
        shuffle=True,
        aggregation=args.folding_aggregation,
        **base_params,
    )

    _fit_with_metrics(
        folding,
        lambda: folding.fit(
            sample["mc"][columns].to_numpy(),
            sample["data"][columns].to_numpy(),
            ow=sample["w_mc"],
            tw=sample["w_data"],
        ),
        "ONNXFolding",
        len(sample["mc"]),
        len(sample["data"]),
        stage_repetitions=base_params.get("n_estimators", 1),
        outer_repetitions=max(n_folds - 1, 1),
    )

    tag = _tag(columns)
    folding = _persist_onnx(folding, f"{weightsdir}/onnxfolding_model_{tag}")
    new_mc_weights = _clip_predicted_weights(
        args,
        _predict_full_weights(
            folding, sample, columns, use_onnx_api=True, out_of_fold=True
        ),
    )
    _dump_weights(weightsdir, "onnxfolding_weights", tag, new_mc_weights)

    return folding, new_mc_weights


def nnfolding(args, nn, sample, columns, n_folds, weightsdir, n_iterations=5):
    """
    Train an ONNXINNFoldingReweighter using the base NN model, then predict weights on mc_test.

    Args:
        args: Command-line arguments containing configuration options.
        nn (ONNXINNReweighter): Base NN reweighter model to use for folding.
        sample (dict): Dictionary containing training and test data along with their weights.
        columns (list): List of column names to use for training.
        n_folds (int): Number of folds for k-folding.
        weightsdir (str): Directory to save the model and weights.
        n_iterations (int): Number of iterations for the base NN model.
    """
    base_params = nn.get_params()
    base_params["n_iterations"] = (
        n_iterations  # Ensure consistent number of iterations for folding
    )
    folding = ONNXINNFoldingReweighter(
        n_folds=n_folds,
        shuffle=True,
        aggregation=args.folding_aggregation,
        **base_params,
    )

    _fit_with_metrics(
        folding,
        lambda: folding.fit(
            sample["mc"][columns].to_numpy(),
            sample["data"][columns].to_numpy(),
            ow=sample["w_mc"],
            tw=sample["w_data"],
        ),
        "NNFolding",
        len(sample["mc"]),
        len(sample["data"]),
        stage_repetitions=base_params.get("n_iterations", n_iterations),
        outer_repetitions=max(n_folds - 1, 1),
    )

    tag = _tag(columns)
    folding = _persist_onnx(folding, f"{weightsdir}/nnfolding_model_{tag}")
    new_mc_weights = _clip_predicted_weights(
        args,
        _predict_full_weights(
            folding, sample, columns, use_onnx_api=True, out_of_fold=True
        ),
    )
    _dump_weights(weightsdir, "nnfolding_weights", tag, new_mc_weights)

    return folding, new_mc_weights


def binning_reweight(args, sample, columns, n_bins, n_neighs, weightsdir):
    """
    Train an ONNXBinsReweighter and predict weights for the test MC data.
    Mirrors the behavior of the original BinsReweighter version, with ONNX support.

    Args:
        args: Command-line arguments containing configuration options.
        sample (dict): Dictionary containing training and test data along with their weights.
        columns (list): List of column names to use for training.
        n_bins (int): Number of bins to use for the reweighter.
        n_neighs (float): Number of neighbors for smoothing (ignored in this function).
        weightsdir (str): Directory to save the model and weights.
    """
    bins = ONNXBinsReweighter(
        n_bins=n_bins,
        verbosity=args.verbosity,
        n_neighs=n_neighs,
        transform=args.transform,
    )
    tag = _tag(columns)

    _fit_with_metrics(
        bins,
        lambda: bins.fit(
            sample["mc_train"][columns],
            sample["data_train"][columns],
            ow=sample["w_mc_train"],
            tw=sample["w_data_train"],
        ),
        "Bins",
        len(sample["mc_train"]),
        len(sample["data_train"]),
    )

    bins = _persist_onnx(bins, f"{weightsdir}/binning_model_{tag}")
    new_mc_weights = _clip_predicted_weights(
        args, _predict_test_weights(bins, sample, columns, use_onnx_api=True)
    )
    _dump_weights(weightsdir, "onnx_binning_weights", tag, new_mc_weights)

    return bins, new_mc_weights
