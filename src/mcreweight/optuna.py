import os
import numpy as np
import joblib
import optuna
from mcreweight.models.onnxreweighter import (
    ONNXGBReweighter,
    ONNXINNReweighter,
    ONNXIXGBReweighter,
)
from mcreweight.utils.utils import evaluate_reweighting
from mcreweight.io import flatten_vars
from hep_ml.reweight import GBReweighter


def optuna_tuning(
    args, mc, data, mcweights, sweights, columns, n_trials, weightsdir, classifier_type
):
    """
    Run Optuna to optimize hyperparameters for GBReweighter and ONNX-capable
    reweighters, with caching to avoid redundant runs.

    Args:
        args: Command-line arguments containing configuration options.
        mc (pd.DataFrame): MC data.
        data (pd.DataFrame): Data to reweight to.
        mcweights (np.ndarray): Weights for the MC data.
        sweights (np.ndarray): Sweights for the data.
        columns (list): List of column names to use for training.
        n_trials (int): Number of trials for hyperparameter optimization.
        weightsdir (str): Directory to save the weights and model.
        classifier_type (str): Type of classifier ("GB", "ONNXGB", "XGB" or
            "NN").
    Returns:
        study: Optuna study object containing the results of the optimization.
    """
    saving_vars = flatten_vars(columns)
    study_path = (
        f"{weightsdir}/optuna_study_{classifier_type}_" + "_".join(saving_vars) + ".pkl"
    )
    if joblib.os.path.exists(study_path):
        # Load existing study if it exists
        print(f"Loading existing study from {study_path}")
        study = joblib.load(study_path)
    else:
        # Run Optuna to create a new study
        print(f"Creating new Optuna study and saving to {study_path}")
        # Run optuna optimization
        study = run_optuna(
            args,
            mc=mc,
            data=data,
            mcweights=mcweights,
            sweights=sweights,
            columns=columns,
            n_trials=n_trials,
            weightsdir=weightsdir,
            classifier_type=classifier_type,
        )
        joblib.dump(study, study_path)
    return study


def run_optuna(
    args, mc, data, mcweights, sweights, columns, n_trials, weightsdir, classifier_type
):
    """
    Hyperparameter tuning for GBReweighter and ONNX reweighters:
      - ONNXGBReweighter (hep_ml-like tree ensemble)
      - ONNXIXGBReweighter (XGB base estimator)
      - ONNXINNReweighter (NN base estimator)
    and for stage-wise update (n_iterations, mixing_learning_rate).

    Optimizes BOTH:
      (A) iterative reweighting hyperparams (stage-wise updates)
      (B) base classifier hyperparams (XGB or MLP)
    Or just the direct estimator hyperparameters if you want to optimize
    GBReweighter or ONNXGBReweighter.

    The objective is the AUC of a classifier trained to distinguish reweighted MC from Data.

    Note: the iterative reweighter should internally handle the iterative updates and the base estimator training.

    Args:
        args: Command line arguments (for verbosity, transform, etc.)
        mc, data (pd.DataFrame): MC and Data samples.
        mcweights, sweights (np.ndarray): Weights for MC and Data.
        columns (list): List of feature columns to use for training the reweighter and evaluating the AUC.
        n_trials (int): Number of Optuna trials for hyperparameter optimization.
        weightsdir (str): Directory to save the trained reweighter weights.
        classifier_type (str): Type of base classifier to use ("GB", "ONNXGB",
            "XGB" or "NN").

    Returns:
        optuna.Study: The Optuna study object containing the results of the hyperparameter optimization
    """

    os.makedirs(weightsdir, exist_ok=True)
    saving_vars = flatten_vars(columns)
    tag = "_".join(saving_vars)

    classifier_type = classifier_type.upper().strip()
    if classifier_type not in ("GB", "ONNXGB", "XGB", "NN"):
        raise ValueError(f"Unsupported classifier type: {classifier_type}")

    def suggest_iterative_params(trial):
        # These control the GBReweighter-like stage-wise update.
        return dict(
            n_iterations=trial.suggest_int("n_iterations", 5, 25),
            mixing_learning_rate=trial.suggest_float(
                "mixing_learning_rate", 0.05, 0.3, log=True
            ),
        )

    def suggest_gb_params(trial):
        """
        Hyperparameters for a GBReweighter-like reweighter (if you want to optimize that instead of the iterative ONNX ones).

        Keep the search space relatively small to avoid long training times, but feel free to expand it as needed.
        """
        return dict(
            n_estimators=trial.suggest_int("gb_n_estimators", 50, 150, step=10),
            learning_rate=trial.suggest_float("gb_learning_rate", 0.05, 0.3, log=True),
            max_depth=trial.suggest_int("gb_max_depth", 3, 8, step=1),
        )

    def suggest_onnxgb_params(trial):
        """
        Hyperparameters for the native ONNXGB reweighter.

        This starts from the same boosting controls as the hep_ml-compatible GB
        study and adds the extra ONNXGB tree/update knobs exposed by the class.
        """
        return dict(
            n_estimators=trial.suggest_int("gb_n_estimators", 50, 150, step=10),
            learning_rate=trial.suggest_float("gb_learning_rate", 0.05, 0.3, log=True),
            max_depth=trial.suggest_int("gb_max_depth", 3, 8, step=1),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 50, 500, step=50),
            loss_regularization=trial.suggest_float(
                "loss_regularization", 1.0, 20.0, log=True
            ),
            subsample=trial.suggest_float("subsample", 0.5, 1.0, step=0.1),
        )

    def suggest_xgb_params(trial):
        """
        Base estimator params for the XGB reweighter.

        Keep the search space relatively small to avoid long training times, but feel free to expand it as needed.
        """
        return dict(
            xgb_learning_rate=trial.suggest_float(
                "xgb_learning_rate", 0.05, 0.3, log=True
            ),
            max_depth=trial.suggest_int("max_depth", 4, 8, step=1),
            subsample=trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 5.0, step=0.5),
            reg_lambda=trial.suggest_float("reg_lambda", 1.0, 10.0, step=1),
        )

    def suggest_nn_params(trial):
        """
        Base estimator params for the NN reweighter.

        Keep the search space relatively small to avoid long training times, but feel free to expand it as needed.
        """
        n1 = trial.suggest_int("hidden1", 32, 128, step=16)
        n2 = trial.suggest_int("hidden2", 16, 64, step=16)

        return dict(
            hidden_layer_sizes=(n1, n2),
            alpha=trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            learning_rate_init=trial.suggest_float(
                "nn_learning_rate_init", 1e-4, 5e-3, log=True
            ),
            batch_size=trial.suggest_categorical("batch_size", [256, 512, 1024]),
            max_iter=trial.suggest_int("max_iter", 50, 180, step=10),
        )

    def objective(trial):
        try:
            if classifier_type == "GB":
                gb_params = suggest_gb_params(trial)
                classifier = GBReweighter(
                    n_estimators=gb_params["n_estimators"],
                    learning_rate=gb_params["learning_rate"],
                    max_depth=gb_params["max_depth"],
                )
                classifier.fit(
                    mc[columns].to_numpy(),
                    data[columns].to_numpy(),
                    original_weight=mcweights,
                    target_weight=sweights,
                )
                weights_pred = classifier.predict_weights(
                    mc[columns].to_numpy(), original_weight=mcweights
                )

            elif classifier_type == "ONNXGB":
                onnxgb_params = suggest_onnxgb_params(trial)
                classifier = ONNXGBReweighter(
                    verbosity=args.verbosity,
                    transform=args.transform,
                    n_estimators=onnxgb_params["n_estimators"],
                    learning_rate=onnxgb_params["learning_rate"],
                    max_depth=onnxgb_params["max_depth"],
                    min_samples_leaf=onnxgb_params["min_samples_leaf"],
                    loss_regularization=onnxgb_params["loss_regularization"],
                    subsample=onnxgb_params["subsample"],
                )
                classifier.fit(
                    mc[columns].to_numpy(),
                    data[columns].to_numpy(),
                    ow=mcweights,
                    tw=sweights,
                )

                prefix = f"{weightsdir}/optuna_kfold_{classifier_type}_{tag}"
                classifier.save(prefix)
                classifier.load(prefix)

                weights_pred = classifier.predict_weights(
                    mc[columns].to_numpy(), ow=mcweights
                )

            else:
                iter_params = suggest_iterative_params(trial)

                if classifier_type == "XGB":
                    xgb_params = suggest_xgb_params(trial)

                    classifier = ONNXIXGBReweighter(
                        verbosity=args.verbosity,
                        transform=args.transform,
                        # iterative knobs (your folding class should forward these to the iterative reweighter)
                        n_iterations=iter_params["n_iterations"],
                        mixing_learning_rate=iter_params["mixing_learning_rate"],
                        # base XGB knobs
                        learning_rate=xgb_params[
                            "xgb_learning_rate"
                        ],  # XGB internal LR
                        max_depth=xgb_params["max_depth"],
                        subsample=xgb_params["subsample"],
                        reg_alpha=xgb_params["reg_alpha"],
                        reg_lambda=xgb_params["reg_lambda"],
                    )

                else:  # "NN"
                    nn_params = suggest_nn_params(trial)

                    classifier = ONNXINNReweighter(
                        verbosity=args.verbosity,
                        transform=args.transform,
                        # iterative knobs
                        n_iterations=iter_params["n_iterations"],
                        mixing_learning_rate=iter_params["mixing_learning_rate"],
                        # base MLP knobs
                        hidden_layer_sizes=nn_params.get(
                            "hidden_layer_sizes", (64, 32)
                        ),
                        alpha=nn_params.get("alpha", 1e-4),
                        learning_rate_init=nn_params.get(
                            "nn_learning_rate_init",
                            nn_params.get("learning_rate_init", 1e-3),
                        ),
                        batch_size=nn_params.get("batch_size", 1024),
                    )

                # ---- Fit folding ----
                classifier.fit(
                    mc[columns].to_numpy(),
                    data[columns].to_numpy(),
                    ow=mcweights,
                    tw=sweights,
                )

                # ---- Save/load (use prefix, not necessarily .onnx) ----
                prefix = f"{weightsdir}/optuna_kfold_{classifier_type}_{tag}"
                classifier.save(prefix)
                classifier.load(prefix)

                weights_pred = classifier.predict_weights(
                    mc[columns].to_numpy(), ow=mcweights
                )

            score = evaluate_reweighting(
                mc=mc[columns].values,
                data=data[columns].values,
                weights_mc=weights_pred,
                weights_data=sweights,
                label=f"trial_{classifier_type}",
                ax=None,
            )

            if np.isnan(score) or np.isinf(score):
                raise ValueError("Invalid score")

            return float(score)

        except Exception as e:
            print(f"[WARNING] Trial failed: {e}")
            return +np.inf

    # ----- Seeds: different for GB, XGB and NN -----
    if classifier_type == "GB":
        initial_params = dict(
            gb_n_estimators=100,
            gb_learning_rate=0.1,
            gb_max_depth=5,
        )
    elif classifier_type == "ONNXGB":
        initial_params = dict(
            gb_n_estimators=100,
            gb_learning_rate=0.1,
            gb_max_depth=4,
            min_samples_leaf=200,
            loss_regularization=5.0,
            subsample=1.0,
        )
    elif classifier_type == "XGB":
        initial_params = dict(
            n_iterations=5,
            mixing_learning_rate=0.1,
            xgb_learning_rate=0.1,
            max_depth=6,
            subsample=0.9,
            reg_alpha=1.0,
            reg_lambda=5.0,
        )
    else:
        initial_params = dict(
            n_iterations=5,
            mixing_learning_rate=0.1,
            hidden1=64,
            hidden2=32,
            alpha=1e-4,
            nn_learning_rate_init=1e-3,
            batch_size=1024,
        )

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.enqueue_trial(initial_params)

    if args.verbosity >= 2:
        print(f"[INFO] Optuna {classifier_type}: {n_trials} trials, features={columns}")
        print(
            f"[INFO] Shapes: MC={mc.shape}, Data={data.shape}, wMC={mcweights.shape}, wData={sweights.shape}"
        )

    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    return study
