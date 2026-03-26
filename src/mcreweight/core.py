import json

import joblib
import numpy as np

import mcreweight.utils.plotting_utils as plt
from mcreweight.io import (
    def_aliases,
    flatten_vars,
    load_data,
    read_x_labels,
    resolve_x_labels_path,
    save_data,
)
from mcreweight.models.onnxfolding import (
    ONNXFoldingReweighter,
    ONNXINNFoldingReweighter,
    ONNXIXGBFoldingReweighter,
)
from mcreweight.models.onnxreweighter import (
    ONNXBinsReweighter,
    ONNXGBReweighter,
    ONNXINNReweighter,
    ONNXIXGBReweighter,
)
from mcreweight.optuna import optuna_tuning
from mcreweight.train import (
    binning_reweight,
    gbreweight,
    gbfolding,
    nnfolding,
    nnreweight,
    onnxgbreweight,
    onnxfolding,
    train_and_test,
    xgbfolding,
    xgbreweight,
)
from mcreweight.utils.utils import update_scores_with_importance


FOLDING_METHODS = {"Folding", "XGBFolding", "ONNXFolding", "NNFolding"}
ONNX_APPLY_MODELS = {
    "ONNXGB": ONNXGBReweighter,
    "ONNXFolding": ONNXFoldingReweighter,
    "XGB": ONNXIXGBReweighter,
    "XGBFolding": ONNXIXGBFoldingReweighter,
    "NN": ONNXINNReweighter,
    "NNFolding": ONNXINNFoldingReweighter,
    "Bins": ONNXBinsReweighter,
}
MODEL_PREFIXES = {
    "GB": "gbr_model",
    "Folding": "folding_model",
    "ONNXGB": "onnxgb_model",
    "ONNXFolding": "onnxfolding_model",
    "XGB": "ixgb_model",
    "XGBFolding": "xgbfolding_model",
    "NN": "inn_model",
    "NNFolding": "nnfolding_model",
    "Bins": "binning_model",
}
METHOD_PLOT_LABELS = {
    "GB": "gb",
    "Folding": "folding",
    "ONNXGB": "onnxgb",
    "ONNXFolding": "onnxfolding",
    "XGB": "xgb",
    "XGBFolding": "xgbfolding",
    "NN": "nn",
    "NNFolding": "nnfolding",
    "Bins": "binning",
}
TRAINING_GROUPS = (
    {
        "base_method": "GB",
        "folding_method": "Folding",
        "classifier_type": "GB",
        "base_train_fn": gbreweight,
        "fold_train_fn": gbfolding,
        "base_result_key": "GB",
        "fold_result_key": "Folding",
        "fold_arg_name": "gb",
    },
    {
        "base_method": "ONNXGB",
        "folding_method": "ONNXFolding",
        "classifier_type": "GB",
        "base_train_fn": onnxgbreweight,
        "fold_train_fn": onnxfolding,
        "base_result_key": "ONNXGB",
        "fold_result_key": "ONNXFolding",
        "fold_arg_name": "onnxgb",
    },
    {
        "base_method": "XGB",
        "folding_method": "XGBFolding",
        "classifier_type": "XGB",
        "base_train_fn": xgbreweight,
        "fold_train_fn": xgbfolding,
        "base_result_key": "XGB",
        "fold_result_key": "XGBFolding",
        "fold_arg_name": "xgb",
    },
    {
        "base_method": "NN",
        "folding_method": "NNFolding",
        "classifier_type": "NN",
        "base_train_fn": nnreweight,
        "fold_train_fn": nnfolding,
        "base_result_key": "NN",
        "fold_result_key": "NNFolding",
        "fold_arg_name": "nn",
    },
)


def _log_loaded_samples(args, mc, data, mc_weights, sweights):
    if args.verbosity < 3:
        return
    print(
        f"[INFO] Loaded MC data with shape {mc.shape} and Data with shape {data.shape}."
    )
    if mc_weights is not None:
        print(
            f"[INFO] MC weights: min={np.min(mc_weights)}, max={np.max(mc_weights)}, "
            f"mean={np.mean(mc_weights)}, sum={np.sum(mc_weights)}"
        )
    else:
        print("[INFO] No MC weights provided, using uniform weights.")
    if sweights is not None:
        print(
            f"[INFO] Data weights: min={np.min(sweights)}, max={np.max(sweights)}, "
            f"mean={np.mean(sweights)}, sum={np.sum(sweights)}"
        )
    else:
        print("[INFO] No Data weights provided, using uniform weights.")


def _load_training_inputs(args):
    vars_list = list(args.training_vars)
    monitoring_vars_list = list(args.monitoring_vars or [])
    columns = vars_list + monitoring_vars_list

    data, sweights = load_data(
        path=args.path_data,
        tree=args.tree_data,
        columns=columns,
        weights_col=args.sweights_name,
    )
    mc, mc_weights = load_data(
        path=args.path_mc,
        tree=args.tree_mc,
        columns=columns,
        weights_col=args.mcweights_name,
    )
    _log_loaded_samples(args, mc, data, mc_weights, sweights)
    return mc, data, mc_weights, sweights


def _log_sample_split(args, sample):
    if args.verbosity < 2:
        return
    print(
        f"[INFO] Training sample: MC shape {sample['mc_train'].shape}, Data shape {sample['data_train'].shape}"
    )
    print(
        f"[INFO] Testing sample: MC shape {sample['mc_test'].shape}, Data shape {sample['data_test'].shape}"
    )
    print(
        f"[INFO] Training weights: MC avg {np.mean(sample['w_mc_train'])}, Data avg {np.mean(sample['w_data_train'])}"
    )
    print(
        f"[INFO] Testing weights: MC avg {np.mean(sample['w_mc_test'])}, Data avg {np.mean(sample['w_data_test'])}"
    )


def _describe_weights(label, weights):
    return (
        f"{label}: min={np.min(weights):.6g}, max={np.max(weights):.6g}, "
        f"mean={np.mean(weights):.6g}, sum={np.sum(weights):.6g}"
    )


def _plot_input_distributions(args, sample, x_labels, plotdir):
    if args.verbosity >= 1:
        print("[INFO] Plotting original distributions...")

    plt.plot_distributions(
        args,
        mc=sample["mc_train"],
        data=sample["data_train"],
        mc_weights=sample["w_mc_train"],
        data_weights=sample["w_data_train"],
        columns=args.training_vars,
        x_labels=x_labels,
        output_file=f"{plotdir}/input_features_training.png",
    )
    plt.plot_distributions(
        args,
        mc=sample["mc_test"],
        data=sample["data_test"],
        mc_weights=sample["w_mc_test"],
        data_weights=sample["w_data_test"],
        columns=args.training_vars,
        x_labels=x_labels,
        output_file=f"{plotdir}/input_features_testing.png",
    )

    if args.transform is not None:
        if args.verbosity >= 1:
            print("[INFO] Plotting distributions after transformation...")
        plt.plot_distributions(
            args,
            mc=sample["mc_train"],
            data=sample["data_train"],
            mc_weights=sample["w_mc_train"],
            data_weights=sample["w_data_train"],
            columns=args.training_vars,
            x_labels=x_labels,
            transform=args.transform,
            output_file=f"{plotdir}/input_features_training_transformed.png",
        )
        plt.plot_distributions(
            args,
            mc=sample["mc_test"],
            data=sample["data_test"],
            mc_weights=sample["w_mc_test"],
            data_weights=sample["w_data_test"],
            columns=args.training_vars,
            x_labels=x_labels,
            transform=args.transform,
            output_file=f"{plotdir}/input_features_testing_transformed.png",
        )

    if args.monitoring_vars is not None:
        if args.verbosity >= 1:
            print("[INFO] Plotting original distributions for monitoring variables...")
        plt.plot_distributions(
            args,
            mc=sample["mc_train"],
            data=sample["data_train"],
            mc_weights=sample["w_mc_train"],
            data_weights=sample["w_data_train"],
            columns=args.monitoring_vars,
            x_labels=x_labels,
            output_file=f"{plotdir}/other_vars_training.png",
        )
        plt.plot_distributions(
            args,
            mc=sample["mc_test"],
            data=sample["data_test"],
            mc_weights=sample["w_mc_test"],
            data_weights=sample["w_data_test"],
            columns=args.monitoring_vars,
            x_labels=x_labels,
            output_file=f"{plotdir}/other_vars_testing.png",
        )


def _train_models(args, sample, columns, weightsdir, mc, data, mc_weights, sweights):
    results = {}

    if args.verbosity >= 1:
        print("[INFO] Training reweighting model...")

    for group in TRAINING_GROUPS:
        if (
            group["base_method"] not in args.methods
            and group["folding_method"] not in args.methods
        ):
            continue

        if args.n_trials > 1:
            study = optuna_tuning(
                args,
                mc=mc,
                data=data,
                mcweights=mc_weights,
                sweights=sweights,
                columns=columns,
                n_trials=args.n_trials,
                weightsdir=weightsdir,
                classifier_type=group["classifier_type"],
            )
        else:
            study = None

        base_model, base_weights = group["base_train_fn"](
            args,
            sample=sample,
            columns=columns,
            study=study,
            weightsdir=weightsdir,
        )
        results[group["base_result_key"]] = {
            "model": base_model,
            "weights": base_weights,
        }

        if group["folding_method"] in args.methods:
            fold_model, fold_weights = group["fold_train_fn"](
                args,
                sample=sample,
                columns=columns,
                n_folds=args.n_folds,
                weightsdir=weightsdir,
                **{group["fold_arg_name"]: base_model},
            )
            results[group["fold_result_key"]] = {
                "model": fold_model,
                "weights": fold_weights,
            }

    if "Bins" in args.methods:
        bins, binning_weights = binning_reweight(
            args,
            sample=sample,
            columns=columns,
            n_bins=args.n_bins,
            n_neighs=args.n_neighs,
            weightsdir=weightsdir,
        )
        results["Bins"] = {"model": bins, "weights": binning_weights}

    return results


def _plot_reweighted_feature_sets(args, sample, raw_data, x_labels, plotdir, results):
    if args.verbosity >= 1:
        print("[INFO] Plotting distributions after reweighting...")

    for method in args.methods:
        if method not in results:
            continue

        suffix = METHOD_PLOT_LABELS[method]
        is_folding = method in FOLDING_METHODS
        mc_frame = raw_data["mc"] if is_folding else sample["mc_test"]
        data_frame = raw_data["data"] if is_folding else sample["data_test"]
        data_weights = raw_data["w_data"] if is_folding else sample["w_data_test"]

        plt.plot_distributions(
            args,
            mc=mc_frame,
            data=data_frame,
            mc_weights=results[method]["weights"],
            data_weights=data_weights,
            columns=args.training_vars,
            x_labels=x_labels,
            output_file=f"{plotdir}/input_features_{suffix}_weighted.png",
        )

        if args.monitoring_vars is not None:
            plt.plot_distributions(
                args,
                mc=mc_frame,
                data=data_frame,
                mc_weights=results[method]["weights"],
                data_weights=data_weights,
                columns=args.monitoring_vars,
                x_labels=x_labels,
                output_file=f"{plotdir}/other_vars_{suffix}_weighted.png",
            )


def _build_artifact_maps(args, results):
    weights = {
        method: results[method]["weights"]
        for method in args.methods
        if method in results
    }
    models = {
        method: results[method]["model"] for method in args.methods if method in results
    }
    return weights, models


def _write_throughput_outputs(args, plotdir, models):
    throughput = {
        method: getattr(model, "training_metrics_", None)
        for method, model in models.items()
        if getattr(model, "training_metrics_", None) is not None
    }
    if not throughput:
        return

    if args.verbosity >= 1:
        print("[INFO] Training throughput summary:")
        for method, metrics in throughput.items():
            print(
                f"  - {method}: "
                f"fit={metrics['fit_seconds']:.3f}s, "
                f"dataset_events={metrics['dataset_events']}, "
                f"effective_events={metrics['effective_events']}, "
                f"dataset_rate={metrics['dataset_events_per_second']:.1f} ev/s, "
                f"effective_rate={metrics['effective_events_per_second']:.1f} ev/s"
            )

    with open(f"{plotdir}/training_throughput.json", "w", encoding="ascii") as f:
        json.dump(throughput, f, indent=2)
    plt.plot_training_throughput(
        throughput=throughput,
        output_file=f"{plotdir}/training_throughput.png",
    )


def _sample_for_method(sample, method):
    if method in FOLDING_METHODS:
        return sample["mc"], sample["data"], sample["w_data"]
    return sample["mc_test"], sample["data_test"], sample["w_data_test"]


def _plot_post_training_diagnostics(
    args, plotdir, x_labels, sample, results, weights, models
):
    scores = plt.plot_roc_curve(
        sample=sample,
        weights=weights,
        methods=args.methods,
        columns=args.training_vars,
        output_file=f"{plotdir}/roc_curve.png",
    )
    if args.shap:
        scores = update_scores_with_importance(
            args,
            models=models,
            scores=scores,
            sample=sample,
            methods=args.methods,
            columns=args.training_vars,
        )

    plt.plot_classifier_output(
        scores=scores,
        weights=weights,
        methods=args.methods,
        output_file=f"{plotdir}/classifier_output.png",
    )
    plt.plot_weight_distributions(
        weights=weights, output_file=f"{plotdir}/weight_distributions.png"
    )

    for method in args.methods:
        if method not in results:
            continue

        mc_sample, data_sample, w_data = _sample_for_method(sample, method)

        if args.shap and method not in FOLDING_METHODS:
            shap_values, X_eval = scores[f"{method}_importances"]
            plt.plot_feature_importance(
                shap_values=shap_values,
                mc=X_eval,
                x_labels=x_labels,
                method=method,
                feature_names=args.training_vars,
                output_file=f"{plotdir}/feature_importance_{method}.png",
                max_display=15,
            )

        if len(args.training_vars) > 1:
            plt.plot_2d_pull_maps(
                mc=mc_sample,
                data=data_sample,
                mc_weights=weights[method],
                data_weights=w_data,
                method=method,
                columns=args.training_vars,
                x_labels=x_labels,
                output_file=f"{plotdir}/pull_map_{method}.png",
            )
            plt.plot_2d_score_maps(
                sample=sample,
                weights=weights[method],
                classifier_scores=scores[method],
                method=method,
                vars=args.training_vars,
                x_labels=x_labels,
                output_file=f"{plotdir}/score_map_{method}.png",
            )


def run_reweighting_pipeline(args, plotdir, weightsdir):
    """
    Main function to run the reweighting pipeline.
    """
    if args.verbosity >= 1:
        print("[INFO] Starting reweighting pipeline...")

    x_labels = read_x_labels(resolve_x_labels_path(args.path_xlabels))
    mc, data, mc_weights, sweights = _load_training_inputs(args)

    if len(args.training_vars) > 1:
        print("[INFO] Plotting correlation matrices...")
        plt.plot_correlation_matrix(
            args,
            df=data,
            columns=args.training_vars,
            weights=sweights,
            x_labels=x_labels,
            title=args.data_label,
            output_file=f"{plotdir}/corr_data.png",
        )
        plt.plot_correlation_matrix(
            args,
            df=mc,
            columns=args.training_vars,
            weights=mc_weights,
            x_labels=x_labels,
            title=args.mc_label,
            output_file=f"{plotdir}/corr_mc.png",
        )

    sample = train_and_test(
        mc=mc,
        data=data,
        mcweights=mc_weights,
        sweights=sweights,
        columns=args.training_vars + list(args.monitoring_vars or []),
        test_size=args.test_size,
    )
    _log_sample_split(args, sample)
    _plot_input_distributions(args, sample, x_labels, plotdir)

    if args.verbosity >= 2:
        print("[INFO] Weight summaries for MC and Data samples:")
        print(f"  - {_describe_weights('MC training', sample['w_mc_train'])}")
        print(f"  - {_describe_weights('MC testing', sample['w_mc_test'])}")
        print(f"  - {_describe_weights('Data training', sample['w_data_train'])}")
        print(f"  - {_describe_weights('Data testing', sample['w_data_test'])}")
    if args.verbosity >= 4:
        print("[INFO] Full weight arrays:")
        print(f"\tMC weights training: {sample['w_mc_train']}")
        print(f"\tMC weights testing: {sample['w_mc_test']}")
        print(f"\tData weights training: {sample['w_data_train']}")
        print(f"\tData weights testing: {sample['w_data_test']}")

    results = _train_models(
        args,
        sample=sample,
        columns=args.training_vars,
        weightsdir=weightsdir,
        mc=mc,
        data=data,
        mc_weights=mc_weights,
        sweights=sweights,
    )

    if args.verbosity >= 1:
        print("[INFO] Reweighting complete.")

    raw_data = {"mc": mc, "data": data, "w_data": sweights}
    _plot_reweighted_feature_sets(args, sample, raw_data, x_labels, plotdir, results)

    weights, models = _build_artifact_maps(args, results)
    _write_throughput_outputs(args, plotdir, models)
    _plot_post_training_diagnostics(
        args, plotdir, x_labels, sample, results, weights, models
    )


def _load_application_inputs(args):
    vars_list = list(args.application_vars)
    monitoring_vars_list = [
        v for v in (args.monitoring_vars or []) if v not in vars_list
    ]
    mc, mc_weights, mc_row_mask = load_data(
        path=args.path_mc,
        tree=args.tree_mc,
        columns=vars_list + monitoring_vars_list,
        weights_col=args.mcweights_name,
        return_mask=True,
    )
    data = sweights = None
    if args.path_data:
        data, sweights = load_data(
            path=args.path_data,
            tree=args.tree_data,
            columns=vars_list + monitoring_vars_list,
            weights_col=args.sweights_name,
        )
    if data is not None:
        _log_loaded_samples(args, mc, data, mc_weights, sweights)
    elif args.verbosity >= 3:
        print(f"[INFO] Loaded MC data with shape {mc.shape}.")
    return mc, data, mc_weights, sweights, mc_row_mask


def _load_classifier_for_application(method, prefix):
    if method in ("GB", "Folding"):
        return joblib.load(f"{prefix}.pkl")
    classifier = ONNX_APPLY_MODELS[method]()
    classifier.load(prefix)
    return classifier


def _predict_application_weights(method, classifier, model_input, mc_weights):
    if method in ("GB", "Folding"):
        return classifier.predict_weights(model_input, original_weight=mc_weights)
    return classifier.predict_weights(model_input, ow=mc_weights)


def apply_weights_pipeline(args, plotdir, weightsdir, out_weightsdir):
    """
    Main function to apply weights to the data using the trained model.
    """
    if args.verbosity >= 1:
        print("[INFO] Loading data for applying weights...")

    x_labels = read_x_labels(resolve_x_labels_path(args.path_xlabels))
    mc, data, mc_weights, sweights, mc_row_mask = _load_application_inputs(args)

    if len(args.application_vars) != len(args.training_vars):
        raise ValueError(
            "[ERROR] The number of variables for reweighting must match the number of training variables."
            " Please check the --application_vars and --training_vars arguments."
        )

    aliases = dict(zip(args.training_vars, args.application_vars))
    mc = def_aliases(mc, aliases)
    model_input = mc[args.training_vars]
    saving_vars = flatten_vars(args.training_vars)
    prefix = f"{weightsdir}/{MODEL_PREFIXES[args.method]}_" + "_".join(saving_vars)

    if args.verbosity >= 1:
        print("[INFO] Loading model and weights...")
    try:
        classifier = _load_classifier_for_application(args.method, prefix)
    except KeyError as exc:
        raise ValueError(
            f"[ERROR] Unknown model type: {args.method}."
            " ----- Currently supported methods are  'GB', 'Folding', 'ONNXGB', "
            "'ONNXFolding', 'XGB', 'XGBFolding', 'Bins', 'NN' and 'NNFolding'."
        ) from exc

    if args.verbosity >= 1:
        print("[INFO] Applying weights to MC data...")
    new_mc_weights = _predict_application_weights(
        args.method, classifier, model_input, mc_weights
    )

    w_normalized = new_mc_weights * (len(new_mc_weights) / np.sum(new_mc_weights))
    print("[INFO] Weights predicted successfully.")
    joblib.dump(
        w_normalized,
        f"{out_weightsdir}/mcweights_" + "_".join(args.application_vars) + ".pkl",
    )

    print("[INFO] Plotting distributions after applying weights...")
    plt.plot_mc_distributions(
        mc=mc,
        original_mc_weights=mc_weights,
        new_mc_weights=w_normalized,
        columns=args.application_vars,
        x_labels=x_labels,
        output_file=f"{plotdir}/mc_vars_reweighting.png",
    )
    if args.monitoring_vars is not None:
        plt.plot_mc_distributions(
            mc=mc,
            original_mc_weights=mc_weights,
            new_mc_weights=w_normalized,
            columns=args.monitoring_vars,
            x_labels=x_labels,
            output_file=f"{plotdir}/mc_other_vars_reweighting.png",
        )
    if args.path_data:
        plt.plot_distributions(
            args,
            mc=mc,
            data=data,
            mc_weights=w_normalized,
            data_weights=sweights,
            columns=args.application_vars,
            x_labels=x_labels,
            output_file=f"{plotdir}/input_features_reweighted.png",
        )
        if args.monitoring_vars is not None:
            plt.plot_distributions(
                args,
                mc=mc,
                data=data,
                mc_weights=w_normalized,
                data_weights=sweights,
                columns=args.monitoring_vars,
                x_labels=x_labels,
                output_file=f"{plotdir}/other_vars_reweighted.png",
            )

    if args.verbosity >= 1:
        print("[INFO] Weights applied to MC data. Saving new weights to output file...")

    weights_array = np.array(w_normalized, dtype=np.float32)
    save_data(
        input_path=args.path_mc,
        tree=args.tree_mc,
        output_path=args.output_path,
        output_tree=args.output_tree,
        branch=args.weights_name,
        weights=weights_array,
        ntuple_type=args.output_ntuple,
        row_mask=mc_row_mask,
    )
    if args.verbosity >= 1:
        print("[INFO] Weights saved successfully.")
