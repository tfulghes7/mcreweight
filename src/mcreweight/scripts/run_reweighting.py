import argparse
from types import SimpleNamespace
from os import makedirs, environ
from mcreweight.core import run_reweighting_pipeline
from mcreweight.io import load_config, get_from_cfg


# ---------------------------------------------------------
# Merge argparse + YAML
# ---------------------------------------------------------


def cli_or_cfg(cli_value, cfg, keys, default=None):
    if cli_value is not None:
        return cli_value
    return get_from_cfg(cfg, keys, default)


def merge_args_with_config(args, cfg):
    """
    CLI overrides YAML.
    YAML fills missing CLI values.
    """

    return SimpleNamespace(
        verbosity=args.verbosity,
        # --------------------
        # Input MC
        # --------------------
        path_mc=cli_or_cfg(args.path_mc, cfg, ["input", "mc", "path"]),
        tree_mc=cli_or_cfg(args.tree_mc, cfg, ["input", "mc", "tree"], "DecayTree"),
        mcweights_name=cli_or_cfg(
            args.mcweights_name, cfg, ["input", "mc", "weights_name"], None
        ),
        mc_label=cli_or_cfg(args.mc_label, cfg, ["input", "mc", "label"], "MC"),
        # --------------------
        # Input Data
        # --------------------
        path_data=cli_or_cfg(args.path_data, cfg, ["input", "data", "path"]),
        tree_data=cli_or_cfg(
            args.tree_data, cfg, ["input", "data", "tree"], "DecayTree"
        ),
        sweights_name=cli_or_cfg(
            args.sweights_name, cfg, ["input", "data", "sweights_name"], "sweight_sig"
        ),
        data_label=cli_or_cfg(args.data_label, cfg, ["input", "data", "label"], "Data"),
        # --------------------
        # Plotting
        # --------------------
        path_xlabels=(
            cli_or_cfg(args.path_xlabels, cfg, ["input", "path_xlabels"], None)
            or get_from_cfg(cfg, ["input", "xlabel_path"], None)
            or get_from_cfg(cfg, ["input", "path_xlabel"], None)
        ),
        # --------------------
        # Variables
        # --------------------
        training_vars=cli_or_cfg(
            args.training_vars,
            cfg,
            ["variables", "training_vars"],
            ["B_DTF_Jpsi_P", "B_DTF_Jpsi_PT", "nPVs", "nLongTracks"],
        ),
        monitoring_vars=cli_or_cfg(
            args.monitoring_vars, cfg, ["variables", "monitoring_vars"], None
        ),
        # --------------------
        # Reweighting
        # --------------------
        sample=cli_or_cfg(args.sample, cfg, ["reweighting", "sample"], "bd_jpsikst_ee"),
        methods=cli_or_cfg(
            args.methods,
            cfg,
            ["reweighting", "methods"],
            ["GB", "XGB", "NN", "Folding", "XGBFolding", "NNFolding", "Bins"],
        ),
        transform=cli_or_cfg(args.transform, cfg, ["reweighting", "transform"], None),
        n_trials=cli_or_cfg(args.n_trials, cfg, ["reweighting", "n_trials"], 10),
        test_size=cli_or_cfg(args.test_size, cfg, ["reweighting", "test_size"], 0.3),
        n_folds=cli_or_cfg(args.n_folds, cfg, ["reweighting", "n_folds"], 10),
        n_bins=cli_or_cfg(args.n_bins, cfg, ["reweighting", "n_bins"], 10),
        n_neighs=cli_or_cfg(args.n_neighs, cfg, ["reweighting", "n_neighs"], 3),
        reweight_validation_fraction=cli_or_cfg(
            args.reweight_validation_fraction,
            cfg,
            ["reweighting", "reweight_validation_fraction"],
            0.2,
        ),
        reweight_early_stopping_rounds=cli_or_cfg(
            args.reweight_early_stopping_rounds,
            cfg,
            ["reweighting", "reweight_early_stopping_rounds"],
            5,
        ),
        reweight_metric_every=cli_or_cfg(
            args.reweight_metric_every, cfg, ["reweighting", "reweight_metric_every"], 1
        ),
        folding_aggregation=cli_or_cfg(
            args.folding_aggregation,
            cfg,
            ["reweighting", "folding_aggregation"],
            "weighted_geometric",
        ),
        shap=args.shap or get_from_cfg(cfg, ["reweighting", "shap"], False),
        # ---------------------
        # Output
        # ---------------------
        weightsdir=cli_or_cfg(args.weightsdir, cfg, ["output", "weightsdir"], None),
        plotdir=cli_or_cfg(args.plotdir, cfg, ["output", "plotdir"], "plots"),
    )


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run MC reweighting using YAML and/or CLI arguments"
    )

    parser.add_argument("--config", help="YAML configuration file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running the pipeline",
    )
    parser.add_argument(
        "--verbosity",
        help="Level of verbosity for logging",
        choices=[1, 2, 3, 4],
        default=1,
        type=int,
    )

    # MC
    parser.add_argument(
        "--path-mc",
        help="Path to input MC ROOT file (overrides YAML config)",
        nargs="+",
    )
    parser.add_argument(
        "--tree-mc", help="Name of the MC TTree (overrides YAML config)"
    )
    parser.add_argument(
        "--mcweights-name", help="Name of the MC weights branch (overrides YAML config)"
    )
    parser.add_argument(
        "--mc-label", help="Label for the MC sample (overrides YAML config)"
    )

    # Data
    parser.add_argument(
        "--path-data",
        help="Path to input Data ROOT file (overrides YAML config)",
        nargs="+",
    )
    parser.add_argument(
        "--tree-data", help="Name of the Data TTree (overrides YAML config)"
    )
    parser.add_argument(
        "--sweights-name",
        help="Name of the signal weights branch (overrides YAML config)",
    )
    parser.add_argument(
        "--data-label", help="Label for the Data sample (overrides YAML config)"
    )

    # Variables
    parser.add_argument(
        "--training-vars",
        nargs="+",
        help="Variables to use for reweighting training, e.g. --training-vars B_P B_PT nPVs nLongTracks (overrides YAML config)",
    )
    parser.add_argument(
        "--monitoring-vars",
        nargs="+",
        help="Variables to monitor for reweighting performance, e.g. --monitoring-vars pt eta (overrides YAML config)",
    )

    # Reweighting
    parser.add_argument(
        "--sample", help="Sample name for the dataset (overrides YAML config)"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Reweighting methods to apply, including GB, Folding, ONNXGB, ONNXFolding, XGB, XGBFolding, NN, NNFolding, and Bins (overrides YAML config)",
    )
    parser.add_argument(
        "--transform",
        help="Transformation to apply to input features for reweighting (overrides YAML config).",
        choices=["quantile", "yeo-johnson", "signed-log", "scaler"],
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        help="Number of trials for the gradient boosting reweighting method (overrides YAML config)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        help="Proportion of the dataset to include in the test split (overrides YAML config)",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        help="Number of folds for k-folding reweighting (overrides YAML config)",
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        help="Number of bins for binning reweighting (overrides YAML config)",
    )
    parser.add_argument(
        "--n_neighs",
        type=int,
        help="Number of nearest neighbors for binning reweighting (overrides YAML config)",
    )
    parser.add_argument(
        "--reweight-validation-fraction",
        dest="reweight_validation_fraction",
        type=float,
        help="Validation fraction used by iterative ONNX reweighters for early stopping (overrides YAML config)",
    )
    parser.add_argument(
        "--reweight-early-stopping-rounds",
        dest="reweight_early_stopping_rounds",
        type=int,
        help="Number of validation checks without improvement before stopping iterative ONNX reweighters (overrides YAML config)",
    )
    parser.add_argument(
        "--reweight-metric-every",
        dest="reweight_metric_every",
        type=int,
        help="Evaluate the iterative ONNX validation metric every N stages (overrides YAML config)",
    )
    parser.add_argument(
        "--folding-aggregation",
        choices=["weighted_geometric", "geometric", "median"],
        help="Aggregation strategy for ONNX folding predictions (overrides YAML config)",
    )
    parser.add_argument(
        "--shap",
        action="store_true",
        help="Whether to compute SHAP values for the reweighting model (overrides YAML config)",
    )

    # Output
    parser.add_argument(
        "--weightsdir",
        help="Root directory where training artifacts will be written; a '<sample>/' subdirectory is created automatically (overrides MCREWEIGHTS_DATA_ROOT and YAML config)",
    )
    parser.add_argument(
        "--plotdir", help="Directory to save plots (overrides YAML config)"
    )

    # Plotting
    parser.add_argument(
        "--path-xlabels",
        help="Path to YAML file containing x-axis labels for plots (overrides YAML config)",
    )

    return parser


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------


def main():

    parser = build_parser()
    args = parser.parse_args()

    cfg = {}
    if args.config:
        cfg = load_config(args.config)

    ns = merge_args_with_config(args, cfg)

    if args.verbosity >= 2:
        print("[INFO] Configuration:")
        for k, v in vars(ns).items():
            print(f"\t{k}: {v}")

    # --------------------------------
    # Directory handling
    # --------------------------------

    plotdir = f"{ns.plotdir}/{ns.sample}"

    if ns.weightsdir is None:
        ns.weightsdir = environ.get("MCREWEIGHTS_DATA_ROOT")

    if not ns.weightsdir:
        raise ValueError("weightsdir not set and MCREWEIGHTS_DATA_ROOT is undefined")

    weightsdir = f"{ns.weightsdir}/{ns.sample}"

    makedirs(plotdir, exist_ok=True)
    makedirs(weightsdir, exist_ok=True)

    # --------------------------------
    # Dry run
    # --------------------------------

    if args.dry_run:
        print("Configuration OK")
        print(vars(ns))
        return

    # --------------------------------
    # Run pipeline
    # --------------------------------

    run_reweighting_pipeline(
        ns,
        plotdir=plotdir,
        weightsdir=weightsdir,
    )


if __name__ == "__main__":
    main()
