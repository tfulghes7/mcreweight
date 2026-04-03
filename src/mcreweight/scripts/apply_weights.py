import argparse
from types import SimpleNamespace
from os import makedirs, environ
from mcreweight.core import apply_weights_pipeline
from mcreweight.io import load_config, get_from_cfg


def cli_or_cfg(cli_value, cfg, keys, default=None):
    if cli_value is not None:
        return cli_value
    return get_from_cfg(cfg, keys, default)


def merge_args_with_config(args, cfg):

    return SimpleNamespace(
        # --------------------
        # Input MC
        # --------------------
        path_mc=cli_or_cfg(args.path_mc, cfg, ["input", "mc", "path"]),
        tree_mc=cli_or_cfg(args.tree_mc, cfg, ["input", "mc", "tree"], "DecayTree"),
        mcweights_name=cli_or_cfg(
            args.mcweights_name, cfg, ["input", "mc", "weights_branch"], None
        ),
        # --------------------
        # Input Data
        # --------------------
        path_data=cli_or_cfg(args.path_data, cfg, ["input", "data", "path"], None),
        tree_data=cli_or_cfg(
            args.tree_data, cfg, ["input", "data", "tree"], "DecayTree"
        ),
        sweights_name=(
            cli_or_cfg(
                args.sweights_name, cfg, ["input", "data", "sweights_name"], None
            )
            or get_from_cfg(cfg, ["input", "data", "sweights_branch"], "sweight_sig")
        ),
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
        application_vars=(
            cli_or_cfg(
                args.application_vars, cfg, ["variables", "application_vars"], None
            )
            or get_from_cfg(
                cfg,
                ["variables", "vars"],
                ["B_DTF_Jpsi_P", "B_DTF_Jpsi_PT", "nPVs", "nLongTracks"],
            )
        ),
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
        method=cli_or_cfg(args.method, cfg, ["reweighting", "method"], "XGB"),
        training_sample=cli_or_cfg(
            args.training_sample,
            cfg,
            ["reweighting", "training_sample"],
            "bd_jpsikst_ee",
        ),
        application_sample=cli_or_cfg(
            args.application_sample,
            cfg,
            ["reweighting", "application_sample"],
            "bd_jpsikst_ee",
        ),
        weightsdir=cli_or_cfg(
            args.weightsdir, cfg, ["reweighting", "weightsdir"], None
        ),
        plotdir=cli_or_cfg(args.plotdir, cfg, ["reweighting", "plotdir"], "plots"),
        verbosity=args.verbosity,
        # --------------------
        # Output
        # --------------------
        output_path=(
            cli_or_cfg(args.output_path, cfg, ["output", "output_path"], None)
            or get_from_cfg(cfg, ["output", "path"])
        ),
        output_ntuple=(
            cli_or_cfg(args.output_ntuple, cfg, ["output", "output_ntuple"], None)
            or get_from_cfg(cfg, ["output", "ntuple"], "TTree")
        ),
        output_tree=(
            cli_or_cfg(args.output_tree, cfg, ["output", "output_tree"], None)
            or get_from_cfg(cfg, ["output", "tree"], "DecayTree")
        ),
        weights_name=(
            cli_or_cfg(args.weights_name, cfg, ["output", "weights_name"], None)
            or get_from_cfg(cfg, ["output", "weights_branch"], "weights")
        ),
    )


# ---------------------------------------------------------
# CLI definition
# ---------------------------------------------------------
def build_parser():

    parser = argparse.ArgumentParser(
        description="Apply MC reweighting using YAML and/or CLI arguments"
    )

    parser.add_argument("--config", help="YAML configuration file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running the application pipeline",
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
        "--path-mc", help="Path to MC file (overrides config)", nargs="+"
    )
    parser.add_argument("--tree-mc", help="Name of the MC tree (overrides config)")
    parser.add_argument(
        "--mcweights-name", help="Name of the MC weights branch (overrides config)"
    )

    # Data
    parser.add_argument(
        "--path-data", help="Path to data file (overrides config)", nargs="+"
    )
    parser.add_argument("--tree-data", help="Name of the data tree (overrides config)")
    parser.add_argument(
        "--sweights-name", help="Name of the data sweights branch (overrides config)"
    )

    # Variables
    parser.add_argument(
        "--vars",
        dest="application_vars",
        nargs="+",
        help="List of variables to reweight (overrides config)",
    )
    parser.add_argument(
        "--training-vars",
        nargs="+",
        help="List of training variables (overrides config)",
    )
    parser.add_argument(
        "--monitoring-vars",
        nargs="+",
        help="List of monitoring variables (overrides config)",
    )

    # Reweighting
    parser.add_argument(
        "--method",
        help="Reweighting method (overrides config)",
        choices=[
            "GB",
            "Folding",
            "ONNXGB",
            "ONNXFolding",
            "XGB",
            "XGBFolding",
            "NN",
            "NNFolding",
            "Bins",
        ],
    )
    parser.add_argument(
        "--training-sample", help="Name of the training sample (overrides config)"
    )
    parser.add_argument(
        "--application-sample", help="Name of the application sample (overrides config)"
    )
    parser.add_argument(
        "--weightsdir",
        help="Root directory containing trained artifacts; '<training-sample>/' is read and '<application-sample>/' is written automatically (overrides config)",
    )
    parser.add_argument(
        "--plotdir",
        help="Root directory for plots; an '<application-sample>/' subdirectory is created automatically (overrides config)",
    )

    # Output
    parser.add_argument(
        "--output-path", help="Path for output files (overrides config)"
    )
    parser.add_argument(
        "--output-ntuple", help="Name of the output ntuple (overrides config)"
    )
    parser.add_argument(
        "--output-tree", help="Name of the output tree (overrides config)"
    )
    parser.add_argument(
        "--weights-name", help="Name of the output weights branch (overrides config)"
    )

    # Plotting
    parser.add_argument(
        "--path-xlabels", help="Path to xlabels file (overrides config)"
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

    # --------------------------------
    # Directory handling
    # --------------------------------
    plotdir = f"{ns.plotdir}/{ns.application_sample}"

    if ns.weightsdir is None:
        ns.weightsdir = environ.get("MCREWEIGHTS_DATA_ROOT")

    if not ns.weightsdir:
        raise ValueError("weightsdir not set and MCREWEIGHTS_DATA_ROOT is undefined")

    weightsdir = f"{ns.weightsdir}/{ns.training_sample}"
    out_weightsdir = f"{ns.weightsdir}/{ns.application_sample}"

    makedirs(plotdir, exist_ok=True)
    makedirs(out_weightsdir, exist_ok=True)

    # --------------------------------
    # Dry run
    # --------------------------------
    if args.dry_run:
        print("Configuration OK")
        print(vars(ns))
        return

    # --------------------------------
    # Run
    # --------------------------------
    apply_weights_pipeline(
        ns,
        plotdir=plotdir,
        weightsdir=weightsdir,
        out_weightsdir=out_weightsdir,
    )


if __name__ == "__main__":
    main()
