import itertools
import logging
import math
import os

import matplotlib.pyplot as plt


import numpy as np
import seaborn as sns
import shap
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import binned_statistic_2d

from .utils import fit_transform, apply_transform
from mcreweight.utils.utils import (
    evaluate_reweighting,
    get_scores,
    weighted_corr_matrix,
    weighted_ks_statistic,
)

# Suppress "findfont: Font family '...' not found" messages that come from
# matplotlib scanning system fonts referenced by fontconfig but not installed
# (e.g. Times New Roman on Linux).  DejaVu Sans is used as the fallback and
# renders correctly; the warnings are not actionable.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

_STYLE = "plain"
_SAMPLE_LABEL = None
_EXTRA_LABEL = None


def configure_style(style="plain", sample_label=None, extra_label=None):
    """Configure the plot style and optional sample label for the session.

    Args:
        style (str): ``"plain"`` (default custom LHCb-like style) or
            ``"LHCb"`` (mplhep LHCb2 style with experiment label on each plot).
        sample_label (str | None): Text placed in the top-right of each frame
            when style is ``"LHCb"``. Ignored for ``"plain"``.
        extra_label (str | None): Text placed in italic immediately after "LHCb"
            on the top-left (e.g. ``"Simulation"`` or ``"Preliminary"``).
            Ignored for ``"plain"``.
    """
    global _STYLE, _SAMPLE_LABEL, _EXTRA_LABEL
    _STYLE = style
    _SAMPLE_LABEL = sample_label
    _EXTRA_LABEL = extra_label
    _apply_style()


def _apply_style():
    """Apply the currently configured style."""
    if _STYLE == "LHCb":
        import mplhep as hep

        # hep.style.use triggers findfont log messages for fonts not installed on
        # this system (Tex Gyre Termes → Times New Roman fallback).  Silence the
        # font-manager logger for that call; the rcParams block below replaces all
        # serif/mathtext references with DejaVu Sans so rendering stays clean.
        _fm_log = logging.getLogger("matplotlib.font_manager")
        _prev_level = _fm_log.level
        _fm_log.setLevel(logging.ERROR)
        hep.style.use("LHCb2")
        _fm_log.setLevel(_prev_level)

        plt.rcParams.update(
            {
                # TeX Gyre Termes is metric-compatible with Times New Roman and is
                # installed on this system.  Use the exact capitalisation so
                # matplotlib finds it without falling back to Times New Roman.
                "font.family": "serif",
                "font.serif": [
                    "TeX Gyre Termes",
                    "Liberation Serif",
                    "Nimbus Roman",
                    "DejaVu Serif",
                ],
                # STIX fonts (bundled with matplotlib) provide math symbols in a
                # Times-compatible style — no external font needed for $...$  text.
                "mathtext.fontset": "stix",
                # Restore readable sizes (LHCb2 ships labelsize=32, markersize=16)
                "font.size": 14,
                "figure.dpi": 100,
                "axes.labelsize": 26,
                "xtick.labelsize": 24,
                "ytick.labelsize": 24,
                "legend.fontsize": 24,
                "legend.title_fontsize": 24,
                "lines.markersize": 5,
                "lines.linewidth": 1.5,
                "lines.markeredgewidth": 0.8,
                "errorbar.capsize": 2,
            }
        )
    else:
        set_lhcb_style()


def _add_labels(
    ax, x_min_lhcb=0.0, y_min_lhcb=1.02, x_min_sample=1.0, y_min_sample=1.02
):
    """Place "LHCb" outside the frame on the top-left and the optional sample
    label outside the frame on the top-right.

    Uses ``ax.text`` directly so the font (TeX Gyre Termes from rcParams) and
    weight/style match the rest of the plot exactly.
    No-op when style is not ``"LHCb"``.
    """
    if _STYLE != "LHCb":
        return

    label_size = plt.rcParams.get("axes.labelsize", 26)

    lhcb_text = ax.text(
        x_min_lhcb,
        y_min_lhcb,
        "LHCb",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=label_size,
        clip_on=False,
    )
    if _EXTRA_LABEL:
        ax.annotate(
            f" {_EXTRA_LABEL}",
            xycoords=lhcb_text,
            xy=(1, 0),
            xytext=(0, 0),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=label_size,
            fontstyle="italic",
            clip_on=False,
        )
    if _SAMPLE_LABEL:
        ax.text(
            x_min_sample,
            y_min_sample,
            _SAMPLE_LABEL,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=label_size,
            clip_on=False,
        )


MC_COLOR = "#c62828"
MC_DARK_COLOR = "#7f0000"
DATA_COLOR = "#000000"
NEUTRAL_COLOR = "#666666"
METHOD_COLORS = {
    "GB": "#c62828",
    "Folding": "#ef6c00",
    "ONNXGB": "#1565c0",
    "ONNXFolding": "#00897b",
    "XGB": "#6a1b9a",
    "XGBFolding": "#8e24aa",
    "NN": "#2e7d32",
    "NNFolding": "#558b2f",
    "Bins": "#795548",
}


def _label_for(x_labels, feature_name):
    return x_labels.get(feature_name, feature_name)


def _hide_unused_grouped_axes(axes, n_plots, n_cols):
    """Hide only subplot cells not occupied by grouped variable panels."""
    used_positions = set()
    for idx in range(n_plots):
        row = (idx // n_cols) * 3
        col = idx % n_cols
        used_positions.update({(row, col), (row + 1, col), (row + 2, col)})

    for row in range(axes.shape[0]):
        for col in range(axes.shape[1]):
            if (row, col) not in used_positions:
                axes[row, col].axis("off")


def _subplot_column_count(n_plots):
    """Choose a compact, readable subplot grid width."""
    if n_plots == 4:
        return 2
    if n_plots >= 10:
        return 5
    return min(3, n_plots)


def _reshape_axes_grid(axes, n_rows, n_cols):
    """Return subplot axes as a consistent 2D ndarray."""
    return np.atleast_2d(np.array(axes, dtype=object)).reshape(n_rows, n_cols)


def set_lhcb_style(grid=True, size=12, usetex=False):
    """
    Set matplotlib plotting style close to "official" LHCb style
    (TeX Gyre Termes serif font, inward ticks on all sides, minor ticks, light grid).
    """
    plt.rc("font", family="serif", size=size)
    plt.rc(
        "font",
        **{
            "serif": [
                "TeX Gyre Termes",
                "Liberation Serif",
                "Nimbus Roman",
                "DejaVu Serif",
            ]
        },
    )
    plt.rc("text", usetex=usetex)
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams.update(
        {
            "figure.max_open_warning": 40,
            "axes.linewidth": 1.3,
            "axes.grid": grid,
            "grid.alpha": 0.3,
            "axes.axisbelow": False,
            "xtick.major.width": 1,
            "ytick.major.width": 1,
            "xtick.minor.width": 1,
            "ytick.minor.width": 1,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.bottom": True,
            "xtick.top": True,
            "ytick.left": True,
            "ytick.right": True,
            "axes.titlesize": 26,
            "axes.labelsize": 26,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
        }
    )


try:
    _apply_style()
except Exception:
    pass


def plot_correlation_matrix(args, df, columns, weights, x_labels, title, output_file):
    """
    Plot a correlation matrix for the given DataFrame columns.

    Args:
        args (argparse.Namespace): Command line arguments containing verbosity flag.
        df (pd.DataFrame): DataFrame containing the data.
        columns (list): List of column names to include in the correlation matrix.
        weights (np.ndarray, optional): Weights for the correlation calculation. If None, unweighted correlation is used.
        x_labels (dict): Mapping of column names to x-axis labels for the plot.
        title (str): Title of the plot.
        output_file (str): Path to save the output plot.
    """
    _apply_style()
    if args.verbosity >= 3:
        print(
            f"[INFO] Computing correlation matrix for columns: {columns} with weights: {weights is not None}"
        )
        print(
            f"[INFO] Data sample size: {len(df)}, Weights sample size: {len(weights) if weights is not None else 'N/A'}"
        )
    corr_mode = "unweighted"
    if weights is not None:
        corr, corr_mode = weighted_corr_matrix(df, columns, weights)
        if args.verbosity >= 2 and corr_mode == "absolute":
            print(
                "[INFO] Correlation matrix fallback: using absolute weights because "
                "signed weights produced a non-positive variance."
            )
    else:
        corr = df[columns].corr()
    plt.figure(figsize=(16, 12))
    ax_corr = plt.gca()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        cbar_kws={"shrink": 0.75},
        xticklabels=[_label_for(x_labels, col) for col in columns],
        yticklabels=[_label_for(x_labels, col) for col in columns],
        annot_kws={"size": 20},
    )
    ax_corr.set_xticks(range(len(columns)))
    ax_corr.set_xticklabels(
        [_label_for(x_labels, col) for col in columns], fontsize=22, rotation=45
    )
    plt.yticks(fontsize=22)
    if corr_mode == "absolute":
        plt.title(f"{title} (|weights| correlation)")
    else:
        plt.title(title)
    _add_labels(ax_corr)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def plot_distributions(
    args,
    mc,
    data,
    mc_weights,
    data_weights,
    columns,
    x_labels,
    output_file,
    transform=None,
    x_edges=None,
    pull_clip=5,
):
    """
    Plot distributions with pull plots, handling MC and Data with different statistics.
    Histograms are normalized as densities, and pulls are correctly computed.

    Args:
        args (argparse.Namespace): Command line arguments containing verbosity flag.
        mc, data (pd.DataFrame): MC and Data samples.
        mc_weights, data_weights (np.ndarray): Weights for MC and Data.
        columns (list): Columns to plot.
        x_labels (dict): Mapping column names -> x-axis labels.
        output_file (str): Path to save figure.
        transform (callable, optional): Transformation function to apply to the data.
        x_edges (dict, optional): Column -> bin edges mapping.
        pull_clip (float): Maximum absolute value for pull display.
    """
    _apply_style()
    if args.verbosity >= 3:
        print(f"[INFO] Plotting columns: {columns}")
        print(f"[INFO] MC size: {len(mc)}, Data size: {len(data)}")

    n_plots = len(columns)
    n_cols = _subplot_column_count(n_plots)
    n_rows = math.ceil(n_plots / n_cols)
    grid_rows = n_rows * 3

    fig, axes = plt.subplots(
        grid_rows,
        n_cols,
        figsize=(10 * n_cols, 8 * n_rows),
        gridspec_kw={"height_ratios": [3.0, 1.0, 0.55] * n_rows},
        constrained_layout=False,
    )
    axes = _reshape_axes_grid(axes, grid_rows, n_cols)

    if transform is not None:
        # build matrices for transform
        X_mc = mc[columns].to_numpy()
        X_data = data[columns].to_numpy()
        mc_finite = np.isfinite(X_mc).all(axis=1)
        data_finite = np.isfinite(X_data).all(axis=1)
        X_mix = np.vstack([X_mc[mc_finite], X_data[data_finite]])
        transformed = fit_transform(X_mix, transform)

        X_mc_tr = apply_transform(X_mc, transformed)
        X_data_tr = apply_transform(X_data, transformed)

        if args.verbosity >= 3:
            print(
                f"[INFO] Before transformation: MC shape: {X_mc.shape}, Data shape: {X_data.shape}"
            )
            print(f"[INFO] Applied transformation: {transform}")
            print(
                f"[INFO] Transformed MC shape: {X_mc_tr.shape}, Transformed Data shape: {X_data_tr.shape}"
            )

    for idx, col_name in enumerate(columns):
        row = (idx // n_cols) * 3
        col = idx % n_cols

        if transform is not None:
            x_mc = X_mc_tr[:, idx]
            x_data = X_data_tr[:, idx]

            mc_mask = np.isfinite(x_mc)
            data_mask = np.isfinite(x_data)
        else:
            x_mc = mc[col_name].to_numpy()
            x_data = data[col_name].to_numpy()

            mc_mask = np.isfinite(x_mc)
            data_mask = np.isfinite(x_data)

        if x_edges and col_name in x_edges:
            bins = x_edges[col_name]
        else:
            x_all = np.hstack([x_mc[mc_mask], x_data[data_mask]])
            xlim = np.percentile(x_all, [0.01, 99.99])
            if xlim[0] == xlim[1]:
                xlim[1] = xlim[0] + 1e-10
            bins = np.linspace(xlim[0], xlim[1], 51)

        bin_widths = np.diff(bins)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # Histogram counts using the chosen arrays + matching weight masks
        mc_w = mc_weights[mc_mask]
        data_w = data_weights[data_mask]

        mc_counts, _ = np.histogram(x_mc[mc_mask], bins=bins, weights=mc_w)
        data_counts, _ = np.histogram(x_data[data_mask], bins=bins, weights=data_w)

        mc_var, _ = np.histogram(x_mc[mc_mask], bins=bins, weights=mc_w**2)
        data_var, _ = np.histogram(x_data[data_mask], bins=bins, weights=data_w**2)

        # Convert to densities
        mc_sum = mc_w.sum()
        data_sum = data_w.sum()
        mc_density = mc_counts / (mc_sum * bin_widths)
        data_density = data_counts / (data_sum * bin_widths)
        mc_density_var = mc_var / (mc_sum**2 * bin_widths**2)
        data_density_var = data_var / (data_sum**2 * bin_widths**2)

        # Pulls
        total_unc = np.sqrt(mc_density_var + data_density_var)
        pulls = np.divide(
            data_density - mc_density,
            total_unc,
            out=np.zeros_like(data_density),
            where=total_unc > 0,
        )
        pulls = np.clip(pulls, -pull_clip, pull_clip)

        # --- Main plot ---
        ax_main = axes[row, col]
        ax_main.step(
            bin_centers,
            mc_density,
            where="mid",
            label=args.mc_label,
            linewidth=1.5,
            color=MC_COLOR,
        )
        ax_main.errorbar(
            bin_centers,
            mc_density,
            yerr=np.sqrt(mc_density_var),
            fmt="none",
            ecolor=MC_COLOR,
            elinewidth=1,
            capsize=2,
        )
        ax_main.errorbar(
            bin_centers,
            data_density,
            yerr=np.sqrt(data_density_var),
            fmt="o",
            color=DATA_COLOR,
            markerfacecolor=DATA_COLOR,
            markeredgecolor=DATA_COLOR,
            label=args.data_label,
            markersize=5,
            elinewidth=1,
            capsize=3,
        )
        ax_main.set_ylabel("A.U.")
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)
        ax_main.set_xticklabels([])
        _add_labels(ax_main)

        # --- Pull plot ---
        ax_pull = axes[row + 1, col]
        ax_pull.axhline(0, color=NEUTRAL_COLOR, linestyle="--")
        ax_pull.bar(bin_centers, pulls, width=bin_widths, color=MC_COLOR, alpha=0.6)
        ax_pull.set_ylabel("Pull")
        ax_pull.set_xlabel(_label_for(x_labels, col_name))
        ax_pull.set_ylim(-pull_clip, pull_clip)
        ax_pull.grid(True, alpha=0.3)

        # Spacer row between grouped panels
        axes[row + 2, col].axis("off")

    _hide_unused_grouped_axes(axes, len(columns), n_cols)

    fig.subplots_adjust(
        left=0.10,
        right=0.96,
        top=0.93,
        bottom=0.10,
        wspace=0.38,
        hspace=0.18,
    )

    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_mc_distributions(
    mc,
    original_mc_weights,
    new_mc_weights,
    columns,
    x_labels,
    output_file,
    x_edges=None,
):
    """
    Plot distributions of MC data with weights.

    Args:
        mc (pd.DataFrame): MC data.
        original_mc_weights (np.ndarray): Original weights for the MC data.
        new_mc_weights (np.ndarray): Weights for the MC data.
        columns (list): List of column names to plot.
        x_labels (dict): Dictionary mapping column names to x-axis labels.
        output_file (str): Path to save the output plot.
        x_edges (dict, optional): Dictionary mapping column names to bin edges for histogramming.
    """
    _apply_style()
    hist_settings = dict(bins=50, histtype="step", linewidth=1.5)
    n_cols = _subplot_column_count(len(columns))
    n_rows = math.ceil(len(columns) / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(10 * n_cols, 6.5 * n_rows), constrained_layout=True
    )
    axes = _reshape_axes_grid(axes, n_rows, n_cols)

    for idx, column in enumerate(columns):
        row = idx // n_cols
        col = idx % n_cols

        # Determine binning
        if x_edges and column in x_edges:
            bins = x_edges[column]
        else:
            xlim = np.percentile(mc[column], [0.01, 99.99])
            bins = np.linspace(xlim[0], xlim[1], hist_settings["bins"] + 1)

        # Histogramming
        hist_orig, _ = np.histogram(mc[column], bins=bins, weights=original_mc_weights)
        hist_new, _ = np.histogram(mc[column], bins=bins, weights=new_mc_weights)

        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        ax_main = axes[row, col]
        step_settings = {
            k: v for k, v in hist_settings.items() if k not in ["bins", "histtype"]
        }
        ax_main.step(
            bin_centers,
            hist_orig,
            where="mid",
            label="Original MC",
            color=MC_COLOR,
            **step_settings,
        )
        ax_main.step(
            bin_centers,
            hist_new,
            where="mid",
            label="Reweighted MC",
            linestyle="--",
            color=MC_DARK_COLOR,
            **step_settings,
        )
        ax_main.set_ylabel("A.U.")
        ax_main.set_xlabel(_label_for(x_labels, column))
        ax_main.legend()
        _add_labels(ax_main)

    # Hide unused subplots
    total_plots = len(columns)
    for i in range(total_plots, axes.shape[0] * axes.shape[1]):
        axes.flat[i].axis("off")

    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def plot_training_throughput(throughput, output_file):
    """
    Plot training throughput metrics for each method.

    Args:
        throughput (dict): Mapping method -> throughput metric dictionary.
        output_file (str): Output file path.
    """
    if not throughput:
        return

    _apply_style()

    items = sorted(
        throughput.items(),
        key=lambda item: item[1].get("dataset_events_per_second", 0.0),
        reverse=True,
    )
    methods = [method for method, _ in items]
    dataset_rates = [
        metrics.get("dataset_events_per_second", 0.0) for _, metrics in items
    ]

    y = np.arange(len(methods))
    fig_height = max(5, 0.9 * len(methods) + 2)

    colors = [METHOD_COLORS.get(m, MC_COLOR) for m in methods]

    fig, ax = plt.subplots(figsize=(12, fig_height), constrained_layout=True)
    ax.barh(y, dataset_rates, height=0.6, color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.invert_yaxis()
    ax.set_xlabel("Training Throughput [events/s]")
    ax.set_title("Training Throughput by Method")
    ax.grid(True, axis="x", alpha=0.3)
    _add_labels(ax)

    xmax = max(dataset_rates, default=0.0)
    if xmax > 0:
        ax.set_xlim(0, xmax * 1.15)

    for ypos, rate in zip(y, dataset_rates):
        ax.text(rate, ypos, f" {rate:.1f}", va="center", ha="left", fontsize=14)

    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_training_memory(memory_profile, output_file):
    """
    Plot training memory metrics for each method.

    Args:
        memory_profile (dict): Mapping method -> memory metric dictionary.
        output_file (str): Output file path.
    """
    if not memory_profile:
        return

    _apply_style()

    items = sorted(
        memory_profile.items(),
        key=lambda item: item[1].get("rss_peak_bytes") or 0,
        reverse=True,
    )
    methods = [method for method, _ in items]
    peak_mb = [(metrics.get("rss_peak_bytes") or 0) / (1024**2) for _, metrics in items]

    y = np.arange(len(methods))
    fig_height = max(5, 0.9 * len(methods) + 2)

    colors = [METHOD_COLORS.get(m, MC_COLOR) for m in methods]

    fig, ax = plt.subplots(figsize=(12, fig_height), constrained_layout=True)
    ax.barh(y, peak_mb, height=0.6, color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.invert_yaxis()
    ax.set_xlabel("Memory [MB]")
    ax.set_title("Training Memory by Method")
    ax.grid(True, axis="x", alpha=0.3)
    _add_labels(ax)

    xmax = max(peak_mb, default=0.0)
    if xmax > 0:
        ax.set_xlim(0, xmax * 1.15)

    for ypos, value in zip(y, peak_mb):
        ax.text(value, ypos, f" {value:.1f}", va="center", ha="left", fontsize=14)

    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_roc_curve(sample, weights, methods, columns, output_file):
    """
    Plot ROC curve for the different reweighting methods.

    Args:
        sample (dict): Dictionary containing MC and Data samples and their weights.
        weights (dict): Dictionary containing weights for each method (GB, Folding, XGB, k-Folding, Bins, NN).
        methods (list): List of methods to include in the plot.
        columns (list): List of column names to use for plotting.
        output_file (str): Path to save the output plot.

    Returns:
        scores: Dictionaries containing classifier scores for each method.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(16, 12))

    evaluate_reweighting(
        sample["mc_test"][columns].values,
        sample["data_test"][columns].values,
        sample["w_mc_test"],
        sample["w_data_test"],
        "Unweighted",
        ax,
        {"MC": None, "Data": None},
    )

    scores = get_scores(
        sample=sample, weights=weights, methods=methods, columns=columns, ax=ax
    )

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    _add_labels(ax)
    plt.savefig(output_file)
    print(f"[INFO] ROC curve saved to: {output_file}")
    plt.close()

    return scores


def plot_classifier_output(
    scores, weights, methods, output_file, min_score=0.0, max_score=1.0
):
    """
    Produce two sets of classifier output plots.

    1. ``output_file``: all methods' MC score distributions overlaid on a
       single axes — no Target line — for a quick visual comparison.
    2. ``output_file`` with ``_{method}`` inserted before the extension: one
       file per method showing MC (solid) and Target (dashed) from the *same*
       per-method classifier, so the KS comparison is self-consistent.
    """
    active = [m for m in methods if scores.get(m, {}).get("MC") is not None]
    if not active:
        return

    hist_kw = dict(bins=50, density=True, range=(min_score, max_score))
    stem, ext = os.path.splitext(output_file)

    # ── 1. Combined overlay (MC only, all methods) ───────────────────────────
    _apply_style()
    fig, ax = plt.subplots(figsize=(16, 12))
    for method in active:
        score_mc = scores[method]["MC"]
        w_mc = weights.get(method, np.ones_like(score_mc))
        ax.hist(
            score_mc,
            weights=w_mc,
            alpha=0.6,
            color=METHOD_COLORS.get(method, MC_COLOR),
            label=method,
            **hist_kw,
        )
    ax.set_xlabel("Classifier output")
    ax.set_ylabel("Density")
    ax.set_title("Classifier score distributions")
    ax.legend()
    _add_labels(ax)
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Classifier output plot saved to: {output_file}")

    # ── 2. Per-method: MC vs Target from the same classifier ─────────────────
    for method in active:
        _apply_style()
        fig, ax = plt.subplots(figsize=(12, 9))

        score_mc = scores[method]["MC"]
        score_target = scores[method]["Data"]
        w_mc = weights.get(method, np.ones_like(score_mc))
        w_data_test = weights.get("Data")
        w_data_full = weights.get("DataFull")
        if w_data_full is not None and len(w_data_full) == len(score_target):
            w_target = w_data_full
        elif w_data_test is not None and len(w_data_test) == len(score_target):
            w_target = w_data_test
        else:
            w_target = np.ones_like(score_target, dtype=float)
        method_color = METHOD_COLORS.get(method, MC_COLOR)

        ks_val = weighted_ks_statistic(score_mc, score_target, w1=w_mc, w2=w_target)

        ax.hist(
            score_mc,
            weights=w_mc,
            alpha=0.6,
            color=method_color,
            label=f"{method} (KS={ks_val:.3f})",
            **hist_kw,
        )
        ax.hist(
            score_target,
            weights=w_target,
            histtype="step",
            linewidth=2,
            linestyle="--",
            color=DATA_COLOR,
            label="Target",
            **hist_kw,
        )

        ax.set_xlabel("Classifier output")
        ax.set_ylabel("Density")
        ax.set_title(f"{method} score")
        ax.legend()
        _add_labels(ax)

        per_method_file = f"{stem}_{method}{ext}"
        plt.savefig(per_method_file, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Classifier output plot saved to: {per_method_file}")


def plot_weight_distributions(weights, output_file, bins=50, xlim=(0, 10)):
    """
    Plot histograms of weight distributions.

    Args:
        weights (dict): Dictionary where keys are labels and values are arrays of weights.
        output_file (str): Output file path for the plot.
        bins (int): Number of histogram bins.
        xlim (tuple or None): Limit for the x-axis, e.g., (0, 5). Default: (0, 10).
    """
    _apply_style()
    plt.figure(figsize=(10, 7))
    for label, w in weights.items():
        color = DATA_COLOR if label == "Data" else METHOD_COLORS.get(label, MC_COLOR)
        plt.hist(
            w,
            bins=bins,
            density=True,
            alpha=0.6,
            label=label,
            range=xlim,
            histtype="stepfilled",
            color=color,
        )

    plt.xlabel("weights")
    plt.ylabel("Density")
    plt.legend()
    if xlim:
        plt.xlim(xlim)
    plt.yscale("log")  # Helps visualize long tails
    _add_labels(plt.gca())
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"[INFO] Weight distributions plot saved to: {output_file}")
    plt.close()


def plot_2d_score_maps(
    sample, weights, classifier_scores, method, vars, output_file, x_labels, n_bins=40
):
    """
    Plot 2D heatmaps of mean classifier score vs all possible pairs of variables.

    Args:
        sample (dict): Dictionary containing MC and Data samples.
        weights (dict): Dictionary of weights for each sample.
        classifier_scores (dict): Dictionary of classifier scores for each sample.
        method (str): Reweighter method name.
        vars (list): List of variables to consider for 2D plots.
        output_file (str): Path to save the figure.
        x_labels (dict): Dictionary mapping column names to x-axis labels.
        n_bins (int): Number of bins for the 2D histogram.
    """
    _apply_style()

    var_pairs = list(itertools.combinations(vars, 2))
    n_plots = len(var_pairs)

    n_cols = 2 if len(vars) <= 4 and len(vars) != 3 else 3

    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.array(axes).reshape(-1)

    # Select samples
    if method in ("Folding", "XGBFolding", "NNFolding", "ONNXFolding"):
        mc = sample["mc"]
    else:
        mc = sample["mc_test"]

    scores = classifier_scores["MC"]

    for idx, (var_x, var_y) in enumerate(var_pairs):

        ax = axes[idx]

        x = mc[var_x].to_numpy()
        y = mc[var_y].to_numpy()

        # ---- weighted mean = sum(w*s) / sum(w)
        if weights is not None:

            sum_ws, x_edges, y_edges, _ = binned_statistic_2d(
                x, y, scores * weights, statistic="sum", bins=n_bins
            )

            sum_w, _, _, _ = binned_statistic_2d(
                x, y, weights, statistic="sum", bins=n_bins
            )

            score_map = np.divide(
                sum_ws, sum_w, out=np.zeros_like(sum_ws), where=sum_w > 0
            )

        # ---- unweighted mean
        else:

            score_map, x_edges, y_edges, _ = binned_statistic_2d(
                x, y, scores, statistic="mean", bins=n_bins
            )

            score_map = np.nan_to_num(score_map, nan=0.0)

        # ---- plot
        im = ax.imshow(
            score_map.T,  # transpose for correct orientation
            origin="lower",
            aspect="auto",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            cmap="viridis",
            vmin=0,
        )

        ax.set_xlabel(_label_for(x_labels, var_x))
        ax.set_ylabel(_label_for(x_labels, var_y))
        fig.colorbar(im, ax=ax)
        _add_labels(ax, x_min_lhcb=0.02, y_min_lhcb=0.98)

    # Hide unused axes
    for i in range(n_plots, len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"Mean classifier score for {method}", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"[INFO] 2D score maps saved to: {output_file}")


def plot_feature_importance(
    shap_values, feature_names, mc, x_labels, method, output_file, max_display=None
):
    """
    Plot SHAP beeswarm (summary) plot for a reweighter.

    Args:
        shap_values (dict): Dictionary of SHAP values for each method. Keys should match method names.
        feature_names (list): Feature names
        mc (pd.DataFrame): MC sample used for SHAP value computation (for feature values)
        x_labels (dict): Dictionary mapping column names to x-axis labels.
        method (str): Reweighter method name.
        output_file (str): Path to save figure
        max_display (int): Max number of features to show
    """
    _apply_style()

    X = mc[feature_names]

    shap_values = np.column_stack([shap_values[f] for f in feature_names])

    plt.figure(figsize=(9, 6))

    shap.summary_plot(
        shap_values,
        X,
        feature_names=[_label_for(x_labels, f) for f in feature_names],
        max_display=max_display,
        show=False,
    )

    plt.xlabel("SHAP value (impact on log weight)")
    plt.title(f"Feature importance for {method}")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_2d_pull_maps(
    mc,
    data,
    mc_weights,
    data_weights,
    columns,
    x_labels,
    method,
    output_file,
    n_bins=40,
    pull_clip=5,
):
    """
    Plot 2D pull maps for all variable pairs.

    The pull in each bin is computed as
    ``(data_density - mc_density) / sqrt(var_data + var_mc)``.

    Both MC and Data are normalized to densities so that
    different statistics are handled correctly.

    Args:
        mc, data (pd.DataFrame): MC and Data samples.
        mc_weights, data_weights (np.ndarray): Weights for MC and Data.
        columns (list): List of column names to consider for the pull maps.
        x_labels (dict): Dictionary mapping column names to x-axis labels.
        method (str): Reweighting method name (for plot title).
        output_file (str): Path to save figure
        n_bins (int): Number of bins for the 2D histograms
        pull_clip (float): Maximum absolute value for pull map clipping
    """

    _apply_style()

    var_pairs = list(itertools.combinations(columns, 2))
    n_plots = len(var_pairs)

    n_cols = 2 if len(columns) <= 4 and len(columns) != 3 else 3

    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

    axes = np.array(axes).reshape(-1)

    for idx, (var_x, var_y) in enumerate(var_pairs):

        ax = axes[idx]

        x_mc = mc[var_x].to_numpy()
        y_mc = mc[var_y].to_numpy()
        x_data = data[var_x].to_numpy()
        y_data = data[var_y].to_numpy()

        # --------------------------------------------------
        # MC weighted counts and variances
        mc_sumw, x_edges, y_edges, _ = binned_statistic_2d(
            x_mc, y_mc, mc_weights, statistic="sum", bins=n_bins
        )

        mc_sumw2, _, _, _ = binned_statistic_2d(
            x_mc, y_mc, mc_weights**2, statistic="sum", bins=[x_edges, y_edges]
        )

        # --------------------------------------------------
        # Data weighted counts and variances
        data_sumw, _, _, _ = binned_statistic_2d(
            x_data, y_data, data_weights, statistic="sum", bins=[x_edges, y_edges]
        )

        data_sumw2, _, _, _ = binned_statistic_2d(
            x_data, y_data, data_weights**2, statistic="sum", bins=[x_edges, y_edges]
        )

        # --------------------------------------------------
        # Convert to densities
        dx = np.diff(x_edges)
        dy = np.diff(y_edges)
        area = dx[:, None] * dy[None, :]

        mc_norm = mc_weights.sum()
        data_norm = data_weights.sum()

        mc_density = mc_sumw / (mc_norm * area)
        data_density = data_sumw / (data_norm * area)

        mc_var = mc_sumw2 / (mc_norm**2 * area**2)
        data_var = data_sumw2 / (data_norm**2 * area**2)

        # --------------------------------------------------
        # Pull map
        total_unc = np.sqrt(mc_var + data_var)

        pull_map = np.divide(
            data_density - mc_density,
            total_unc,
            out=np.zeros_like(data_density),
            where=total_unc > 0,
        )

        pull_map = np.clip(pull_map, -pull_clip, pull_clip)

        # --------------------------------------------------
        # Plot
        norm = TwoSlopeNorm(vmin=-pull_clip, vcenter=0.0, vmax=pull_clip)

        im = ax.imshow(
            pull_map.T,
            origin="lower",
            aspect="auto",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            cmap="coolwarm",
            norm=norm,
        )

        ax.set_xlabel(_label_for(x_labels, var_x))
        ax.set_ylabel(_label_for(x_labels, var_y))

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Pull")
        _add_labels(ax, x_min_lhcb=0.02, y_min_lhcb=0.98)

    # --------------------------------------------------
    # Hide unused pads
    for i in range(n_plots, len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"2D Pull Maps for {method}", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"[INFO] 2D pull maps saved to: {output_file}")
