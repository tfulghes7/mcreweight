import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import math
from mcreweight.utils.utils import (
    evaluate_reweighting,
    weighted_corr_matrix,
    weighted_ks_statistic,
    get_scores,
)
import seaborn as sns
import itertools
import shap
from scipy.stats import binned_statistic_2d
from .utils import fit_transform, apply_transform


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


def set_lhcb_style(grid=True, size=10, usetex=False):
    """
    Set matplotlib plotting style close to "official" LHCb style
    (serif fonts, tick sizes and location, etc.)
    """
    plt.rc("font", family="serif", size=size)
    plt.rc("text", usetex=usetex)
    plt.rcParams["figure.max_open_warning"] = 40
    plt.rcParams["axes.linewidth"] = 1.3
    plt.rcParams["axes.grid"] = grid
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["axes.axisbelow"] = False
    plt.rcParams["xtick.major.width"] = 1
    plt.rcParams["ytick.major.width"] = 1
    plt.rcParams["xtick.minor.width"] = 1
    plt.rcParams["ytick.minor.width"] = 1
    plt.rcParams["xtick.major.size"] = 6
    plt.rcParams["ytick.major.size"] = 6
    plt.rcParams["xtick.minor.size"] = 3
    plt.rcParams["ytick.minor.size"] = 3
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["ytick.minor.visible"] = True
    plt.rcParams["xtick.bottom"] = True
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.left"] = True
    plt.rcParams["ytick.right"] = True


hist_settings = {"bins": 50, "density": True, "alpha": 0.7}
plt.rcParams.update(
    {
        "axes.titlesize": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
    }
)


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
    set_lhcb_style()
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
    ax = plt.gca()
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(
        [_label_for(x_labels, col) for col in columns], fontsize=22, rotation=45
    )
    plt.yticks(fontsize=22)
    if corr_mode == "absolute":
        plt.title(f"{title} (|weights| correlation)")
    else:
        plt.title(title)
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
    set_lhcb_style()
    if args.verbosity >= 3:
        print(f"[INFO] Plotting columns: {columns}")
        print(f"[INFO] MC size: {len(mc)}, Data size: {len(data)}")

    n_plots = len(columns)
    n_cols = min(3, n_plots) if n_plots != 4 else 2
    if n_plots >= 10:
        n_cols = 5
    n_rows = math.ceil(n_plots / n_cols)
    grid_rows = n_rows * 3

    fig, axes = plt.subplots(
        grid_rows,
        n_cols,
        figsize=(8.8 * n_cols, 6.6 * n_rows),
        gridspec_kw={"height_ratios": [3.0, 1.0, 0.55] * n_rows},
        constrained_layout=False,
    )
    axes = (
        np.array(axes).reshape(grid_rows, n_cols)
        if n_rows * n_cols > 1
        else np.array([[axes[0]], [axes[1]], [axes[2]]])
    )

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
            label="MC",
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
            label="Data",
            capsize=3,
        )
        ax_main.set_ylabel("A.U.")
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)
        ax_main.set_xticklabels([])

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

    # Hide unused axes
    used_axes = len(columns) * 3
    for i in range(used_axes, axes.size):
        axes.flat[i].axis("off")

    fig.subplots_adjust(
        left=0.10,
        right=0.96,
        top=0.93,
        bottom=0.14,
        wspace=0.34,
        hspace=0.08,
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
    set_lhcb_style()
    hist_settings = dict(bins=50, histtype="step", linewidth=1.5)
    n_cols = 3 if len(columns) != 4 else 2
    if len(columns) >= 10:
        n_cols = 5
    n_rows = math.ceil(len(columns) / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows), constrained_layout=True
    )
    if isinstance(axes, np.ndarray):
        axes = axes.reshape(n_rows, n_cols)

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

    set_lhcb_style()

    items = sorted(
        throughput.items(),
        key=lambda item: item[1].get("effective_events_per_second", 0.0),
        reverse=True,
    )
    methods = [method for method, _ in items]
    dataset_rates = [
        metrics.get("dataset_events_per_second", 0.0) for _, metrics in items
    ]
    effective_rates = [
        metrics.get("effective_events_per_second", 0.0) for _, metrics in items
    ]

    y = np.arange(len(methods))
    height = 0.36
    fig_height = max(5, 0.9 * len(methods) + 2)

    fig, ax = plt.subplots(figsize=(12, fig_height), constrained_layout=True)
    ax.barh(
        y - height / 2,
        dataset_rates,
        height=height,
        label="Dataset events/s",
        alpha=0.85,
    )
    ax.barh(
        y + height / 2,
        effective_rates,
        height=height,
        label="Effective events/s",
        alpha=0.85,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.invert_yaxis()
    ax.set_xlabel("Training Throughput [events/s]")
    ax.set_title("Training Throughput by Method")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)

    xmax = max(max(dataset_rates, default=0.0), max(effective_rates, default=0.0))
    if xmax > 0:
        ax.set_xlim(0, xmax * 1.15)

    for ypos, rate in zip(y - height / 2, dataset_rates):
        ax.text(rate, ypos, f" {rate:.1f}", va="center", ha="left", fontsize=14)
    for ypos, rate in zip(y + height / 2, effective_rates):
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

    set_lhcb_style()

    items = sorted(
        memory_profile.items(),
        key=lambda item: item[1].get("rss_peak_bytes") or 0,
        reverse=True,
    )
    methods = [method for method, _ in items]
    peak_mb = [(metrics.get("rss_peak_bytes") or 0) / (1024**2) for _, metrics in items]
    delta_mb = [
        (metrics.get("rss_delta_bytes") or 0) / (1024**2) for _, metrics in items
    ]

    y = np.arange(len(methods))
    height = 0.36
    fig_height = max(5, 0.9 * len(methods) + 2)

    fig, ax = plt.subplots(figsize=(12, fig_height), constrained_layout=True)
    ax.barh(
        y - height / 2,
        peak_mb,
        height=height,
        label="Peak RSS [MB]",
        alpha=0.85,
    )
    ax.barh(
        y + height / 2,
        delta_mb,
        height=height,
        label="Peak RSS increase [MB]",
        alpha=0.85,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.invert_yaxis()
    ax.set_xlabel("Memory [MB]")
    ax.set_title("Training Memory by Method")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)

    xmax = max(max(peak_mb, default=0.0), max(delta_mb, default=0.0))
    if xmax > 0:
        ax.set_xlim(0, xmax * 1.15)

    for ypos, value in zip(y - height / 2, peak_mb):
        ax.text(value, ypos, f" {value:.1f}", va="center", ha="left", fontsize=14)
    for ypos, value in zip(y + height / 2, delta_mb):
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
    set_lhcb_style()
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
    ax.set_title("ROC Curve: Classifier distinguishing reweighted MC from Data")
    ax.legend(loc="lower right")
    plt.savefig(output_file)
    print(f"[INFO] ROC curve saved to: {output_file}")
    plt.close()

    return scores


def plot_classifier_output(
    scores, weights, methods, output_file, min_score=0.0, max_score=1.0
):
    """
    Plot classifier output distributions and show weighted KS vs Data.

    Parameters are the classifier ``scores`` returned by ``plot_roc_curve``,
    the per-method ``weights`` mapping, the list of ``methods`` to display, the
    destination ``output_file``, and the histogram range defined by
    ``min_score`` and ``max_score``.
    """

    set_lhcb_style()

    plt.figure(figsize=(16, 12))

    example_method = next(iter(scores))
    score_data = scores[example_method]["Data"]
    weights_data = weights.get("Data", np.ones_like(score_data))

    for method in methods:
        if "MC" not in scores[method] or scores[method]["MC"] is None:
            continue

        score_mc = scores[method]["MC"]
        w_mc = weights.get(method, np.ones_like(score_mc))
        method_color = METHOD_COLORS.get(method, MC_COLOR)

        # KS statistic vs Data
        ks_val = weighted_ks_statistic(score_mc, score_data, w1=w_mc, w2=weights_data)
        legend_label = f"{method} (KS = {ks_val:.3f})"

        plt.hist(
            score_mc,
            bins=50,
            density=True,
            weights=w_mc,
            alpha=0.6,
            range=(min_score, max_score),
            label=legend_label,
            color=method_color,
        )

    # Also show Data distribution
    plt.hist(
        score_data,
        bins=50,
        density=True,
        weights=weights_data,
        alpha=0.6,
        range=(min_score, max_score),
        label="Data",
        color=DATA_COLOR,
    )

    plt.xlabel("Classifier output")
    plt.ylabel("Density")
    plt.title("Classifier score distributions")
    plt.legend()
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Classifier output plot saved to: {output_file}")


def plot_weight_distributions(weights, output_file, bins=50, xlim=(0, 10)):
    """
    Plot histograms of weight distributions.

    Args:
        weights (dict): Dictionary where keys are labels and values are arrays of weights.
        output_file (str): Output file path for the plot.
        bins (int): Number of histogram bins.
        xlim (tuple or None): Limit for the x-axis, e.g., (0, 5). Default: (0, 10).
    """
    set_lhcb_style()
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
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
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
    set_lhcb_style()

    var_pairs = list(itertools.combinations(vars, 2))
    n_plots = len(var_pairs)

    if len(vars) <= 4 and len(vars) != 3:
        n_cols = 2
    else:
        n_cols = 3

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
    set_lhcb_style()

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

    set_lhcb_style()

    var_pairs = list(itertools.combinations(columns, 2))
    n_plots = len(var_pairs)

    if len(columns) <= 4 and len(columns) != 3:
        n_cols = 2
    else:
        n_cols = 3

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

    # --------------------------------------------------
    # Hide unused pads
    for i in range(n_plots, len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"2D Pull Maps for {method}", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"[INFO] 2D pull maps saved to: {output_file}")
