import matplotlib
import numpy as np
import pandas as pd

from mcreweight.utils.plotting_utils import plot_mc_distributions


matplotlib.use("Agg")


def test_plot_mc_distributions_supports_single_column(tmp_path):
    mc = pd.DataFrame({"pt": [1.0, 2.0, 3.0, 4.0]})
    original_weights = np.ones(len(mc))
    reweighted = np.array([1.0, 1.2, 0.8, 1.1])
    output_file = tmp_path / "single_column_plot.png"

    plot_mc_distributions(
        mc=mc,
        original_mc_weights=original_weights,
        new_mc_weights=reweighted,
        columns=["pt"],
        x_labels={"pt": "pT"},
        output_file=output_file,
    )

    assert output_file.exists()
    assert output_file.stat().st_size > 0
