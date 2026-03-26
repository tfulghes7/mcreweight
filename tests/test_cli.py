from pathlib import Path
import os
import subprocess

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]

CI_WEIGHTS_DIR = os.environ.get("MCREWEIGHT_CI_WEIGHTS_DIR")
CI_PLOTS_DIR = os.environ.get("MCREWEIGHT_CI_PLOTS_DIR")
CI_OUTPUT_FILE = os.environ.get("MCREWEIGHT_CI_OUTPUT_FILE")


def _write_run_config(path: Path, weights_dir: Path, plot_dir: Path) -> Path:
    cfg = {
        "input": {
            "mc": {"path": ["tests_run/test_mc.root"], "tree": "DecayTree"},
            "data": {
                "path": ["tests_run/test_data.root"],
                "tree": "DecayTree",
                "sweights_name": "sweight_sig",
            },
            "path_xlabels": "src/mcreweight/utils/default_xlabels.yaml",
        },
        "variables": {
            "training_vars": ["B_DTF_Jpsi_P", "B_DTF_Jpsi_PT", "nPVs", "nLongTracks"],
            "monitoring_vars": ["B_PHI", "B_ETA"],
        },
        "reweighting": {
            "methods": ["Bins"],
            "sample": "ci_test_sample",
            "transform": None,
            "n_trials": 0,
            "n_folds": 2,
            "test_size": 0.3,
            "n_bins": 10,
            "n_neighs": 3,
            "shap": False,
        },
        "output": {
            "weightsdir": str(weights_dir),
            "plotdir": str(plot_dir),
        },
    }

    cfg_path = path / "run_reweighting_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def _write_apply_config(
    path: Path, weights_dir: Path, plot_dir: Path, output_path: Path
) -> Path:
    cfg = {
        "input": {
            "mc": {"path": ["tests_run/test_mc.root"], "tree": "DecayTree"},
            "data": {
                "path": ["tests_run/test_data.root"],
                "tree": "DecayTree",
                "sweights_name": "sweight_sig",
            },
            "path_xlabels": "src/mcreweight/utils/default_xlabels.yaml",
        },
        "variables": {
            "application_vars": [
                "B_DTF_Jpsi_P",
                "B_DTF_Jpsi_PT",
                "nPVs",
                "nLongTracks",
            ],
            "training_vars": ["B_DTF_Jpsi_P", "B_DTF_Jpsi_PT", "nPVs", "nLongTracks"],
            "monitoring_vars": ["B_PHI", "B_ETA"],
        },
        "reweighting": {
            "method": "Bins",
            "application_sample": "ci_test_sample",
            "training_sample": "ci_test_sample",
            "weightsdir": str(weights_dir),
            "plotdir": str(plot_dir),
        },
        "output": {
            "output_path": str(output_path),
            "output_ntuple": "TTree",
            "output_tree": "DecayTree",
            "weights_name": "test_weights",
        },
    }

    cfg_path = path / "apply_weights_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def _run_command(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.cli
def test_run_reweight(tmp_path):
    weights_dir = Path(CI_WEIGHTS_DIR) if CI_WEIGHTS_DIR else tmp_path / "weights"
    plot_dir = Path(CI_PLOTS_DIR) if CI_PLOTS_DIR else tmp_path / "plots"

    weights_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = _write_run_config(tmp_path, weights_dir, plot_dir)

    run_proc = _run_command(["run-reweight", "--config", str(run_cfg)])

    assert run_proc.returncode == 0, (
        "run-reweight failed\n"
        f"stdout:\n{run_proc.stdout}\n"
        f"stderr:\n{run_proc.stderr}"
    )

    sample_dir = weights_dir / "ci_test_sample"
    assert (
        sample_dir.exists()
    ), f"Expected weights sample directory to exist: {sample_dir}"


@pytest.mark.cli
def test_apply_weights(tmp_path):
    weights_dir = Path(CI_WEIGHTS_DIR) if CI_WEIGHTS_DIR else tmp_path / "weights"
    plot_dir = Path(CI_PLOTS_DIR) if CI_PLOTS_DIR else tmp_path / "plots"
    output_file = (
        Path(CI_OUTPUT_FILE) if CI_OUTPUT_FILE else tmp_path / "applied_weights.root"
    )

    if not (weights_dir / "ci_test_sample").exists():
        weights_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(parents=True, exist_ok=True)
        run_cfg = _write_run_config(tmp_path, weights_dir, plot_dir)
        run_proc = _run_command(["run-reweight", "--config", str(run_cfg)])
        assert run_proc.returncode == 0, (
            "precondition run-reweight failed\n"
            f"stdout:\n{run_proc.stdout}\n"
            f"stderr:\n{run_proc.stderr}"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    apply_cfg = _write_apply_config(tmp_path, weights_dir, plot_dir, output_file)

    apply_proc = _run_command(["apply-weights", "--config", str(apply_cfg)])

    assert apply_proc.returncode == 0, (
        "apply-weights failed\n"
        f"stdout:\n{apply_proc.stdout}\n"
        f"stderr:\n{apply_proc.stderr}"
    )
    assert output_file.exists(), f"Expected output file to exist: {output_file}"
