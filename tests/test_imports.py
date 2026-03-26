import mcreweight
from importlib.metadata import version
import subprocess


def test_version():
    assert isinstance(mcreweight.__version__, str)
    parts = mcreweight.__version__.split(".")
    assert len(parts) >= 2


def test_public_api():
    assert callable(mcreweight.run_reweighting_pipeline)
    assert callable(mcreweight.apply_weights_pipeline)
    assert callable(mcreweight.load_data)
    assert callable(mcreweight.save_data)
    assert callable(mcreweight.def_aliases)
    assert callable(mcreweight.train_and_test)
    assert callable(mcreweight.gbreweight)
    assert callable(mcreweight.kfolding)
    assert callable(mcreweight.binning_reweight)
    assert callable(mcreweight.optuna_tuning)
    assert isinstance(mcreweight.X_LABELS, dict)


def test_version_matches_installed_metadata():
    assert mcreweight.__version__ == version("mcreweight")


def test_module_docstring_present():
    assert isinstance(mcreweight.__doc__, str)
    assert "reweight" in mcreweight.__doc__.lower()


def test_extended_training_api_exports():
    assert callable(mcreweight.xgbreweight)
    assert callable(mcreweight.nnreweight)
    assert callable(mcreweight.gbfolding)
    assert callable(mcreweight.xgbfolding)
    assert callable(mcreweight.nnfolding)


def test_cli_commands_show_help():
    for command in ["run-reweight", "apply-weights"]:
        proc = subprocess.run(
            [command, "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0
        assert "usage:" in proc.stdout.lower()
