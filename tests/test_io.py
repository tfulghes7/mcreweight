import pandas as pd

from mcreweight.io import (
    def_aliases,
    extract_variables_from_expression,
    flatten_vars,
    get_from_cfg,
    load_config,
    read_x_labels,
)


def test_extract_simple_variable():
    assert extract_variables_from_expression("pt") == {"pt"}


def test_extract_expression():
    result = extract_variables_from_expression("log(pt)")
    assert "pt" in result
    assert "log" not in result


def test_extract_division():
    result = extract_variables_from_expression("eCalTot/nEcalClusters")
    assert result == {"eCalTot", "nEcalClusters"}


def test_flatten_vars_simple():
    assert flatten_vars(["pt", "eta"]) == ["pt", "eta"]


def test_flatten_vars_operators():
    result = flatten_vars(["a/b", "a+b", "a*b", "a-b"])
    assert result == ["aoverb", "aplusb", "atimesb", "aminusb"]


def test_flatten_vars_functions():
    result = flatten_vars(["log(pt)"])
    assert result == ["logpt"]


def test_extract_nested_functions_and_underscores():
    result = extract_variables_from_expression(
        "sqrt(pt_1*pt2) + abs(eta) - exp(alpha_2)"
    )
    assert result == {"pt_1", "pt2", "eta", "alpha_2"}


def test_extract_repeated_variables_are_unique():
    result = extract_variables_from_expression("pt + pt/eta")
    assert result == {"pt", "eta"}


def test_extract_function_like_variable_name_is_kept():
    result = extract_variables_from_expression("log(pt) + log_var")
    assert result == {"pt", "log_var"}


def test_flatten_vars_mixed_types_and_empty_strings():
    result = flatten_vars(["a/b", "", "   ", None, 3])
    assert result == ["aoverb", "", "   ", None, 3]


def test_flatten_vars_function_with_expression_argument():
    result = flatten_vars(["log(a/b)"])
    assert result == ["logaoverb"]


def test_extract_with_numbers_and_parentheses():
    result = extract_variables_from_expression("(pt1 + 2*pt2)/(eta_1 - 3)")
    assert result == {"pt1", "pt2", "eta_1"}


def test_extract_empty_expression_returns_empty_set():
    assert extract_variables_from_expression("") == set()


def test_flatten_vars_preserves_order_after_processing():
    result = flatten_vars(["a/b", "x", "y-z", "log(pt)"])
    assert result == ["aoverb", "x", "yminusz", "logpt"]


def test_flatten_vars_nested_function_name_pattern_not_removed():
    # Current regex only flattens function calls with bare word args.
    result = flatten_vars(["sqrt(log(pt))"])
    assert result == ["sqrtlogpt"]


def test_get_from_cfg_returns_nested_value():
    cfg = {"a": {"b": {"c": 42}}}
    assert get_from_cfg(cfg, ["a", "b", "c"]) == 42


def test_get_from_cfg_returns_default_for_missing_path():
    cfg = {"a": {"b": 1}}
    assert get_from_cfg(cfg, ["a", "x"], default="missing") == "missing"


def test_get_from_cfg_returns_default_if_intermediate_is_none():
    cfg = {"a": None}
    assert get_from_cfg(cfg, ["a", "b"], default=0) == 0


def test_def_aliases_copies_column_and_evaluates_expression():
    df = pd.DataFrame({"pt": [1.0, 2.0], "eta": [0.5, 1.5]})
    aliases = {"pt_copy": "pt", "sum_pt_eta": "pt + eta"}

    out = def_aliases(df.copy(), aliases)

    assert list(out["pt_copy"]) == [1.0, 2.0]
    assert list(out["sum_pt_eta"]) == [1.5, 3.5]


def test_def_aliases_invalid_expression_does_not_add_column():
    df = pd.DataFrame({"pt": [1.0, 2.0]})

    out = def_aliases(df.copy(), {"bad": "missing_col + 1"})

    assert "bad" not in out.columns


def test_load_config_reads_yaml(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("a:\n  b: 3\n", encoding="utf-8")

    out = load_config(str(cfg_path))

    assert out == {"a": {"b": 3}}


def test_read_x_labels_reads_yaml(tmp_path):
    labels_path = tmp_path / "labels.yaml"
    labels_path.write_text("pt: pT\neta: eta\n", encoding="utf-8")

    out = read_x_labels(str(labels_path))

    assert out == {"pt": "pT", "eta": "eta"}
