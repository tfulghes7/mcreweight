#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import uproot


def _to_branch_dict(arrays):
    return {field: ak.to_numpy(arrays[field]) for field in ak.fields(arrays)}


def copy_first_entries(
    input_path: Path,
    output_path: Path,
    tree_name: str,
    n_events: int,
    output_ntuple: str,
) -> None:
    with uproot.open(input_path) as src:
        tree = src[tree_name]
        arrays = tree.arrays(entry_stop=n_events, library="ak")

    branch_dict = _to_branch_dict(arrays)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(output_path) as dst:
        if output_ntuple == "TTree":
            dst.mktree(tree_name, branch_dict)
            dst[tree_name].extend(branch_dict)
        elif output_ntuple == "RNTuple":
            writer = dst.mkrntuple(tree_name, branch_dict)
            writer.extend(branch_dict)
        else:
            raise ValueError(f"Unsupported output ntuple type: {output_ntuple}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy the first N events from source ROOT files into tests_run fixtures."
    )
    parser.add_argument(
        "--input-data",
        default="test_data.root",
        help="Path to the source data ROOT file.",
    )
    parser.add_argument(
        "--input-mc",
        default="test_mc.root",
        help="Path to the source MC ROOT file.",
    )
    parser.add_argument(
        "--output-data",
        default="tests_run/test_data.root",
        help="Path to the output data ROOT file.",
    )
    parser.add_argument(
        "--output-mc",
        default="tests_run/test_mc.root",
        help="Path to the output MC ROOT file.",
    )
    parser.add_argument(
        "--tree",
        default="DecayTree",
        help="Name of the TTree/RNTuple to copy from both files.",
    )
    parser.add_argument(
        "--n-events",
        type=int,
        default=5000,
        help="Number of events to copy from each input file.",
    )
    parser.add_argument(
        "--output-ntuple",
        choices=["TTree", "RNTuple"],
        default="TTree",
        help="Output object type to create in the destination ROOT files.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_data = Path(args.input_data)
    input_mc = Path(args.input_mc)
    output_data = Path(args.output_data)
    output_mc = Path(args.output_mc)

    copy_first_entries(
        input_data, output_data, args.tree, args.n_events, args.output_ntuple
    )
    print(
        f"Wrote {args.n_events} events from {input_data} "
        f"to {output_data} ({args.tree}, {args.output_ntuple})."
    )

    copy_first_entries(
        input_mc, output_mc, args.tree, args.n_events, args.output_ntuple
    )
    print(
        f"Wrote {args.n_events} events from {input_mc} "
        f"to {output_mc} ({args.tree}, {args.output_ntuple})."
    )


if __name__ == "__main__":
    main()
