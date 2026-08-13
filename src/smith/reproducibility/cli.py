from __future__ import annotations

import argparse
import json
from pathlib import Path

from .registry import load_cases
from .runner import check_case, run_case


def _print(payload) -> None:
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="smith-repro", description="Run representative SMITH paper analyses.")
    parser.add_argument("--root", default=None, help="Override the reproducibility directory.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list")
    check = subparsers.add_parser("check")
    check.add_argument("case", nargs="?", default="all")
    check.add_argument("--data-root", default=None)
    run = subparsers.add_parser("run")
    run.add_argument("case")
    run.add_argument("--output-dir", default=None)
    run.add_argument("--data-root", required=True)
    run.add_argument("workflow_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cases = load_cases(args.root)
    if args.command == "list":
        _print([case.to_dict() for case in cases.values()])
        return
    selected = cases.values() if args.case == "all" else [cases[args.case]]
    if args.command == "check":
        _print([check_case(case, args.root, args.data_root) for case in selected])
        return
    case = cases[args.case]
    output_dir = Path(args.output_dir or Path("outputs") / "reproducibility" / case.id)
    _print(run_case(case, output_dir, args.root, args.data_root, args.workflow_args))


if __name__ == "__main__":
    main()
