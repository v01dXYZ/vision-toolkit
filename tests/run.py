#!/bin/env python
import argparse
import importlib
import pathlib

def get_cli_args():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("version", type=int, choices=[1, 2])
    arg_parser.add_argument("test_name", choices=["hollywood2", "zemblys"])
    arg_parser.add_argument("cutoff", type=int)
    arg_parser.add_argument("report_name", nargs="?", default="report")

    default_directory_root = "results"
    arg_parser.add_argument("-d", "--directory", help=f"{default_directory_root}/<test_name>/<version>", type=pathlib.Path)

    args = arg_parser.parse_args()

    if args.directory is None:
        args.directory = pathlib.Path(default_directory_root) / args.test_name / f"v{args.version}"

    return args


if __name__ == "__main__":
    args = get_cli_args()

    test_mod = importlib.import_module(f"{args.test_name}")

    test_mod.EntryPoint.main(
        cutoff=args.cutoff,
        report_name=args.report_name,
        directory=args.directory,
        version=args.version,
    )
