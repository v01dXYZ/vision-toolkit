#!/bin/env python
import sys
import argparse
import importlib
import pathlib
from vision_toolkit2.config import Config


def get_cli_args():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("version", type=int, choices=[1, 2])
    arg_parser.add_argument("test_name", choices=["hollywood2", "zemblys"])
    arg_parser.add_argument("cutoff", type=int)
    arg_parser.add_argument("report_name", nargs="?", default="report")
    arg_parser.add_argument("-c", "--config", help="config to add. example: distance_type=euclidean sample_frequency=1000", nargs="*", default=[])

    default_directory_root = "results"
    arg_parser.add_argument("-d", "--directory", help=f"{default_directory_root}/<test_name>/<version>", type=pathlib.Path)

    args = arg_parser.parse_args()

    if args.directory is None:
        args.directory = pathlib.Path(default_directory_root) / args.test_name / f"v{args.version}"

    return args

def config_constructor(c):
    r = c.split("=")

    if len(r) != 2:
        raise ValueError(f"{c!r} contains more than one '='")

    config_key, config_value_str = r

    dataclass_field = Config.__dataclass_fields__.get(config_key)

    if dataclass_field is None:
        raise ValueError(f"{c!r} referencing a non-existing config key")

    dataclass_type = dataclass_field.type

    if dataclass_type is None:
        raise ValueError(f"{c!r} referencing a not-typed yet config key")

    try:
        config_value = dataclass_type(config_value_str)
    except Exception:
        raise ValueError(f"{c!r} caused an error when parsing the string of the value")

    return (config_key, config_value)

if __name__ == "__main__":
    args = get_cli_args()

    errors = []
    config = {}
    for c in args.config:
        try:
            k, v = config_constructor(c)
        except ValueError as ve:
            (msg,) = ve.args
            errors.append(msg)
        else:
            config[k] = v

    if errors:
        print("Error with provided config arguments", file=sys.stderr)
        for err in errors:
            print(" "*5, "-", err, file=sys.stderr)

        exit(-1)

    test_mod = importlib.import_module(f"{args.test_name}")

    test_mod.EntryPoint.main(
        cutoff=args.cutoff,
        report_name=args.report_name,
        directory=args.directory,
        version=args.version,
        config=config,
    )
