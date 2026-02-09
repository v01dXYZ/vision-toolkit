#!/bin/env python

import sys
import json
import argparse

COVERAGE_JSON = "coverage.json"
BASE_URL = "https://github.com"

def create_link(
        repo,
        commit_sha,
        path,
        interval,
):
    return f"[{interval[0]}-{interval[1]}]({BASE_URL}/{repo}/blob/{commit_sha}/src/vision_toolkit/{path}#L{interval[0]}-L{interval[1]})"

def get_intervals(lines):

    if not lines:
        return []

    iter_lines = iter(lines)

    start = next(iter_lines)
    prev = start

    for l in iter_lines:
        if l > prev + 1:
            yield (start, prev)
            start = l

        prev = l

    yield (start, prev)

if __name__ == "__main__":
    arg_parse = argparse.ArgumentParser()

    arg_parse.add_argument("--coverage-filepath", default=COVERAGE_JSON)
    arg_parse.add_argument("-c", "--commit-sha", required=True)
    arg_parse.add_argument("-r", "--repo", required=True)

    args = arg_parse.parse_args()

    with open(args.coverage_filepath) as coverage_file:
        coverage = json.load(coverage_file)

    tbl_data = []
    for f, f_report in coverage["files"].items():
        missing_lines = f_report["missing_lines"]
        if not missing_lines:
            continue

        missing_intervals = list(get_intervals(missing_lines))

        tbl_data.append((f, " ".join(
            [
                create_link(
                    args.repo,
                    args.commit_sha,
                    f,
                    interval,
                )
                for interval in missing_intervals
            ])))

    headers = ["File", "Missing Intervals"]

    sys.stdout.write("|")
    sys.stdout.write("|".join(headers))
    sys.stdout.write("|\n")


    for _ in headers:
        sys.stdout.write("|---")
    sys.stdout.write("|\n")

    for row in tbl_data:
        sys.stdout.write("| ")
        sys.stdout.write(" | ".join(row))
        sys.stdout.write(" |\n")
