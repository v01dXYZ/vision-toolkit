#!/bin/env bash
# /pkg contains the container package
vision_toolkit_srcdir="$(realpath $(dirname $0)/..)"

pip install "$vision_toolkit_srcdir/[test]" --no-deps

$@
