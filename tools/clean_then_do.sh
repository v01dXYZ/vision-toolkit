#!/bin/env bash

# Why this script? pip reuses some of its previous outputs if it finds them.
# - build/, previous files were packages although removed in source tree
# - egg-info, the list of the packages is stored and reused which is a problem
# when changing pyproject.toml config.
# I don't know all the ways pip works and cache, easiest way now is removing
# caches before building.
#
# uv pip create those directories as well (I don't know if it uses it as caches)

if [[ ! -d src ]]; then
    echo "ERROR: should be run in the source directory"
    exit -1
fi

rm -r build dist
find src -name "*.egg-info*" -exec rm -r {} \+

$@
