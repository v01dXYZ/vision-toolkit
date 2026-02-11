#!/bin/env bash
set -e

if [[ -z $VISION_TOOLKIT_VERSION && -z $1 ]]; then
    echo "ERROR: env variable VISION_TOOLKIT_VERSION or first arg should be defined."
    exit -1
fi

if [[ -n $VISION_TOOLKIT_VERSION && -n $1 ]]; then
    echo "ERROR: both env variable VISION_TOOLKIT_VERSION and first arg provided, define only one"
    exit -1
fi

if [[ -z $VISION_TOOLKIT_VERSION ]]; then
    VISION_TOOLKIT_VERSION=$1
fi

if [[ $VISION_TOOLKIT_VERSION == "1" ]]; then
    vision_toolkit_version_suffix=""
elif [[ $VISION_TOOLKIT_VERSION == "2" ]]; then
    vision_toolkit_version_suffix=$VISION_TOOLKIT_VERSION
else
    echo "ERROR: there are only version 1 or 2"
    exit -1
fi


vision_toolkit_srcdir="$(realpath $(dirname $0)/..)"

vision_toolkit_modname=vision_toolkit$vision_toolkit_version_suffix
# do not forget trailing slash
vision_toolkit_moddir=$vision_toolkit_srcdir/src/$vision_toolkit_modname/
source_moddir=$(python -c "import importlib.util; print(importlib.util.find_spec('$vision_toolkit_modname').submodule_search_locations[0])")

# coverage_run="coverage run --source $source_moddir"
coverage_run="coverage run --source $vision_toolkit_modname --rcfile $vision_toolkit_srcdir/pyproject.toml"

datasets="hollywood2 zemblys"
distance_types="euclidean angular"

# cd $vision_toolkit_srcdir/tests

for dataset in $datasets; do
    for distance_type in $distance_types; do
	coverage_datafile=coverage_${dataset}_${distance_type}.sqlite2
	coverage_datafiles="$coverage_datafile $coverage_datafiles"
	$coverage_run --data-file=$coverage_datafile \
		      $vision_toolkit_srcdir/tests/run.py \
		      $VISION_TOOLKIT_VERSION \
		      $dataset \
		      2 \
		      -c distance_type=$distance_type
    done
done

for coverage_datafile in $coverage_datafiles; do
    sqlite3 $coverage_datafile \
        "UPDATE file SET path =  '$vision_toolkit_moddir' || SUBSTR(path, INSTR(path, '$vision_toolkit_modname') + LENGTH('$vision_toolkit_modname') + 1)"
done

# cd $vision_toolkit_moddir
#
# for coverage_datafile in $coverage_datafiles; do
#     ln -s $vision_toolkit_srcdir/tests/$coverage_datafile .
# done

coverage combine $coverage_datafiles
coverage report -m
coverage json
