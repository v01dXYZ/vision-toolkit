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

if [[ ! -d src ]]; then
    # Cython always put into the c/cpp file a relative path. So the plugin will use this path as well.
    # It means there is no other way to run coverage but in the source directory.
    echo "ERROR: This script should be ran at the source directory (Cython internally uses relative paths)"
    exit -1
fi


vision_toolkit_modname=vision_toolkit$vision_toolkit_version_suffix

coverage_run="coverage run --source $vision_toolkit_modname"

datasets="hollywood2 zemblys"
distance_types="euclidean angular"

for dataset in $datasets; do
    for distance_type in $distance_types; do
	coverage_datafile=coverage_${dataset}_${distance_type}.sqlite2
	coverage_datafiles="$coverage_datafile $coverage_datafiles"
	$coverage_run --data-file=$coverage_datafile \
		      ./tests/run.py \
		      $VISION_TOOLKIT_VERSION \
		      $dataset \
		      2 \
		      -c distance_type=$distance_type
    done
done

coverage combine $coverage_datafiles
