#!/bin/env bash
set -e

if [ -z $VISION_TOOLKIT_VERSION ]; then
    echo "ERROR: env variable VISION_TOOLKIT_VERSION not defined."
    exit -1
fi

if [[ $VISION_TOOLKIT_VERSION == "1" ]]; then
    vision_toolkit_version_suffix=""
else
    vision_toolkit_version_suffix=$VISION_TOOLKIT_VERSION
fi


vision_toolkit_srcdir="$(dirname $0)/../"
# do not forget trailing slash
vision_toolkit_moddir=$vision_toolkit_srcdir/src/$vision_toolkit_modname/

site_pkg_dir=/venv/lib/python3.13/site-packages
vision_toolkit_modname=vision_toolkit$vision_toolkit_version_suffix
coverage_run="coverage run --source $site_pkg_dir/$vision_toolkit_modname"

datasets="hollywood2 zemblys"

cd $vision_toolkit_srcdir

for dataset in $datasets; do
    coverage_datafile=coverage_$dataset.sqlite
    coverage_datafiles="$coverage_datafile $coverage_datafiles"
    $coverage_run --data-file=$coverage_datafile $vision_toolkit_srcdir/tests/run.py $VISION_TOOLKIT_VERSION $dataset 1
done

for coverage_datafile in $coverage_datafiles; do
    sqlite3 $coverage_datafile \
        "UPDATE file SET path =  '$vision_toolkit_moddir' || SUBSTR(path, INSTR(path, '$vision_toolkit_modname') + LENGTH('$vision_toolkit_modname') + 1)"
done

cd $vision_toolkit_moddir

for coverage_datafile in $coverage_datafiles; do
    ln -s $vision_toolkit_srcdir/tests/$coverage_datafile .
done

coverage combine coverage_*.sqlite*
coverage report -m
coverage json

cp coverage.json $vision_toolkit_srcdir/tests
