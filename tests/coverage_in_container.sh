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

site_pkg_dir=/venv/lib/python3.13/site-packages
vision_toolkit_modname=vision_toolkit$vision_toolkit_version_suffix
coverage_run="coverage run --source $site_pkg_dir/$vision_toolkit_modname"

datasets="hollywood2 zemblys"

for dataset in $datasets; do
    coverage_datafile=coverage_$dataset.sqlite
    coverage_datafiles="$coverage_datafile $coverage_datafiles"
    echo $coverage_run --data-file=$coverage_datafile run.py $VISION_TOOLKIT_VERSION $dataset 1
    $coverage_run --data-file=$coverage_datafile run.py $VISION_TOOLKIT_VERSION $dataset 1
done

cd /src/tests

# do not forget trailing slash
vision_toolkit_srcdir="/src/src/$vision_toolkit_modname/"

for coverage_datafile in $coverage_datafiles; do
    cp $coverage_datafile{,.rename}
    sqlite3 $coverage_datafile.rename \
        "UPDATE file SET path =  '$vision_toolkit_srcdir' || SUBSTR(path, INSTR(path, '$vision_toolkit_modname') + LENGTH('$vision_toolkit_modname') + 1)"
done

cd $vision_toolkit_srcdir

for coverage_datafile in $coverage_datafiles; do
    ln -s /src/tests/$coverage_datafile.rename .
done

coverage combine coverage_*.sqlite*
coverage report -m
coverage json

cp coverage.json /src/tests
