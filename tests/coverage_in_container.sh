#!/bin/env bash

if [[ VISION_TOOLKIT_VERSION == "1" ]]; then
    VISION_TOOLKIT_VERSION_SUFFIX=""
else
    VISION_TOOLKIT_VERSION_SUFFIX=$VISION_TOOLKIT_VERSION
fi

SITE_PKG_DIR=/venv/lib/python3.13/site-packages
VISION_TOOLKIT_MODNAME=vision_toolkit$VISION_TOOLKIT_VERSION_SUFFIX
coverage_run="coverage run --source $SITE_PKG_DIR/$VISION_TOOLKIT_MODNAME"

$coverage_run --data-file=coverage_hollywood2.sqlite run.py hollywood2 1
$coverage_run --data-file=coverage_zemblys.sqlite run.py zemblys 1

pkgname="vision_toolkit"

coverage_datafiles="coverage_hollywood2.sqlite coverage_zemblys.sqlite"

cd /src/tests

# do not forget trailing slash
VISION_TOOLKIT_SRCDIR="/src/src/$VISION_TOOLKIT_MODNAME/"

for coverage_datafile in $coverage_datafiles; do
    cp  $coverage_datafile{,.rename}
    sqlite3 $coverage_datafile.rename \
        "UPDATE file SET path =  '$VISION_TOOLKIT_SRCDIR' || SUBSTR(path, INSTR(path, '$pkgname') + LENGTH('$pkgname') + 1)"
done

cd $VISION_TOOLKIT_SRCDIR

for coverage_datafile in $coverage_datafiles; do
    ln -s /src/tests/$coverage_datafile.rename .
done

coverage combine coverage_*.sqlite*
coverage report -m
coverage json

cp coverage.json /src/tests
