#!/bin/env bash

coverage_run="coverage run --source /venv/lib/python3.13/site-packages/vision_toolkit"

$coverage_run --data-file=coverage_hollywood2.sqlite run.py hollywood2 1
$coverage_run --data-file=coverage_zemblys.sqlite run.py zemblys 1

pkgname="vision_toolkit"

coverage_datafiles="coverage_hollywood2.sqlite coverage_zemblys.sqlite"

cd /src/tests

for coverage_datafile in $coverage_datafiles; do
    cp  $coverage_datafile{,.rename}
    sqlite3 $coverage_datafile.rename \
        "UPDATE file SET path =  '/src/src/vision_toolkit/' || SUBSTR(path, INSTR(path, '$pkgname') + LENGTH('$pkgname') + 1)"
done

cd /src/src/vision_toolkit

for coverage_datafile in $coverage_datafiles; do
    ln -s /src/tests/$coverage_datafile.rename .
done

coverage combine coverage_*.sqlite*
coverage report -m
coverage json
