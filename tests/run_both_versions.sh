#!/bin/env bash

test_names="hollywood2 zemblys"

for test_name in test_names; do
    for version in 1 2; do
	$(dirname $0)/run.py --predictions $version $test_name 10000 &
    done
done

wait

git diff --no-index ./results/$test_name/v{1,2}/report.md
git diff --no-index ./results/$test_name/v{1,2}/predictions
