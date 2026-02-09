#!/bin/env bash

test_name="hollywood2"

for version in 1 2; do
    ./run.py $version $test_name 3 &
    pids_to_wait="$pid_to_wait $!"
done
    
wait $pids_to_wait

git diff --no-index results/$test_name/v{1,2}/report.md
