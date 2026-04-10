#!/bin/bash
# Choose an example from the available
example=$1
echo "Running example: $example"

# Find the directory where this script is stored
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Command follows the structure
# mlpic_run path_to_yaml.yaml \
#   experiment_name \
#   run_name \
#   mlflow_directory \
mlpic_run $SCRIPT_DIR/$example.yaml \
    examples-train \
    $example \
    mlruns \
    --run_overwrite # overwrite if existing