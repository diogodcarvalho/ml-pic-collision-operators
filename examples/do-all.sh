#!/bin/bash
set -e
rm -rf mlruns

CUDA_DEVICE=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


examples_train=(
   tensor-train
   tensor-train-time_dependent
   nn-train
   nn-train-conditioned
   K-tensor-train
   K-nn-train
)

for example in "${examples_train[@]}"; do
  echo "Running example: $example"
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE mlpic_run $SCRIPT_DIR/$example.yaml \
    examples-train \
    $example \
    mlruns \
    --run_overwrite
done


examples_test=(
   tensor-test
   tensor-test-time_dependent
   nn-test
   nn-test-conditioned
   K-tensor-test
   K-nn-test
)

for example in "${examples_test[@]}"; do
  echo "Running example: $example"
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE mlpic_run $SCRIPT_DIR/$example.yaml \
    examples-test \
    $example \
    mlruns \
    --run_overwrite
done