#!/bin/bash
set -e
rm -rf mlruns

CUDA_DEVICE=1

examples_train=(
   tensor-train
   nn-train
)

for example in "${examples_train[@]}"; do
  echo "Running example: $example"
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE mlpic_run examples/$example.yaml \
    examples-train \
    $example \
    mlruns \
    --run_overwrite
done


examples_test=(
   tensor-test
   nn-test
)

for example in "${examples_test[@]}"; do
  echo "Running example: $example"
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE mlpic_run examples/$example.yaml \
    examples-test \
    $example \
    mlruns \
    --run_overwrite
done