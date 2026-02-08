#!/bin/bash
set -e
rm -rf mlruns

examples=(
#  tensor-train-xs
   nn-train-xs
)
N_NODES=1
N_GPUS_PER_NODE=2

for example in "${examples[@]}"; do
  echo "Running example: $example"
  mlpic_run examples/tests/$example.yaml \
    tests \
    $example \
    mlruns
done

for example in "${examples[@]}"; do
  echo "Running example: $example in DDP mode"
  torchrun --nnodes=$N_NODES --nproc-per-node=$N_GPUS_PER_NODE \
    -m ml_pic_collision_operators.main examples/tests/$example.yaml \
    tests \
    $example \
    mlruns
done