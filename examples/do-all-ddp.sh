#!/bin/bash
set -e

examples_train=(
   tensor-train
   # tensor-train-time_dependent is not supported in DDP
   nn-train
   nn-train-conditioned
   K-tensort-train
   K-nn-train
)

N_NODES=1
N_GPUS_PER_NODE=2

# Train with DDP
for example in "${examples_train[@]}"; do
  echo "Running example: $example in DDP mode"
  torchrun --nnodes=$N_NODES --nproc-per-node=$N_GPUS_PER_NODE \
    -m ml_pic_collision_operators.main examples/$example.yaml \
    examples-train-ddp \
    $example \
    mlruns \
    --run_overwrite
done

# Test is without DDP
examples_test=(
   tensor-test
   nn-test
   nn-test-conditioned
   K-tensor-test
   K-nn-test
)

for example in "${examples_test[@]}"; do
  echo "Running example: $example"
  sed -e "s|examples-train|examples-train-ddp|g" "examples/$example.yaml" > "temp.yaml"
  mlpic_run temp.yaml \
    examples-test-ddp \
    $example \
    mlruns \
    --run_overwrite
done
rm temp.yaml