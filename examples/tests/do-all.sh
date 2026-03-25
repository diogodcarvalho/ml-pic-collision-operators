#!/bin/bash
set -e
rm -rf mlruns

examples=(
   tensor-train-xs
   nn-train-xs
)

# # Non-DPP
# CUDA_DEVICE=1
# # Not compiled
# for example in "${examples[@]}"; do
#   echo "Running example: $example"
#   CUDA_VISIBLE_DEVICES=$CUDA_DEVICE mlpic_run examples/tests/$example.yaml \
#     tests \
#     $example \
#     mlruns \
#     --run_overwrite
# done
# Compiled (does not work in rigel)
# for example in "${examples[@]}"; do
#   echo "Running example: $example"
#   CUDA_VISIBLE_DEVICES=$CUDA_DEVICE mlpic_run examples/tests/$example.yaml \
#     tests \
#     $example-compiled \
#     mlruns \
#     --run_overwrite \
#     --compile_model
# done

# DDP
N_NODES=1
N_GPUS_PER_NODE=2
# Not compiled
for example in "${examples[@]}"; do
  echo "Running example: $example in DDP mode"
  torchrun --nnodes=$N_NODES --nproc-per-node=$N_GPUS_PER_NODE \
    -m ml_pic_collision_operators.main examples/tests/$example.yaml \
    tests \
    $example-ddp \
    mlruns \
    --run_overwrite
done
# # Compiled
# for example in "${examples[@]}"; do
#   echo "Running example: $example in DDP mode"
#   torchrun --nnodes=$N_NODES --nproc-per-node=$N_GPUS_PER_NODE \
#     -m ml_pic_collision_operators.main examples/tests/$example.yaml \
#     tests \
#     $example-ddp-compiled \
#     mlruns \
#     --run_overwrite
# done