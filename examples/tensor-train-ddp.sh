# The command follows the form:
# mlpic_run config_file.yaml \
#   experiment_name \
#   run_name \
#   mlflow_directory
N_NODES=1
N_GPUS_PER_NODE=2
torchrun --nnodes=$N_NODES --nproc-per-node=$N_GPUS_PER_NODE \
    -m ml_pic_collision_operators.main examples/tensor-train.yaml \
    examples-train \
    tensor-train-ddps \
    mlruns