#!/bin/bash
set -e
rm -rf mlruns
# ./examples/tensor-train.sh
./examples/tensor-train-ddp.sh
# ./examples/nn-train.sh
# ./examples/nn-train-ddp.sh
# ./examples/tensor-test.sh
./examples/tensor-test-ddp.sh
# ./examples/nn-test.sh
# ./examples/nn-test-ddp.sh 