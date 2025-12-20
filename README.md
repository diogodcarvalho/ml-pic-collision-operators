# ml-pic-collision-operators

## Installation

### Optional
Create a virtual environment to avoid clashes with other environments

```
python3 -m venv .venv
```

And activate it to ensure packages will be installed under it

```
source .venv/bin/activate
```

### Mandatory

First clone the repository

```
git clone git@github.com:diogodcarvalho/ml-pic-collision-operators.git
cd ml-pic-collision-operators
```

On `rigel`, to install the package dependencies with the correct PyTorch GPU version, you simply run:

```
pip install -r requirements/rigel.txt
```

On a different machine (e.g., your laptop), you can still just run:

```
pip install -e .
```

And it will download the standard CPU or default CUDA version of Torch from the regular Python repository (PyPI).

You can check that PyTorch is correctly installed and using CPU/GPU with 

```
python3 check_torch.py
```

## Quickstart

### Training / Testing Models

To train/test models, one runs in the terminal a command in the form:

```
mlpic_run <config_file.yaml> <experiment_name> <run_name> <mlflow_dir>
```

where:
- `config_file.yaml`: is a configuration file
- `experiment_name`: the name of the experiment to associate this run in MLFlow
- `run_name`: the name of this particular run
- `mflow_dir`: the directory where MLFlow is logging the files to

A few examples of configuration files to train/test different models are provided in `examples/`
together with an example dataset and bash scripts which further illustrate the command line interface.

You can quickly check that everything is working well by running one of the example scripts:
```
chmod +x ./examples/tensor-train.sh
./examples/tensor-train.sh
```

### MLFlow UI

The code logs all metrics / checkpoints / videos / figures / etc. to an mlflow server
(by default a directory in your machine).

To access an interactive MLFlow UI which shows the existing logged experiments you need to do as follows.

For a remote machine run an mlflow server in the background with:

```
mlflow server --host 127.0.0.1 --port 8088 \
  --backend-store-uri file:/path/to/mlruns \
  --default-artifact-root file:/path/to/mlruns &
```

and access it by oppening `http://localhost:8088` in your browser (you can also select a different port value if desired).

You should then see an interface similar to the one below with your experiments.

<p align="center">
  <img src="docs/mlflow_example.png"/>
</p>

