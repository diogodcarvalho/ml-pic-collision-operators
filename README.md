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
