import jax
import jax.numpy as jnp
import mlflow
import equinox as eqx
import optax  # type: ignore[import-untyped]
import tempfile

from jaxtyping import PyTree
from tqdm import tqdm

from src.datasets import *
from src.dataloaders import *
from src.models import *
from src.logging import log_equinox_model


def train(cfg):

    mlflow.log_params(cfg)

    hist = {}
    # hist["loss_physics"] = []
    # hist["loss_reg"] = []
    hist["loss_total"] = []

    train_dataset = BaseDataset(
        folder=cfg["train"]["dataset"]["train"]["folders"][0],
        step_size=cfg["train"]["dataset"]["step_size"],
    )

    print("Dataset Size:", len(train_dataset))

    train_dataloader = BaseDataLoader(
        train_dataset,
        batch_size=len(train_dataset),
    )

    model_kwargs = {
        "grid_size": train_dataset.grid_size,
        "grid_dx": train_dataset.grid_dx,
    }

    model = FokkerPlanck2D(**model_kwargs)
    mlflow.log_params({"model_kwargs": model_kwargs})

    print("Model:", model)

    optim = optax.adamw(float(cfg["train"]["optimizer"]["learning_rate"]))

    # only model jax arrays are optimized
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    def loss_fn(model, y_pred, y):
        y_pred = jax.vmap(model)(x)
        return jnp.mean(jnp.square(y_pred - y))

    @eqx.filter_jit
    def train_step(model, opt_state: PyTree, x: jax.Array, y: jax.Array):
        train_loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, train_loss

    with tempfile.TemporaryDirectory() as tmp_dir:
        step = 0
        for epoch in tqdm(range(cfg["train"]["epochs"])):
            train_loss_epoch = 0
            for x, y in train_dataloader:
                model, opt_state, train_loss_step = train_step(model, opt_state, x, y)
                train_loss_epoch += train_loss_step * len(x) / len(train_dataset)
                mlflow.log_metric("train_loss_step", train_loss_step, step=step)
                step += 1
            mlflow.log_metric("train_loss", train_loss_epoch, step=epoch)
            log_equinox_model(model, tmp_dir)
