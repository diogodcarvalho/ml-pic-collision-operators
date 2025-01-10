import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jaxtyping import PyTree
from tqdm import tqdm

from src.datasets import *
from src.dataloaders import *
from src.models import *


def train(cfg):

    hist = {}
    # hist["loss_physics"] = []
    # hist["loss_reg"] = []
    hist["loss_total"] = []

    train_dataset = BaseDataset(
        folder=cfg["train"]["dataset"]["train"]["folders"][0],
        step_size=cfg["train"]["dataset"]["step_size"],
    )

    print(len(train_dataset))

    train_dataloader = BaseDataLoader(
        train_dataset,
        batch_size=len(train_dataset),
        # **cfg["train"]["dataloader_kwargs"],
    )

    model = FokkerPlanck2D(
        grid_size=train_dataset.grid_size,
        grid_dx=(1.0, 1.0),  # train_dataset.grid_dx,
    )

    print(model)

    optim = optax.adamw(float(cfg["train"]["optimizer"]["learning_rate"]))

    # only model jax arrays are optimized
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    def loss_fn(model, y_pred, y):
        y_pred = jax.vmap(model)(x)
        return jnp.mean(jnp.square(y_pred - y))

    @eqx.filter_jit
    def train_step(model, opt_state: PyTree, x: jax.Array, y: jax.Array):
        train_loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        print(train_loss, grads)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, train_loss

    step = 0
    for epoch in tqdm(range(cfg["train"]["epochs"])):
        for x, y in train_dataloader:
            model, opt_state, train_loss = train_step(model, opt_state, x, y)
            # print(f"{step=}, train_loss={train_loss.item()}, ")
            step += 1

    print(model)
    model.plot("a.png")


# # nop norm - 2 layers
# lr = 1*1e-2

# optimizer = optim.Adam([Ax, Ay, Bxx, Bxy, Byx, Byy], lr)

# ## norm
# # l1
# # alpha_reg = 5*1e-5
# # alpha_reg = 1e-6
# # l2
# # alpha_reg = 5*1e-9
# ## no norm
# # alpha_reg = 1e+2
# alpha_reg = 0

# for epoch in range(num_epochs + 1):

#     dF_pred = predict(F_train, Ax, Ay, Bxx, Bxy, Byx, Byy, dx, dy, n_steps)
#     loss = compute_loss(dF_train, dF_pred, n_steps)
#     loss_reg = alpha_reg * compute_reg_loss([Ax, Ay, Bxx, Bxy, Byx, Byy])

#     loss_total = loss + loss_reg

#     hist["loss_physics"].append(loss.cpu().detach().numpy())
#     hist["loss_reg"].append(loss_reg.cpu().detach().numpy())
#     hist["loss_total"].append(loss_total.cpu().detach().numpy())

#     optimizer.zero_grad()
#     loss_total.backward()
#     optimizer.step()

#     if epoch % 10 == 0:
#         print(f"{epoch} total:{loss_total:.1e} phys:{loss:.1e}")
