"""
Unit and regression test for the snrv package.
"""

# Import package, test suite, and other packages as needed

import numpy as np
import snrv
import torch
import snrv.validation
import snrv.plots


def test_plot_timescales():
    """
    Testing implied timescales plotting
    """

    np.random.seed(42)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # synthetic training data
    x = np.linspace(0, 10 * np.pi, 1000)
    traj_x = np.concatenate(
        (np.reshape(np.sin(x), (-1, 1)), np.reshape(np.cos(x), (-1, 1))), 1
    )
    traj_x_featurized = traj_x
    traj_x_featurized = torch.from_numpy(traj_x_featurized).float()

    # building model
    input_size = traj_x_featurized.size()[1]
    output_size = 4
    hidden_depth = 2
    hidden_size = 100
    batch_norm = True
    dropout_rate = 0.0
    lr = 1e-2
    weight_decay = 0.0
    val_frac = 0.10
    n_epochs = 3
    batch_size = 100
    VAMPdegree = 2
    is_reversible = False

    model = snrv.Snrv(
        input_size,
        output_size,
        hidden_depth=hidden_depth,
        hidden_size=hidden_size,
        batch_norm=batch_norm,
        dropout_rate=dropout_rate,
        lr=lr,
        weight_decay=weight_decay,
        val_frac=val_frac,
        n_epochs=n_epochs,
        batch_size=batch_size,
        VAMPdegree=VAMPdegree,
        is_reversible=is_reversible,
    )
    model = model.to(device)

    lags = [2, 5]

    timescales = snrv.validation.implied_timescales(model, lags, traj_x_featurized)
    snrv.plots.plot_timescales(lags, timescales)
