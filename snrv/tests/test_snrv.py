"""
Unit and regression test for the snrv package.
"""

# Import package, test suite, and other packages as needed
import sys

import numpy as np
import snrv
import torch
import copy


def test_snrv_imported():
    """
    Testing import successful.
    """
    assert "snrv" in sys.modules


def test_fit():
    """
    Testing fit call updates model parameters.
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
    n_epochs = 10
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
        device=device
    )
    model = model.to(device)

    # initial parameters
    params_before = copy.deepcopy(list(model.parameters()))

    # fitting model
    lag_n = 10
    model.fit(traj_x_featurized, lag_n, standardize=True)

    # final parameters
    params_after = copy.deepcopy(list(model.parameters()))

    # checking (at least one) parameter in every parameter block has been updated
    for ii in range(len(params_before)):
        assert (params_before[ii] != params_after[ii]).any()
