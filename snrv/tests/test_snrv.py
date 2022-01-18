"""
Unit and regression test for the snrv package.
"""

# Import package, test suite, and other packages as needed
import sys

import numpy as np
from scipy.linalg import eigh
import pytest
import snrv
import torch
import copy


def test_snrv_imported():
    """
    Testing import successful.
    """
    assert "snrv" in sys.modules


def test__stable_symmetric_inverse():
    """
    Testing inversion of symmetric matrix routine.
    """
    A = np.array([[2.33579236, 0.41457851, 1.0944027],
                  [0.41457851, 1.52068338, 0.36892426],
                  [1.0944027, 0.36892426, 1.88039527]])

    assert np.allclose(A, A.T)

    B = snrv.Snrv._stable_symmetric_inverse(torch.from_numpy(A))
    B = B.cpu().detach().numpy()

    AB = np.matmul(A, B)

    assert np.allclose(AB, np.eye(A.shape[0]))


def test__stable_symmetric_inverse_sqrt():
    """
    Testing inversion of symmetric matrix routine with sqrt return.
    """
    A = np.array([[2.33579236, 0.41457851, 1.0944027],
                  [0.41457851, 1.52068338, 0.36892426],
                  [1.0944027, 0.36892426, 1.88039527]])

    assert np.allclose(A, A.T)

    B = snrv.Snrv._stable_symmetric_inverse(torch.from_numpy(A), ret_sqrt=True)
    B = B.cpu().detach().numpy()

    ABB = np.matmul(np.matmul(A, B), B)

    assert np.allclose(ABB, np.eye(A.shape[0]))


def test__gen_eig_chol():
    """
    Testing Cholesky factorization based solution to symmetric generalized eigenvalue problem.
    """
    C = np.array([[2.33579236, 0.41457851, 1.0944027],
                  [0.41457851, 1.52068338, 0.36892426],
                  [1.0944027, 0.36892426, 1.88039527]])

    assert np.allclose(C, C.T)

    Q = np.array([[1.4536383, 1.24153722, 1.66784209],
                  [1.24153722, 1.70993988, 1.18609631],
                  [1.66784209, 1.18609631, 2.38190968]])

    assert np.allclose(Q, Q.T)

    w, v = eigh(C, Q)
    idx = np.argsort(w)
    idx = np.flip(idx)
    w = w[idx]
    v = v[:, idx]

    w2, v2 = snrv.Snrv._gen_eig_chol(torch.from_numpy(C), torch.from_numpy(Q))
    w2 = w2.cpu().detach().numpy()
    v2 = v2.cpu().detach().numpy()

    assert np.allclose(w, w2) and np.allclose(v, v2)


def test_fit():
    """
    Testing fit call updates model parameters.
    """

    np.random.seed(42)
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # synthetic training data
    x = np.linspace(0, 10 * np.pi, 1000)
    traj_x = np.concatenate((np.reshape(np.sin(x), (-1, 1)), np.reshape(np.cos(x), (-1, 1))), 1)
    traj_x_featurized = traj_x
    traj_x_featurized = torch.from_numpy(traj_x_featurized).float()

    # building model
    input_size = traj_x_featurized.size()[1]
    output_size = 4
    hidden_depth = 2
    hidden_size = 100
    batch_norm = True
    dropout_rate = 0.
    lr = 1E-2
    weight_decay = 0.
    val_frac = 0.10
    n_epochs = 10
    batch_size = 100
    VAMPdegree = 2
    is_reversible = False

    model = snrv.Snrv(input_size, output_size, hidden_depth=hidden_depth, hidden_size=hidden_size,
                      batch_norm=batch_norm, dropout_rate=dropout_rate, lr=lr, weight_decay=weight_decay,
                      val_frac=val_frac, n_epochs=n_epochs, batch_size=batch_size,
                      VAMPdegree=VAMPdegree, is_reversible=is_reversible)
    model = model.to(device)

    # initial parameters
    params_before = copy.deepcopy(list(model.parameters()))

    # fitting model
    lag_n = 10
    model.fit(traj_x_featurized, lag_n)

    # final parameters
    params_after = copy.deepcopy(list(model.parameters()))

    # checking (at least one) parameter in every parameter block has been updated
    for ii in range(len(params_before)):
        assert (params_before[ii] != params_after[ii]).any()