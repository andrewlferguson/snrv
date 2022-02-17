"""
Unit and regression test for the snrv package.
"""

import numpy as np
from scipy.linalg import eigh
import snrv.utils
import torch


def test_stable_symmetric_inverse():
    """
    Testing inversion of symmetric matrix routine.
    """
    A = np.array(
        [
            [2.33579236, 0.41457851, 1.0944027],
            [0.41457851, 1.52068338, 0.36892426],
            [1.0944027, 0.36892426, 1.88039527],
        ]
    )

    assert np.allclose(A, A.T)

    B = snrv.utils.stable_symmetric_inverse(torch.from_numpy(A))
    B = B.cpu().detach().numpy()

    AB = np.matmul(A, B)

    assert np.allclose(AB, np.eye(A.shape[0]))


def test_stable_symmetric_inverse_sqrt():
    """
    Testing inversion of symmetric matrix routine with sqrt return.
    """
    A = np.array(
        [
            [2.33579236, 0.41457851, 1.0944027],
            [0.41457851, 1.52068338, 0.36892426],
            [1.0944027, 0.36892426, 1.88039527],
        ]
    )

    assert np.allclose(A, A.T)

    B = snrv.utils.stable_symmetric_inverse(torch.from_numpy(A), ret_sqrt=True)
    B = B.cpu().detach().numpy()

    ABB = np.matmul(np.matmul(A, B), B)

    assert np.allclose(ABB, np.eye(A.shape[0]))


def test_gen_eig_chol():
    """
    Testing Cholesky factorization based solution to symmetric generalized eigenvalue problem.
    """
    C = np.array(
        [
            [2.33579236, 0.41457851, 1.0944027],
            [0.41457851, 1.52068338, 0.36892426],
            [1.0944027, 0.36892426, 1.88039527],
        ]
    )

    assert np.allclose(C, C.T)

    Q = np.array(
        [
            [1.4536383, 1.24153722, 1.66784209],
            [1.24153722, 1.70993988, 1.18609631],
            [1.66784209, 1.18609631, 2.38190968],
        ]
    )

    assert np.allclose(Q, Q.T)

    w, v = eigh(C, Q)
    idx = np.argsort(w)
    idx = np.flip(idx)
    w = w[idx]
    v = v[:, idx]

    w2, v2 = snrv.utils.gen_eig_chol(torch.from_numpy(C), torch.from_numpy(Q))
    w2 = w2.cpu().detach().numpy()
    v2 = v2.cpu().detach().numpy()

    assert np.allclose(w, w2) and np.allclose(v, v2)
