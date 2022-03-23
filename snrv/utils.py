import torch

__all__ = [
    "stable_symmetric_inverse",
    "gen_eig_chol",
    "accumulate_correlation_matrices",
]


def stable_symmetric_inverse(A, ret_sqrt=False):
    """
    Utility function to return stable inverse A^(-1) or sqrt of inverse A^(-0.5) of a symmetric matrix A
    Ref.: https://github.com/markovmodel/deeptime

    Proceeds by diagonalizing matrix, setting eigenvalues smaller than threshold to zero, performing operations
    on diagonlized matrix, reprojecting back to original basis

    Parameters
    ----------
    A : torch.tensor
        symmetric matrix

    ret_sqrt : bool, default = False
        indicator flag as to whether to return square root of inverse

    Return
    ------
    B : torch.tensor
        stable inverse approximation to A^(-1) or A^(-0.5) depending on ret_sqrt flag
    """
    assert torch.allclose(A, A.t(), atol=1e-6)

    w, V = torch.linalg.eigh(A)

    w = torch.where(w > torch.finfo(torch.float32).eps, w, torch.zeros_like(w))

    if ret_sqrt:
        B = torch.matmul(V, torch.matmul(torch.diag(w ** (-0.5)), V.t()))
    else:
        B = torch.matmul(V, torch.matmul(torch.diag(w ** (-1.0)), V.t()))

    return B


def gen_eig_chol(C, Q):
    """
    Solution of symmetric generalized eigenvalue problem using Cholesky decomposition to convert to regular
    symmetric eigenvalue problem
    Ref: Sidky, Chen, Ferguson J. Chem. Phys. 150, 214114 (2019); doi: 10.1063/1.5092521

    C*v_i = w_i*Q*v_i (generalized eigenvalue problem)

    Q = L*LT
    Ctilde = Linv*C*LTinv
    vtilde_i = LT*v_i

    Ctilde*vtilde_i = w_i*v_i (regular eigenvalue problem)
    v_i = LTinv*vtilde_i

    Parameters
    ----------
    C : torch.tensor
        symmetric matrix

    Q : torch.tensor
        symmetric matrix

    Return
    ------
    w : torch.tensor
        eigenvalues in non-ascending order

    v : torch.tensor
        eigenvectors in non-ascending order
    """
    # assert torch.allclose(C, C.t(), atol=1e-6)
    # assert torch.allclose(Q, Q.t(), atol=1e-6)

    # Cholesky
    # N.B. torch.linalg.cholesky checks for Hermitian matrix automatically and throws runtime error if violated
    
    L = torch.linalg.cholesky(Q)
    Linv = torch.linalg.inv(L)
    LTinv = torch.linalg.inv(L.t())

    # Ctilde
    # N.B. Ctilde guaranteed to be symmetric if C is symmetric:
    # Let A = L^-1 * C * (L^T)^-1 = L^-1 * C * (L^-1)^T.
    # A^T = [L^-1 * C * (L^-1)^T]^T = L^-1 * C^T * (L^-1)^T = A, if C = C^T
    Ctilde = torch.matmul(Linv, torch.matmul(C, LTinv))

    # regular symmetric eigenvalue problem
    w, vtilde = torch.linalg.eigh(Ctilde)

    # correcting to generalized eigenvalue eigenvectors
    v = torch.matmul(LTinv, vtilde)

    # reordering to non-ascending
    w = torch.flip(w, [0])
    v = torch.flip(v, [1])

    return w, v


def accumulate_correlation_matrices(z_t0, z_tt, pathweight, C00, C01, C10, C11):
    """
    Accumulating instantaneous and time-lagged correlations in z_t0 and z_tt into pre-existing C00, C01, C10, C11

    Parameters
    ----------
    z_t0 : torch.tensor, n x n_comp, n = observations, n_comp = number of basis functions produced by network
        trajectory projected into basis functions learned by SNRV encoder

    z_tt : torch.tensor, n x n_comp, n = observations, n_comp = number of basis functions produced by network
        time-lagged trajectory of same length as trajectory projected into basis functions learned by SNRV encoder

    pathweight : float tensor, n = observations
        pathweights from Girsanov theorem between time lagged observations;
        identically unity (no reweighting rqd) for target potential == simulation potential

    C00 : torch.tensor, n_comp x n_comp
        correlation of z_t0 with z_t0

    C01 : torch.tensor, n_comp x n_comp
        correlation of z_t0 with z_tt

    C10 : torch.tensor, n_comp x n_comp
        correlation of z_tt with z_t0

    C11 : torch.tensor, n_comp x n_comp
        correlation of z_tt with z_tt

    Return
    ------
    C00 : torch.tensor, n_comp x n_comp
        correlation of z_t0 with z_t0

    C01 : torch.tensor, n_comp x n_comp
        correlation of z_t0 with z_tt

    C10 : torch.tensor, n_comp x n_comp
        correlation of z_tt with z_t0

    C11 : torch.tensor, n_comp x n_comp
        correlation of z_tt with z_tt
    """

    assert z_t0.size()[0] == z_tt.size()[0] == pathweight.size()[0]
    assert z_t0.size()[1] == z_tt.size()[1]

    # Let R = diag(pathweight), R = R^T
    # Let W = tile_vertical(pathweight)
    # C01 = X^T * (R * Y) = (R * X)^T * Y = X^T * (W .* Y) = (W .* X)^T * Y
    # - R version more elegant, W version less memory intensive
    W = torch.tile(torch.reshape(pathweight, (-1, 1)), (1, z_tt.size()[1]))
    z_tt_r = torch.multiply(W, z_tt)
    z_t0_r = torch.multiply(W, z_t0)
    C00 += torch.matmul(z_t0.t(), z_t0_r)
    C01 += torch.matmul(z_t0.t(), z_tt_r)
    C10 += torch.matmul(z_tt.t(), z_t0_r)
    C11 += torch.matmul(z_tt.t(), z_tt_r)

    return C00, C01, C10, C11
