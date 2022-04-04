import numpy as np
import torch
from copy import deepcopy


__all__ = ["implied_timescales"]


def implied_timescales(
    model, lags, data, ln_dynamical_weight=None, thermo_weight=None, random_seed=42
):
    """
    Compute implied timescales for a SNRV model object at different lag times.

    Independent models are trained at separate lagtimes and the implied timescales
    are calcaculted as:

        t(lag) = -lag / log(eval)

    Examples::
        >>> from snrv.validation import implied_timescales
        >>> from snrv.plots import plot_timescales
        >>> lags = [10, 100, 1000]
        >>> timescales = implied_timescale(snrv_model, lags, training_data)
        >>> plot_timescales(lags, timescales)

    Parameters
    ----------
    model : Snrv
        Snrv model object that will be used to calculate the implied timescales

    lags : list or iterable, n_lags
        list of different lagtimes timescales are calcaulted for

    data : list or torch.tensor, n x dim, n = observations, dim = dimensionality of trajectory featurization
        trajectory data used to train the SNRV model

    ln_dynamical_weight : list or torch.tensor, n, n = observations, default = None
        accumulated sum of the log Girsanov path weights between frames in the trajectory;
        Girsanov theorem measure of the probability of the observed sample path under a target potential
        relative to that which was actually observed under the simulation potential;
        identically unity (no reweighting rqd) for target potential == simulation potential and code as None;
        Ref.: Kieninger and Keller J. Chem. Phys 154 094102 (2021)  https://doi.org/10.1063/5.0038408

    thermo_weight : list or torch.tensor, n, n = observations, default = None
        thermodynamic weights for each trajectory frame corresopnding to Boltzmann factor of the bias potential
        representing a state reweighting from the simulation to the target Hamiltonian for that single frame;
        thermo_weight(x) = exp(-beta*U_bias(x)) [Formally thermo_weight(x) = exp(-beta*U_bias(x)) * Z_sim/Z_target
        but partition function ratio is a constant that cancels either side of VAC generalized eigenproblem]

    random_seed : int, default = 42
        random seed

    Return
    ------
    timescales: np.ndarray, n_lags x (output_size - 1)
        implied timescales calcaulted for each lagtime. First timescale corresponding to the stationary process
        is omitted.
    """

    timescales = list()

    for lag in lags:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        model_train = deepcopy(model)

        model_train.fit(
            data,
            lag,
            ln_dynamical_weight=ln_dynamical_weight,
            thermo_weight=thermo_weight,
        )
        evals = model_train.evals.cpu().detach().numpy()
        timescales.append(-lag / np.log(evals))

    # omit first timescale corresponding to stationary process
    timescales = np.concatenate([e[1:].reshape(1, -1) for e in timescales])
    return timescales
