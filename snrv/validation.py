import numpy as np
import torch
from copy import deepcopy


__all__ = ["implied_timescales"]


def implied_timescales(
    model,
    lags,
    data,
    ln_dynamical_weight=None,
    thermo_weight=None,
    random_seed=42,
    cross_validation_folds=-1,
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

    cross_validation_folds : int, default = -1
        number of cross validation folds used to estimate uncertainty in the timescales. By default is -1, in which
        case no cross-validation is performed. All available data is used to estimate the timescale mean, while
        timescales corresponding to each data fold are also output.

    Return
    ------
    timescales: np.ndarray, n_lags x (output_size - 1)
        implied timescales calcaulted for each lagtime. First timescale corresponding to the stationary process
        is omitted.

    timescales_cv_folds: np.ndarray, n_lags x cross_validation_folds x (output_size - 1)
        only returned if cross_validation_folds > 1. Implied timescales calcaulted for each lagtime for each cross
        validation fold. First timescale corresponding to the stationary process is omitted.
    """

    if (
        cross_validation_folds == 0
        or cross_validation_folds == 1
        or cross_validation_folds < -1
    ):
        raise ValueError(
            """
            Number of cross_validation_folds must be greater than 1 to perform cross validation, or equal to -1
            for no cross validation to be performed.
            """
        )

    if cross_validation_folds != -1:
        timescales_cv_folds = list()

        if isinstance(data, torch.Tensor):
            cv_idxs = torch.arange(data.shape[0])
            cv_size = data.shape[0] // cross_validation_folds
        elif isinstance(data, list):
            cv_idxs = [torch.arange(d.shape[0]) for d in data]
            cv_size = [d.shape[0] // cross_validation_folds for d in data]

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

        if cross_validation_folds != -1:
            timescale_cv = list()
            for _ in range(cross_validation_folds):
                fold_idxs = np.random.choice(
                    cross_validation_folds, size=cross_validation_folds, replace=True,
                )
                if isinstance(data, torch.Tensor):
                    idxs = torch.cat(
                        [cv_idxs[n * cv_size : (n + 1) * cv_size] for n in fold_idxs]
                    )
                    cv_data = data[idxs]
                    cv_ln_dynamical_weight = (
                        ln_dynamical_weight[idxs]
                        if ln_dynamical_weight is not None
                        else None
                    )
                    cv_thermo_weight = (
                        thermo_weight[idxs] if thermo_weight is not None else None
                    )
                elif isinstance(data, list):
                    idxs = [
                        torch.cat(
                            [
                                cv_idxs[n * cv_size : (n + 1) * cv_size]
                                for n in fold_idxs
                            ]
                        )
                        for inds, size in zip(cv_idxs, cv_size)
                    ]
                    cv_data = [d[inds] for d, inds in zip(data, idxs)]
                    cv_ln_dynamical_weight = (
                        [w[inds] for w, inds in zip(ln_dynamical_weight, idxs)]
                        if ln_dynamical_weight is not None
                        else None
                    )
                    cv_thermo_weight = (
                        [w[inds] for w, inds in zip(thermo_weight, idxs)]
                        if thermo_weight is not None
                        else None
                    )

                model_train = deepcopy(model)

                model_train.fit(
                    cv_data,
                    lag,
                    ln_dynamical_weight=cv_ln_dynamical_weight,
                    thermo_weight=cv_thermo_weight,
                )
                evals = model_train.evals.cpu().detach().numpy()
                timescale_cv.append(-lag / np.log(evals))

            # omit first timescale corresponding to stationary process
            timescales_cv_folds.append(
                np.concatenate([e[1:].reshape(1, -1) for e in timescale_cv])
            )

    # omit first timescale corresponding to stationary process
    timescales = np.concatenate([e[1:].reshape(1, -1) for e in timescales])

    if cross_validation_folds != -1:
        timescales_cv_folds = np.concatenate([t[None] for t in timescales_cv_folds])
        return timescales, timescales_cv_folds
    else:
        return timescales
