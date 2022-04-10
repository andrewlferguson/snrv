import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import warnings

__all__ = ["plot_timescales"]


# inspired in part from: https://github.com/markovmodel/PyEMMA/blob/devel/pyemma/plots/timescales.py
def plot_timescales(
    lags,
    timescales,
    timescales_cv_folds=None,
    ax=None,
    xlog=False,
    ylog=True,
    n_timescales=-1,
    n_processes=-1,
    axis_units="frames",
):
    """
    Utility function for plotting implied timescales

    Example::
        >>> from snrv.validation import implied_timescales
        >>> from snrv.plots import plot_timescales
        >>> lags = [10, 100, 1000]
        >>> timescales = implied_timescale(snrv_model, lags, training_data)
        >>> plot_timescales(lags, timescales)

    Can also be used to plot cross validation errors

    Example::
        >>> from snrv.validation import implied_timescales
        >>> from snrv.plots import plot_timescales
        >>> lags = [10, 100, 1000]
        >>> timescales, timescales_cv = implied_timescale(snrv_model, lags, training_data, cross_validation_folds=5)
        >>> plot_timescales(lags, timescales, timescales_cv)

    Parameters
    ----------
    lags : list or np.ndarray, n_lags
        lag times associated with each timescale

    timescales : np.ndarray, n_lags x n_timescales
        implied timescales to be plotted

    timescales_cv_folds : np.ndarray, n_lags x cross_validation_folds x n_timescales, default = None
        if not None, plots separate set of timescales corresponding to the cross validation data.
        Estimates uncertainty in the implied timescales by calcaulting 95% confidence
        interval for each timescale from the `cross_validation_folds` and providing error bars to
        the associated cross validation means.

    ax : matplotlib Axes object, defulat = None
        the axes to plot. If None new Axes will be created to plot

    xlog : bool, defulat = False
        whether the x-axis should be plotted on a log scale

    ylog : bool, default = True
        whether the y-axis should be plotted on a log scale

    n_timescales : int, default = -1
        number of timescales to plot, if set to -1, all timescales are shown

    n_processes : int, default = -1
        number of processes to plot, if set to -1, all processes are shown

    axis_units : str, default = 'frames'
        modify units shown in the x-axis and y-axis labels, by default will be 'frames'

    Return
    ------
    ax: matplotlib Axes object
        Axes object that contains the plot
    """

    if isinstance(lags, list):
        lags = np.array(lags)

    if ax is None:
        ax = plt.gca()

    ax.grid()

    srt = np.argsort(lags)
    if n_timescales != -1:
        srt = srt[:n_timescales]

    if n_processes == -1:
        n_processes = timescales.shape[-1]

    colors = ["blue", "red", "green", "cyan", "purple", "orange", "violet"]

    for i in range(n_processes):
        nan_mask = ~np.isnan(timescales[..., i][srt])
        ax.plot(
            lags[srt][nan_mask],
            timescales[..., i][srt][nan_mask],
            color=colors[i % len(colors)],
        )
        ax.scatter(
            lags[srt][nan_mask],
            timescales[..., i][srt][nan_mask],
            color=colors[i % len(colors)],
        )

        if timescales_cv_folds is not None:
            mean = list()
            lower_bounds = list()
            upper_bounds = list()
            for a in timescales_cv_folds[..., i][srt][nan_mask]:
                if np.isnan(a).any():
                    warnings.warn(
                        """
                        Some cross validation folds contain NaN timescales. Uncertainties for timescales
                        will be estimated from the subset of values that are not NaN. Associated
                        uncertainties that yield negative lower bounds will not be plotted.
                        """
                    )
                low, high = st.t.interval(
                    0.95,
                    len(a[~np.isnan(a)]) - 1,
                    loc=np.nanmean(a),
                    scale=st.sem(a[~np.isnan(a)]),
                )
                lower_bounds.append(low if low > 0 else np.nan)
                upper_bounds.append(high)
                mean.append(np.nanmean(a))

            ax.fill_between(
                lags[srt][nan_mask],
                lower_bounds,
                upper_bounds,
                alpha=0.2,
                color=colors[i % len(colors)],
            )
            ax.plot(
                lags[srt][nan_mask],
                mean,
                color=colors[i % len(colors)],
                alpha=0.5,
                linestyle=":",
            )
            ax.scatter(
                lags[srt][nan_mask],
                mean,
                marker="o",
                color=colors[i % len(colors)],
                alpha=0.5,
            )

    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")

    ax.plot(lags, lags, color="grey", alpha=0.5)
    ax.fill_between(lags, lags, color="grey", alpha=0.5)
    ax.set_xlabel(f"lag time ({axis_units})")
    ax.set_ylabel(f"timescale ({axis_units})")
    ax.set_xlim(1, np.max(lags[srt]))

    return ax
