import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_timescales"]


def plot_timescales(
    lags,
    timescales,
    ax=None,
    xlog=False,
    ylog=True,
    n_timescales=-1,
    n_processes=-1,
    axis_units="frames",
):
    """
    Utility function for plotting implied timescales

    Examples::
        >>> from snrv.validation import implied_timescales
        >>> from snrv.plots import plot_timescales
        >>> lags = [10, 100, 1000]
        >>> timescales = implied_timescale(snrv_model, lags, training_data)
        >>> plot_timescales(lags, timescales)

    Parameters
    ----------
    lags : list or np.ndarray, n_lags
        lag times associated with each timescale

    timescales : np.ndarray, n_lags x n_timescales
        implied timescales to be plotted

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

    for i in range(n_processes):
        nan_mask = ~np.isnan(timescales[..., i][srt])
        ax.plot(lags[srt][nan_mask], timescales[..., i][srt][nan_mask])
        ax.scatter(lags[srt][nan_mask], timescales[..., i][srt][nan_mask], marker="o")

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
