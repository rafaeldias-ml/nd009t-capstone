import seaborn as sns

from klib import missingval_plot, data_cleaning, corr_plot


from typing import Any, Dict, Optional, Tuple, Union, List
import scipy
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# This module is based on patched versions of klib functions (and hence following their code style)
#
# klib is a Python library for importing, cleaning, analyzing and preprocessing data.
# Check it out at https://github.com/akanz1/klib


## Patched version of klib.dist_plot: https://github.com/akanz1/klib/blob/bf635c5067745820c1520f5919e809d751bcee24/klib/describe.py#L420
# Distribution plot
def dist_plot(
    data: pd.DataFrame,
    mean_color: str = "orange",
    figsize: Tuple = (16, 2),
    fill_range: Tuple = (0.025, 0.975),
    showall: bool = False,
    kde_kws: Dict[str, Any] = None,
    rug_kws: Dict[str, Any] = None,
    fill_kws: Dict[str, Any] = None,
    font_kws: Dict[str, Any] = None,
):
    """ Two-dimensional visualization of the distribution of non binary numerical features.
    Parameters
    ----------
    data : pd.DataFrame
        2D dataset that can be coerced into Pandas DataFrame. If a Pandas DataFrame \
        is provided, the index/column information is used to label the plots
    mean_color : str, optional
        Color of the vertical line indicating the mean of the data, by default "orange"
    figsize : Tuple, optional
        Controls the figure size, by default (16, 2)
    fill_range : Tuple, optional
        Set the quantiles for shading. Default spans 95% of the data, which is about \
        two std. deviations above and below the mean, by default (0.025, 0.975)
    showall : bool, optional
        Set to True to remove the output limit of 20 plots, by default False
    kde_kws : Dict[str, Any], optional
        Keyword arguments for kdeplot(), by default {"color": "k", "alpha": 0.7, \
        "linewidth": 1.5, "bw": 0.3}
    rug_kws : Dict[str, Any], optional
        Keyword arguments for rugplot(), by default {"color": "#ff3333", \
        "alpha": 0.05, "linewidth": 4, "height": 0.075}
    fill_kws : Dict[str, Any], optional
        Keyword arguments to control the fill, by default {"color": "#80d4ff", \
        "alpha": 0.2}
    font_kws : Dict[str, Any], optional
        Keyword arguments to control the font, by default {"color":  "#111111", \
        "weight": "normal", "size": 11}
    Returns
    -------
    ax: matplotlib Axes
        Returns the Axes object with the plot for further tweaking.
    """

    # Handle dictionary defaults
    kde_kws = (
        {"alpha": 0.75, "linewidth": 1.5, "bw": 0.4}
        if kde_kws is None
        else kde_kws.copy()
    )
    rug_kws = (
        {"color": "#ff3333", "alpha": 0.05, "linewidth": 4, "height": 0.075}
        if rug_kws is None
        else rug_kws.copy()
    )
    fill_kws = (
        {"color": "#80d4ff", "alpha": 0.2} if fill_kws is None else fill_kws.copy()
    )
    font_kws = (
        {"color": "#111111", "weight": "normal", "size": 11}
        if font_kws is None
        else font_kws.copy()
    )

    data = pd.DataFrame(data.copy()).dropna(axis=1, how="all")
    data = data.loc[:, data.nunique() > 2]
    cols = list(data.select_dtypes(include=["number"]).columns)
    data = data[cols]
    data = data.loc[:, data.nunique() > 2]

    if len(cols) == 0:
        print("No columns with numeric data were detected.")
        return

    elif len(cols) >= 20 and showall is False:
        print(
            "Note: The number of non binary numerical features is very large "
            f"({len(cols)}), please consider splitting the data. Showing plots for "
            "the first 20 numerical features. Override this by setting showall=True."
        )
        cols = cols[:20]

    for col in cols:
        num_dropped_vals = data[col].isna().sum()
        if num_dropped_vals > 0:
            col_data = data[col].dropna(axis=0)
            print(f"Dropped {num_dropped_vals} missing values from column {col}.")

        else:
            col_data = data[col]

        _, ax = plt.subplots(figsize=figsize)
        ax = sns.distplot(
            col_data, hist=False, rug=True, kde_kws=kde_kws, rug_kws=rug_kws,
        )

        # Vertical lines and fill
        x, y = ax.lines[0].get_xydata().T
        ax.fill_between(
            x,
            y,
            where=(
                (x >= np.quantile(col_data, fill_range[0]))
                & (x <= np.quantile(col_data, fill_range[1]))
            ),
            label=f"{fill_range[0]*100:.1f}% - {fill_range[1]*100:.1f}%",
            **fill_kws,
        )

        mean = np.mean(col_data)
        std = scipy.stats.tstd(col_data)
        ax.vlines(
            x=mean,
            ymin=0,
            ymax=np.interp(mean, x, y),
            ls="dotted",
            color=mean_color,
            lw=2,
            label="mean",
        )
        ax.vlines(
            x=np.median(col_data),
            ymin=0,
            ymax=np.interp(np.median(col_data), x, y),
            ls=":",
            color=".3",
            label="median",
        )
        ax.vlines(
            x=[mean - std, mean + std],
            ymin=0,
            ymax=[np.interp(mean - std, x, y), np.interp(mean + std, x, y)],
            ls=":",
            color=".5",
            label="\u03BC \u00B1 \u03C3",
        )

        ax.set_ylim(0)
        ax.set_xlim(ax.get_xlim()[0] * 1.15, ax.get_xlim()[1] * 1.15)

        # Annotations and legend
        ax.text(
            0.01, 0.85, f"Mean: {mean:.2f}", fontdict=font_kws, transform=ax.transAxes
        )
        ax.text(
            0.01, 0.7, f"Std. dev: {std:.2f}", fontdict=font_kws, transform=ax.transAxes
        )
        ax.text(
            0.01,
            0.55,
            f"Skew: {scipy.stats.skew(col_data):.2f}",
            fontdict=font_kws,
            transform=ax.transAxes,
        )
        ax.text(
            0.01,
            0.4,
            f"Kurtosis: {scipy.stats.kurtosis(col_data):.2f}",  # Excess Kurtosis
            fontdict=font_kws,
            transform=ax.transAxes,
        )
        ax.text(
            0.01,
            0.25,
            f"Count: {len(col_data)}",
            fontdict=font_kws,
            transform=ax.transAxes,
        )
        ax.legend(loc="upper right")

    return ax


# This is a variant of klib.corr_plot stripped of some checks and adapted for ploting of cramer correlation matrix 
# https://github.com/akanz1/klib/blob/bf635c5067745820c1520f5919e809d751bcee24/klib/describe.py#L261
# Correlation matrix / heatmap
def crammer_corr_plot(
    corr: np.array,
    labels: List,
    threshold: float = 0.6,
    cmap: str = "BrBG",
    figsize: Tuple = (12, 10),
    annot: bool = True,
    method = 'cramers',
    **kwargs,
):
    """ Two-dimensional visualization of the correlation between feature-columns \
        excluding NA values.
    Parameters
    ----------
    data : pd.DataFrame
        2D dataset that can be coerced into Pandas DataFrame. If a Pandas DataFrame \
        is provided, the index/column information is used to label the plots
    split : Optional[str], optional
        Type of split to be performed {None, "pos", "neg", "high", "low"}, by default \
        None
            * None: visualize all correlations between the feature-columns
            * pos: visualize all positive correlations between the feature-columns \
                above the threshold
            * neg: visualize all negative correlations between the feature-columns \
                below the threshold
            * high: visualize all correlations between the feature-columns for \
                which abs (corr) > threshold is True
            * low: visualize all correlations between the feature-columns for which \
                abs(corr) < threshold is True
    threshold : float, optional
        Value between 0 and 1 to set the correlation threshold, by default 0 unless \
            split = "high" or split = "low", in which case default is 0.3
    target : Optional[Union[pd.Series, str]], optional
        Specify target for correlation. E.g. label column to generate only the \
        correlations between each feature and the label, by default None
    method : str, optional
        method: {"pearson", "spearman", "kendall"}, by default "pearson"
            * pearson: measures linear relationships and requires normally \
                distributed and homoscedastic data.
            * spearman: ranked/ordinal correlation, measures monotonic relationships.
            * kendall: ranked/ordinal correlation, measures monotonic relationships. \
                Computationally more expensive but more robust in smaller dataets \
                than "spearman".
    cmap : str, optional
        The mapping from data values to color space, matplotlib colormap name or \
        object, or list of colors, by default "BrBG"
    figsize : Tuple, optional
        Use to control the figure size, by default (12, 10)
    annot : bool, optional
        Use to show or hide annotations, by default True
    dev : bool, optional
        Display figure settings in the plot by setting dev = True. If False, the \
        settings are not displayed, by default False
    Keyword Arguments : optional
        Additional elements to control the visualization of the plot, e.g.:
            * mask: bool, default True
                If set to False the entire correlation matrix, including the upper \
                triangle is shown. Set dev = False in this case to avoid overlap.
            * vmax: float, default is calculated from the given correlation \
                coefficients.
                Value between -1 or vmin <= vmax <= 1, limits the range of the cbar.
            * vmin: float, default is calculated from the given correlation \
                coefficients.
                Value between -1 <= vmin <= 1 or vmax, limits the range of the cbar.
            * linewidths: float, default 0.5
                Controls the line-width inbetween the squares.
            * annot_kws: dict, default {"size" : 10}
                Controls the font size of the annotations. Only available when \
                annot = True.
            * cbar_kws: dict, default {"shrink": .95, "aspect": 30}
                Controls the size of the colorbar.
            * Many more kwargs are available, i.e. "alpha" to control blending, or \
                options to adjust labels, ticks ...
        Kwargs can be supplied through a dictionary of key-value pairs (see above).
    Returns
    -------
    ax: matplotlib Axes
        Returns the Axes object with the plot for further tweaking.
    """

    labels = np.array(labels)
    idxs = (corr >= threshold)
    idxs = np.apply_along_axis(np.any, 0, arr=idxs)
    
    labels = labels[idxs]
    corr = corr[idxs, :]
   
    
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    vmax = np.round(np.max(corr)+0.05, 2)
    vmin = np.round(np.min(corr)-0.05, 2)

    fig, ax = plt.subplots(figsize=figsize)

    # Specify kwargs for the heatmap
    kwargs = {
        "mask": mask,
        "cmap": cmap,
        "annot": annot,
        "vmax": vmax,
        "vmin": vmin,
        "linewidths": 0.5,
        "annot_kws": {"size": 8},
        "cbar_kws": {"shrink": 0.95, "aspect": 30},
        **kwargs,
    }

    # Draw heatmap with mask and default settings
    sns.heatmap(corr, center=0, fmt=".1f", **kwargs)

    ax.set_title(f"Feature-correlation ({method})", fontdict={"fontsize": 18})
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels, rotation=0)

    return ax