""" offers some helper functions for plotting """

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def rm_axes_elements(axes, elements, y_axis=True, x_axis=True):
    """removes elements from the axes

    :param axes: list of axes to remove the elements from
    :param elements: elements to remove (e.g. "ticks,ticklabels,plain,labels,grid,legend")
    :param y_axis: whether to remove the elements from the y-axis
    :param x_axis: whether to remove the elements from the x-axis
    """

    elements = elements.split(",")

    # pylint: disable=unidiomatic-typecheck
    if type(axes) != list:
        axes = [axes]
    for ax in axes:
        if ax is not None:
            if "ticks" in elements:
                if x_axis:
                    ax.set_xticks([])
                if y_axis:
                    ax.set_yticks([])
            if "ticklabels" in elements:
                if x_axis:
                    ax.set_xticklabels("")
                if y_axis:
                    ax.set_yticklabels("")
            if "plain" in elements:
                ax.axis("off")
            if "labels" in elements:
                if x_axis:
                    ax.set_xlabel("")
                if y_axis:
                    ax.set_ylabel("")
            if "grid" in elements:
                if x_axis:
                    ax.xaxis.grid(False)
                if y_axis:
                    ax.yaxis.grid(False)
            if "legend" in elements:
                ax.legend_.remove()


def plot_text(text, ax, font_scale=1.0, rotation=0, x_pos=0.5, y_pos=1.0):
    """plots text in the specified axes"""

    rm_axes_elements(ax, "plain")
    ax.text(
        x_pos,
        y_pos,
        text,
        fontsize=plt.rcParams["font.size"] * 1.5 * font_scale,
        ha="center",
        va="top",
        rotation=rotation,
    )


def plot_legend_patches(legend, ax, location="center"):
    """plots a legend in the specified axes

    the legend is showing patches with the specified colors and labels
    """

    patches = []
    for key, value in legend.items():
        patches.append(mpatches.Patch(color=value, label=key))

    ax.legend(
        handles=patches,  # loc='upper right')
        loc=location,
        fancybox=False,
        shadow=False,
        ncol=1,
    )

    rm_axes_elements(ax, "plain")


def plot_legend_scatter(ax, labels, markers, colors, **kwargs):
    """plots a legend in the specified axes

    the legend is showing scatter plot elements with the specified colors and labels
    """

    legend_elements = [
        plt.scatter([0], [0], label=l, linewidth=2, marker=m, color=c)
        for l, m, c in zip(labels, markers, colors)
    ]
    ax.legend(handles=legend_elements, title=None, fancybox=False, **kwargs)
