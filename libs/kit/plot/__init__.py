import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def rm_axes_elements(axes, elements):
    elements = elements.split(',')
    if type(axes) != list:
        axes = [axes]
    for ax in axes:
        if ax is not None:
            if 'ticks' in elements:
                ax.set_xticks([])
                ax.set_yticks([])
            if 'plain' in elements:
                ax.axis('off')
            if 'labels' in elements:
                ax.set_ylabel('')
                ax.set_xlabel('')
            if 'grid' in elements:
                ax.yaxis.grid(False)
                ax.xaxis.grid(False)
