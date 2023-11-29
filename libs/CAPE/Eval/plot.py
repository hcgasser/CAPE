""" this module contains functions to plot evaluation results """

import re
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter
import seaborn as sns

from kit.plot import (
    rm_axes_elements,
    plot_legend_patches,
    plot_legend_scatter,
    plot_text,
)
from kit.maths import ttest
from kit.log import log_info

from CAPE.Eval.utils import pack_to_source_profile_step
from CAPE.profiles import Profile


VIS_PROFILES = [str(p) for p in Profile]

PALETTES = None
MARKERS = None
DASHES = None


def set_palettes(p):
    """set the palettes to be used in the plots"""

    global PALETTES
    PALETTES = p


def set_markers(m):
    """set the markers to be used in the plots"""

    global MARKERS
    MARKERS = m


def set_dashes(d):
    """set the dashes to be used in the plots"""

    global DASHES
    DASHES = d


def get_palette_from_source_profile_step(source, profile, step):
    """generates a palette for the selected source, profile and step

    It goes throught the pack palette and selects the entries that match the
    function parameters.

    The parameter that is 'None' determines the name within the returned palette.
    The other parameters only need to match the pack name within the original palette.
    """

    global PALETTES
    palette_packs = PALETTES["pack"]

    query = [source, profile, step]

    pal = {}
    for key, value in palette_packs.items():
        target = pack_to_source_profile_step(key)

        match, new_key = True, None
        for q, t in zip(query, target):
            if q is None:
                new_key = t
            elif re.match(q, t) is None:
                match = False

        if match:
            pal[new_key] = value

    return pal


def get_label_from_pack(pack, label_pack_level=2):
    """generates a label from the pack name

    a pack name has the form source.profile.step

    :param pack: the pack name
    :param label_pack_level: the number of levels
        to be used in the label (strings before '.')
    :return: the label
    """

    return " ".join(pack.split(".")[:label_pack_level])


def plot_tsne_kde(
    title,
    df_eval,
    dots_packs,
    area,
    font_scale=1.0,
    label_pack_level=2,
    plot_kde=True,
    xlim=None,
    ylim=None,
    rm_xlabel=False,
    rm_ylabel=False,
):
    """plots the t-SNE of the selected packs. The natural distribution is plotted as a KDE.

    :param title: str - the title of the plot
    :param df_eval: pd.DataFrame - the evaluation dataframe
    :param dots_packs: list - the packs to be plotted as dots
    :param area: GridSpec - the area to plot to (matplotlib subplotspec)
    :param font_scale: float - the font scale
    :param label_pack_level: int - the number of levels to be used in the pack label
    :param plot_kde: bool - whether to plot the natural distribution as a KDE
    :param xlim: tuple - the x-axis limits
    :param ylim: tuple - the y-axis limits
    :param rm_xlabel: bool - whether to remove the x-axis label
    :param rm_ylabel: bool - whether to remove the y-axis label
    """

    global PALETTES, MARKERS

    fig = plt.gcf()
    with sns.axes_style("whitegrid"):
        ax = fig.add_subplot(area)

        sns.set(font_scale=font_scale)
        sns.set_style("whitegrid")
        kde_color = sns.light_palette(PALETTES["source"]["natural"], as_cmap=True)(0.05)
        if plot_kde:
            sns.kdeplot(
                ax=ax,
                x="t-SNE 1",
                y="t-SNE 2",
                data=df_eval.query("source == 'natural'"),
                fill=True,
                levels=20,
                color=kde_color,
            )

        df_dots, palette_labels, markers_labels = [], {}, {}
        for dots_pack in dots_packs:
            df = df_eval.query(f'pack == "{dots_pack}"').copy()
            if len(df) > 0:
                label = get_label_from_pack(dots_pack, label_pack_level)
                df["label"] = label
                palette_labels[label] = PALETTES["pack"][dots_pack]
                markers_labels[label] = MARKERS["pack"][dots_pack]
                df_dots.append(df)

        if len(df_dots) > 0:
            df_dots = pd.concat(df_dots)
            df_dots = df_dots.sample(frac=1)
            sns.scatterplot(
                ax=ax,
                x="t-SNE 1",
                y="t-SNE 2",
                data=df_dots,
                hue="label",
                palette=palette_labels,
                style="label",
                markers=markers_labels,
                s=80,
                linewidth=2,
            )

        ax.legend(title=None)
        ax.tick_params(axis="both", which="major")
        ax.set_title(title, fontsize=16 * font_scale, fontweight="bold")
        if xlim is not None:
            ax.set_xlim(xlim)
            ax.set_xticks(np.linspace(xlim[0], xlim[1], num=13))
        if ylim is not None:
            ax.set_ylim(ylim)
            ax.set_yticks(np.linspace(ylim[0], ylim[1], num=13))

        rm_axes_elements(ax, "ticklabels")
        rm_axes_elements(ax, "labels", x_axis=rm_xlabel, y_axis=rm_ylabel)

        return ax


def plot_avg_dissimilarity_boxplots(
    df_eval,
    dissimilarity_src_packs,
    dissimilarity_tgt_packs,
    max_dissimilarity,
    area,
    font_scale=1.0,
    label_pack_level=2,
    show_ylabel=True,
):
    """plots the average dissimilarity between the selected packs

    :param df_eval: pd.DataFrame - the evaluation dataframe
    :param dissimilarity_src_packs: list - the source packs
        (the ones for which the dissimilarity is calculated)
    :param dissimilarity_tgt_packs: list - the target packs
        (the ones to whom the dissimilarity is calculated)
    :param max_dissimilarity: float - the maximum dissimilarity to be plotted (y-axis limit)
    :param area: GridSpec - the area to plot to (matplotlib subplotspec)
    :param font_scale: float - the font scale
    :param label_pack_level: int - the number of levels to be used in the pack label
    :param show_ylabel: bool - whether to show the y-axis label
    :return: the header axes
    """

    title = "Average dissimilarity"
    n_src_packs = np.sum([len(t[1]) for t in dissimilarity_src_packs])

    gs = area.subgridspec(
        len(dissimilarity_tgt_packs) + 1,
        n_src_packs,
        height_ratios=[1] + [2] * len(dissimilarity_tgt_packs),
        width_ratios=[1] * n_src_packs,
        wspace=0.0,
        hspace=0.0,
    )
    fig = plt.gcf()
    sns.set(font_scale=font_scale)
    sns.set_style("whitegrid")

    if len(dissimilarity_tgt_packs) == 1:
        title += (
            " to \n"
            + get_label_from_pack(dissimilarity_tgt_packs[0], label_pack_level)
            + " sequences"
        )

    ax_header = fig.add_subplot(gs[0, :])
    rm_axes_elements(ax_header, "plain")
    ax_header.text(
        0.5,
        1.0,
        title,
        fontsize=plt.rcParams["font.size"] * 1.5 * font_scale,
        ha="center",
        va="top",
    )
    for i, pack_tgt_name in enumerate(dissimilarity_tgt_packs):
        column = f"avg_dissimilarity_{pack_tgt_name}"
        j_start = 0
        for pack_src in dissimilarity_src_packs:  # a loop fills a column
            pack_src_source, pack_src_packs = pack_src
            j_end = j_start + len(pack_src_packs)
            ax = fig.add_subplot(gs[i + 1, j_start:j_end])

            order = None if pack_src_source in ["support", "natural"] else VIS_PROFILES
            sns.boxplot(
                df_eval.query(f"pack in {pack_src_packs}").rename(
                    columns={column: "dissimilarity"}
                ),
                y="dissimilarity",
                x="profile",
                palette=get_palette_from_source_profile_step(
                    f"^{pack_src_source}$", None, ".+"
                ),
                ax=ax,
                order=order,
            )
            ax.set_ylim(0, max_dissimilarity)

            if i != len(dissimilarity_tgt_packs) - 1:
                rm_axes_elements(ax, "labels,ticks")
            else:
                rm_axes_elements(ax, "labels")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")

            if j_start == 0 and show_ylabel:
                if len(dissimilarity_tgt_packs) == 1:
                    ylabel = "Avg. dissimilarity"
                else:
                    ylabel = get_label_from_pack(
                        pack_tgt_name, label_pack_level
                    ).replace(" ", "\n")
                    ylabel = f"to {ylabel} seqs"
                ax.set_ylabel(ylabel, rotation=90)
                ax.set_yticks([0, 500, 1000, 1500])
            else:
                ax.tick_params(labelleft=False)

            if i == 0:
                ax.set_title(pack_src_source)
            ax.yaxis.grid(False)
            j_start = j_end
    return ax_header


def plot_boxplots(
    title, df_eval, y_axis, sources, area, font_scale=1.0, add_nat_mean_line=False
):
    """plots a boxplot for a common y_axis

    The boxplots are grouped by the source. Within each source, the boxplots are
    grouped by the profile.

    :param title: str - the title of the plot
    :param df_eval: pd.DataFrame - the dataframe where the y_axis is located as a column
    :param y_axis: str - the column name of the y_axis
    :param sources: list - the sources to be plotted
    :param area: GridSpec - the area to plot to (matplotlib subplotspec)
    :param font_scale: float - the font scale
    :param add_nat_mean_line: bool - whether to add a dashed line for the natural mean
    :return: the header axes
    """

    global PALETTES

    dfs, width_ratios = {}, {}
    for source in sources:
        dfs[source] = df_eval.query(f'source == "{source}" and not {y_axis}.isnull()')
        width_ratios[source] = len(dfs[source].profile.unique())

    gs = area.subgridspec(
        2,
        len(width_ratios),
        height_ratios=[1, 6],
        width_ratios=[width_ratios[s] for s in sources],
        wspace=0.1,
        hspace=0.0,
    )
    fig = plt.gcf()
    sns.set(font_scale=font_scale)
    sns.set_style("whitegrid")

    ax_header = fig.add_subplot(gs[0, :])
    plt.subplots_adjust(
        left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2
    )
    rm_axes_elements(ax_header, "plain")
    ax_header.text(
        0.5,
        1.0,
        title,
        fontsize=plt.rcParams["font.size"] * 1.5 * font_scale,
        ha="center",
        va="top",
    )

    ax = None

    y_line_value = -1
    y_line_color = None
    for idx, source in enumerate(sources):
        ax = fig.add_subplot(gs[1, idx], sharey=ax)

        if source in ["natural", "support"]:
            y_line_value = dfs[source][y_axis].mean()
            y_line_color = PALETTES["source"][source]
        elif add_nat_mean_line and y_line_color is not None:
            ax.axhline(y=y_line_value, color=y_line_color, linestyle="--", linewidth=1)

        order = [
            p
            for p in ["natural", "support"] + VIS_PROFILES
            if p in dfs[source].profile.unique().tolist()
        ]
        sns.boxplot(
            ax=ax,
            data=dfs[source],
            x="profile",
            y=y_axis,
            palette=get_palette_from_source_profile_step(f"^{source}$", None, ".+"),
            order=order,
        )
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_title(f"{source}")

        rm_axes_elements(ax, "labels")
        if idx > 0:
            ax.tick_params(labelleft=False)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
        ax.set_xlabel(ax.get_xlabel(), labelpad=10)

    return ax_header


def plot_kmer_similarity(
    df_eval,
    packs,
    kmer_similarity_lengths,
    area,
    font_scale=1.0,
    tick_pad_x=-3,
    label_pack_level=2,
):
    """plots the k-mer length dependent k-mer similarity for the selected packs

    A line is plotted for each pack. The x-axis is the k-mer length and the y-axis
    is the k-mer similarity (the average of how many of these k-mers can be found
    in the natural sequences)

    :param df_eval: pd.DataFrame - the evaluation dataframe
    :param packs: list - the packs to be plotted
    :param kmer_similarity_lengths: list - the k-mer lengths to be plotted"""

    global DASHES

    grp_by_columns = ["pack"]
    selected_columns = grp_by_columns + [
        f"{k}mer_similarity" for k in kmer_similarity_lengths
    ]
    df_kmers_pack = (
        df_eval.query('pack in ["' + '", "'.join(packs) + '"]')[selected_columns]
        .groupby(grp_by_columns)
        .mean()
    )
    df_kmers_pack = df_kmers_pack.transpose()
    # df_kmers_pack.columns = [c.replace('.', ' ') for c in df_kmers_pack.columns]
    df_kmers_pack = df_kmers_pack.melt(
        ignore_index=False, var_name="pack", value_name="kmer similarity"
    )
    df_kmers_pack["kmer length"] = [
        i.removesuffix("mer_similarity") for i in df_kmers_pack.index
    ]

    gs = area.subgridspec(
        2, 1, width_ratios=[1], height_ratios=[1, 6], wspace=0.0, hspace=0.0
    )
    fig = plt.gcf()

    ax_header = fig.add_subplot(gs[0, :])
    rm_axes_elements(ax_header, "plain")
    ax_header.text(
        0.5, 1.0, "k-mer similarity", fontsize=16 * font_scale, ha="center", va="top"
    )

    ax = fig.add_subplot(gs[1, :])

    sns.set(font_scale=font_scale)
    sns.set_style("whitegrid")

    for pack in packs:
        sns.lineplot(
            df_kmers_pack.query(f'pack == "{pack}"'),
            x="kmer length",
            y="kmer similarity",
            label=get_label_from_pack(pack, label_pack_level),
            color=PALETTES["pack"][pack],
            linestyle=DASHES["pack"][pack],
            ax=ax,
        )

    ax.tick_params(axis="x", which="major", pad=tick_pad_x)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    plt.legend(title=None)
    return ax_header


def plot_kmer_similarity_box(
    df_eval,
    pack,
    kmer_similarity_lengths,
    quant,
    area,
    font_scale=1.0,
    tick_pad=-3,
    title_rows=2,
    rm_y_label=False,
):
    """plots the k-mer length dependent k-mer similarity for the selected pack
    restricted to the selected quantile of the natural sequences

    A boxplot is plotted for each k-mer length. The x-axis is the k-mer length and the y-axis
    is the k-mer similarity (the distribution of how many of these k-mers can be found
    in the natural sequences)

    :param df_eval: pd.DataFrame - the evaluation dataframe
    :param pack: str - the pack to be plotted
    :param kmer_similarity_lengths: list - the k-mer lengths to be plotted
    :param quant: float - the visibility quantile of sequences to be plotted
    :param area: GridSpec - the area to plot to (matplotlib subplotspec)
    :param font_scale: float - the font scale
    :param tick_pad: int - the tick padding
    :param title_rows: int - the number of rows in the title
    :param rm_y_label: bool - whether to remove the y-axis label
    :return: the header axes
    """

    global PALETTES

    source, profile, _ = pack_to_source_profile_step(pack)

    oper, quant = (
        ("<", quant) if profile == str(Profile.VIS_DOWN) else (">", 1.0 - quant)
    )
    threshold = df_eval.query('source == "natural"').visibility.quantile(quant)
    df_kmers = df_eval.query(f"pack == '{pack}' and visibility {oper} {threshold}")

    df_kmers = df_kmers[[f"{k}mer_similarity" for k in kmer_similarity_lengths]]
    df_kmers = df_kmers.transpose()
    df_kmers = df_kmers.melt(
        ignore_index=False, var_name="idx", value_name="kmer similarity"
    )
    df_kmers["kmer length"] = [
        int(i.removesuffix("mer_similarity")) for i in df_kmers.index
    ]
    df_kmers.reset_index(inplace=True, drop=True)

    gs = area.subgridspec(
        2,
        1,
        width_ratios=[1],
        height_ratios=[1 + title_rows - 2, 6],
        wspace=0.0,
        hspace=0.0,
    )
    fig = plt.gcf()

    ax_header = fig.add_subplot(gs[0, :])
    rm_axes_elements(ax_header, "plain")

    title = "below" if oper == "<" else "above"
    title_text = (
        f"{source} {profile}\n visibility {title} "
        + ("\n" if title_rows == 3 else "")
        + f"{quant * 100} percentile of natural"
    )
    ax_header.text(
        0.5, 1.0, title_text, fontsize=16 * font_scale, ha="center", va="top"
    )

    ax = fig.add_subplot(gs[1, :])

    sns.set(font_scale=font_scale)
    sns.set_style("whitegrid")
    sns.boxplot(
        df_kmers.rename(
            columns={"kmer_similarity": "kmer similarity", "length": "kmer length"}
        ),
        x="kmer length",
        y="kmer similarity",
        ax=ax,
        palette={l: PALETTES["pack"][pack] for l in list(kmer_similarity_lengths)},
    )

    ax.set_ylim((0.0, 1.0))

    ax.tick_params(axis="both", which="major", pad=tick_pad)
    if rm_y_label:
        rm_axes_elements(ax, "labels,ticklabels", x_axis=False)
    return ax_header


def plot_vs_visibility(
    df_eval,
    natural_source,
    packs,
    y_values,
    y_label,
    area,
    ax_sharex=None,
    xlabel_scatter=False,
    xlabel_boxplots=False,
    label_pack_level=2,
    ylim=1.0,
):
    """plots the selected y_values against the visibility for the selected sequences

    Each sequence is represented by a dot. The x-axis is the visibility and the y-axis
    is the y_value (e.g. the 9-mer similarity, TM score, ...)

    :param df_eval: pd.DataFrame - the evaluation dataframe
    :param natural_source: str - the pack representing the
        natural sequences (support/natural)
    :param packs: list - the packs to be plotted
    :param y_values: str - the y_values to be plotted
    :param y_label: str - the y_label
    :param area: GridSpec - the area to plot to (matplotlib subplotspec)
    :param ax_sharex: matplotlib.axes - the x-axis to be shared
    :param xlabel_scatter: bool - whether to show the x-axis label for the scatterplot
    :param xlabel_boxplots: bool - whether to show the x-axis label for the boxplots
    :param label_pack_level: int - the number of levels to be used in the pack label
    :param ylim: float - the y-axis limit
    :return: the header axes
    """

    global PALETTES

    axes = []
    width_ratios = [10, 1] + [1] * len(packs)

    gs = area.subgridspec(
        1,
        2 + len(packs),
        width_ratios=width_ratios,
        height_ratios=[1],
        wspace=0.0,
        hspace=0.2,
    )
    fig = plt.gcf()

    ax1 = fig.add_subplot(gs[0, 0], sharex=ax_sharex)
    ax1.set_ylabel(y_label)
    ax1.set_ylim(0.0, ylim)
    ax1.set_title(f"{y_label} vs. immune-visibility")

    df_packs = {}
    palette_labels, markers_labels = {}, {}
    labels = []
    for pack in packs:
        df_packs[pack] = df_eval.query(f"pack == '{pack}'").copy()
        label = get_label_from_pack(pack, label_pack_level)
        labels.append(label)
        df_packs[pack]["label"] = label
        palette_labels[label] = PALETTES["pack"][pack]
        markers_labels[label] = MARKERS["pack"][pack]

    df_scatter = pd.concat(df_packs.values()).sample(frac=1)
    sns.scatterplot(
        ax=ax1,
        data=df_scatter,
        x="visibility",
        y=y_values,
        hue="label",
        palette=palette_labels,
        style="label",
        markers=markers_labels,
        s=80,
        linewidth=1,
    )

    ax1.set_xlabel("")

    ax2 = None
    if not y_values.endswith("mer_similarity"):
        ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
        sns.boxplot(
            ax=ax2,
            data=df_eval.query(f"source == '{natural_source}'"),
            x="source",
            y=y_values,
            palette=PALETTES["source"],
        )
        ax1.legend([], frameon=False)
    else:
        plot_legend_scatter(
            ax1,
            labels,
            [markers_labels[l] for l in labels],
            [palette_labels[l] for l in labels],
        )

    axes += [ax1, ax2]
    for idx, pack in enumerate(packs):
        df_pack = df_eval.query(f"pack == '{pack}'").copy()
        df_pack["label"] = get_label_from_pack(pack, label_pack_level)
        ax = fig.add_subplot(gs[0, 2 + idx], sharey=ax1)
        axes.append(ax)
        sns.boxplot(
            ax=ax, data=df_pack, x="label", y=y_values, color=PALETTES["pack"][pack]
        )

    rm_axes_elements(axes[1:], "labels")

    if not xlabel_scatter:
        ax1.tick_params(labelbottom=False)

    for ax in axes[1:]:
        if ax is not None:
            if xlabel_boxplots:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            else:
                ax.tick_params(labelbottom=False)
            ax.tick_params(labelleft=False)

    return ax1


def plot_seq_epitopes(area, df_seq_kmers, legend=True, title=""):
    """plots a sequence with each 9-mer being represented by a colored square

    the color represents the precision of the 9-mer (how many of the 9-mers
    are present in the natural sequences). The border of the square is red
    if the 9-mer is visible (present in the natural sequences) and black
    otherwise.

    :param area: GridSpec - the area to plot to (matplotlib subplotspec)
    :param df_seq_kmers: pd.DataFrame - the dataframe containing the 9-mers
    :param legend: bool - whether to show the legend
    :param title: str - the title of the plot
    :return: the axes
    """

    fig = plt.gcf()
    ax = fig.add_subplot(area)

    n_columns = int(np.ceil(np.sqrt(len(df_seq_kmers))))

    colors = 2 * np.ones(shape=n_columns * n_columns)
    colors = colors.reshape((n_columns, n_columns))

    border_rects = []
    first_x, first_y = -0.5, -0.5
    for idx, (_, row) in enumerate(df_seq_kmers.iterrows()):
        row_idx = idx // n_columns
        column_idx = idx % n_columns

        colors[row_idx, column_idx] = row.precision if row.precision != 0 else -1
        if row.presented:
            border_rects.append(
                mpatches.Rectangle(
                    (first_x + column_idx, first_y + row_idx),
                    1,
                    1,
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none",
                )
            )

    cmap = plt.cm.get_cmap("viridis")  # RdYlBu')
    cmap.set_over("white")
    cmap.set_under("black")
    ax.imshow(colors, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.axis("off")

    # Create legend patches
    patch_100 = mpatches.Patch(color=cmap(1.0), label=r"100 \%")
    patch_75 = mpatches.Patch(color=cmap(0.75), label=r"75 \%")
    patch_50 = mpatches.Patch(color=cmap(0.5), label=r"50 \%")
    patch_0 = mpatches.Patch(color=cmap(0.0), label=r"1 \%")
    patch_not = mpatches.Patch(color="black", label="anatural")
    patch_vis = mpatches.Patch(color="red", label="visible")

    # Create legend
    # ax.legend(handles=[patch_100, patch_75, patch_50, patch_0,
    #           patch_not, patch_vis], # loc='upper right')
    #          loc='upper center', bbox_to_anchor=(0.5, 0.1),
    #                     fancybox=False, shadow=False, ncol=2)
    if legend:
        ax.legend(
            handles=[
                patch_100,
                patch_75,
                patch_50,
                patch_0,
                patch_not,
                patch_vis,
            ],  # loc='upper right')
            loc="upper left",
            bbox_to_anchor=(-0.4, 1.0),
            fancybox=False,
            shadow=False,
            ncol=1,
        )
    ax.set_title(f"{title}9-mer precision and visibility")

    # Add the border rectangles to the plot
    for rect in border_rects:
        ax.add_patch(rect)

    return ax


def precision_format(x):
    """helper function to format the precision"""

    return str(x).rstrip("0").rstrip(".") if x != 0 else "0"


def plot_seq_precision_pie(area, df_seq_kmers):
    """plots a pie chart of the precision of the 9-mers of the selected sequence"""

    fig = plt.gcf()
    ax = fig.add_subplot(area)
    bins = [0, 0.001, 0.01, 0.1, 0.5, 1.0]
    # Compute histogram with custom bin edges
    hist, _ = np.histogram(df_seq_kmers[df_seq_kmers.presented].precision, bins=bins)

    # Create pie chart with custom bin edges
    labels = [
        rf"{precision_format(100*von)}\%-{precision_format(100*bis)}\%"
        for von, bis in zip(bins[:-1], bins[1:])
    ]

    def absolute_value(val):
        a = np.round(val / 100.0 * hist.sum(), 0)
        return precision_format(a)

    ax.pie(
        hist, labels=labels, startangle=90, counterclock=False, autopct=absolute_value
    )
    ax.set_title("visible 9-mer precision")

    return ax


def plot_seqs_precision_bar(
    area,
    d_df_seq_kmers,
    kind="barh",
    font_scale=1.0,
    bins=(0, 0.001, 0.01, 0.1, 0.5, 1.0),
    round_digits=10,
):
    """plots a bar chart of the precision of the 9-mers of the selected sequences"""

    fig = plt.gcf()
    ax = fig.add_subplot(area)
    c_seq = len(d_df_seq_kmers)

    precision = [
        (
            rf"{precision_format(100*von)}-{precision_format(100*bis)}\%"
            if von != bis
            else rf"{precision_format(100*von)}\%"
        )
        for von, bis in zip(
            np.round(bins[:-1], round_digits), np.round(bins[1:], round_digits)
        )
    ]

    data = {"Precision": precision}
    totals = []
    for source, df_seq_kmers in d_df_seq_kmers.items():
        # Compute histogram with custom bin edges
        hist, _ = np.histogram(
            df_seq_kmers[df_seq_kmers.presented].precision, bins=bins
        )
        data.update({f"{source} abs": hist})
        total = hist.sum()
        data.update({f"{source}": hist / total})
        totals.append(total)

    df = pd.DataFrame(data).set_index("Precision")
    df_bar = df[[f"{source}" for source in d_df_seq_kmers.keys()]].transpose()
    df_bar.plot(ax=ax, kind=kind, stacked=True, width=1.0)

    for c in ax.containers:
        precision_category = c.get_label()
        labels = []
        for idx, _ in enumerate(c):
            if kind == "barh":
                source = ax.get_yticklabels()[idx].get_text()
            elif kind == "bar":
                source = ax.get_xticklabels()[idx].get_text()
            _abs = df.at[precision_category, f"{source} abs"]
            pc = df.at[precision_category, f"{source}"]
            if kind == "barh":
                labels.append(rf"{_abs}\n{pc*100:.1f}\%")
            elif kind == "bar":
                labels.append(rf"{_abs} ({pc*100:.1f}\%)")

        ax.bar_label(c, label_type="center", labels=labels, size=16 * font_scale)

    if kind == "barh":
        ax.set_xlabel("proportion of epitiopes in precision categories")
        legend = ax.legend(
            loc="upper center",
            fancybox=False,
            shadow=False,
            ncol=len(bins),
            bbox_to_anchor=(0.5, 1.15),
        )
        legend.set_alpha(0.1)
        ax.set_xlim(left=0.0, right=1.0)
        ax.set_ylim(bottom=-0.5, top=-0.5 + c_seq)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    elif kind == "bar":
        ax.set_ylabel(
            "proportion of epitiopes \n in precision categories",
            rotation=-90,
            labelpad=20,
        )
        ax.yaxis.set_label_position("right")
        legend = ax.legend(
            loc="upper center",
            fancybox=False,
            shadow=False,
            ncol=2,
            bbox_to_anchor=(0.5, 1.3),
        )
        legend.set_alpha(0.1)
        ax.set_ylim(bottom=0.0, top=1.0)
        ax.set_xlim(left=-0.5, right=-0.5 + c_seq)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.yaxis.tick_right()
        # plt.xticks(rotation=90)
    return ax


def plot_epitope_recall_by_seq(area, df_eval, seq_hash, peptide_lengths_col):
    """plots the recall of the selected sequence

    The x-axis represents the recall value and the y-axis the number of
    natural sequences for which these many epitopes are recalled.
    """

    fig = plt.gcf()
    ax = fig.add_subplot(area)

    sns.histplot(df_eval[f"recall_{peptide_lengths_col}_{seq_hash}"], ax=ax)
    ax.set_xlabel(f"visible {peptide_lengths_col}-mer recall")
    ax.set_ylabel(r"\# natural sequences")
    ax.set_title(
        f"visible {peptide_lengths_col}-mer recall distribution over natural sequences"
    )
    ax.set_xlim(left=0.0)

    return ax


def plot_natural_vs_pack(
    df_eval,
    pack,
    x_function_name,
    x_metric,
    y_function_name,
    y_metric,
    peptide_lengths_col,
    area,
    vis_up_visibility_threshold=-1,
    xlim=None,
    ylim=None,
    position="right",
):
    """plots the selected x_metric and y_metric for the selected pack as well as
    for the natural sequences"""

    global PALETTES, MARKERS

    df_natural = df_eval.query(
        f"source == 'natural' and visibility >= {vis_up_visibility_threshold}"
    ).copy()
    df_natural["label"] = "natural"
    df_pack = df_eval.query(
        f"pack == '{pack}' and visibility >= {vis_up_visibility_threshold}"
    ).copy()
    label_pack = get_label_from_pack(pack, 2)
    df_pack["label"] = label_pack
    log_info(f"natural: {len(df_natural)}, {label_pack}: {len(df_pack)}")

    palette_labels, markers_labels = {}, {}
    palette_labels["natural"] = PALETTES["source"]["natural"]
    palette_labels[label_pack] = PALETTES["pack"][pack]
    markers_labels["natural"] = MARKERS["source"]["natural"]
    markers_labels[label_pack] = MARKERS["pack"][pack]

    x_column = f"{x_function_name}_{x_metric}_{peptide_lengths_col}"
    y_column = f"{y_function_name}_{y_metric}_{peptide_lengths_col}"
    x_label = f"{x_function_name}. {x_metric}"
    y_label = f"{y_function_name}. {y_metric}"

    df_tmp = pd.concat([df_natural, df_pack])
    df_tmp = df_tmp.rename(columns={x_column: x_label, y_column: y_label})

    gs = area.subgridspec(
        3, 2, height_ratios=[3, 20, 6], width_ratios=[10, 3], wspace=0.1, hspace=0.2
    )
    fig = plt.gcf()

    ax0 = fig.add_subplot(gs[0, :])

    text = (
        f"{y_function_name} {y_metric} and {y_function_name} {x_metric} \n"
        f" of natural and {label_pack} sequences"
    )
    if vis_up_visibility_threshold > 0:
        text += f"\n with visibility greater than {vis_up_visibility_threshold}"
    plot_text(text, ax0, y_pos=1.0, font_scale=0.6)

    ax1 = fig.add_subplot(gs[1, 0])
    sns.scatterplot(
        ax=ax1,
        data=df_tmp,
        x=x_label,
        y=y_label,
        hue="label",
        palette=palette_labels,
        style="label",
        markers=markers_labels,
        legend=False,
        linewidth=1,
        s=40,
    )

    rm_axes_elements(ax1, "labels", y_axis=False)
    ax1.tick_params(labelbottom=False)
    if position == "right":
        ax1.tick_params(labelleft=False)
        rm_axes_elements(ax1, "labels")
    if xlim is not None:
        ax1.set_xlim((0.0, xlim))
    if ylim is not None:
        ax1.set_ylim((0, ylim))

    # Axis 2
    ax2 = fig.add_subplot(gs[1, 1], sharey=ax1)
    sns.boxplot(
        ax=ax2,
        data=df_tmp,
        x="source",
        y=y_label,
        hue="label",
        palette=palette_labels,
        dodge=False,
    )
    p_value = ttest(
        df_natural[y_column].astype("float"), df_pack[y_column].astype("float")
    )
    ax2.text(
        0.5,
        -0.01,
        f"p: {p_value:.3f}",
        ha="center",
        va="top",
        weight="bold",
        color="darkblue",
    )
    ax2.set_xlabel("")
    if position == "right":
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
    else:
        ax2.tick_params(labelleft=False)
        rm_axes_elements(ax2, "labels", x_axis=False)
    rm_axes_elements(ax2, "legend,ticklabels", y_axis=False)

    # Axis 3
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    sns.boxplot(
        ax=ax3,
        data=df_tmp,
        y="source",
        x=x_label,
        hue="label",
        palette=palette_labels,
        dodge=False,
    )
    ax3.set_ylabel("")
    rm_axes_elements(ax3, "legend,ticklabels", x_axis=False)
    # ax3.tick_params(axis='x', pad=-4)

    # Axis 4
    if position == "right":
        ax4 = fig.add_subplot(gs[2, 1])
        plot_legend_patches(
            {
                "natural": PALETTES["source"]["natural"],
                label_pack: palette_labels[label_pack],
            },
            ax4,
            location="lower center",
        )

    return ax0
