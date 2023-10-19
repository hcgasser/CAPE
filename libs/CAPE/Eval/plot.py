import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re 
import numpy as np
import pandas as pd

from kit.plot import rm_axes_elements
import seaborn as sns

from .utils import pack_to_source_task_step

palettes = None
markers = None

def set_palettes(p):
    global palettes
    palettes = p


def set_markers(m):
    global markers
    markers = m


def get_palette_from_source_task_step(source, task, step):    
    global palettes
    palette_packs = palettes['pack']

    query = [source, task, step]
    
    pal = {}
    for key, value in palette_packs.items():
        target = pack_to_source_task_step(key)

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
    return " ".join(pack.split('.')[:label_pack_level])


def plot_TSNE_kde(title, df_eval, kde_pack, dots_packs, area, font_scale=1., label_pack_level=2):
    global palettes

    fig = plt.gcf()
    with sns.axes_style("whitegrid"):
        ax = fig.add_subplot(area)

        sns.set(font_scale=font_scale)
        sns.set_style("whitegrid")
        sns.kdeplot(ax=ax, x="t-SNE 1", y="t-SNE 2", 
                    data=df_eval.query(f'pack.str.startswith("{kde_pack}")', engine='python'), 
                    fill=True, levels=20)

        df_dots, palette_labels, markers_labels = [], {}, {}
        for dots_pack in dots_packs:
            df = df_eval.query(f'pack == "{dots_pack}"').copy()
            if len(df) > 0:
                label = get_label_from_pack(dots_pack, label_pack_level)
                df['label'] = label
                palette_labels[label] = palettes['pack'][dots_pack]
                markers_labels[label] = markers['pack'][dots_pack]
                df_dots.append(df)

        if len(df_dots) > 0:
            df_dots = pd.concat(df_dots)
            df_dots = df_dots.sample(frac=1)
            sns.scatterplot(ax=ax, x="t-SNE 1", y="t-SNE 2", 
                            data=df_dots, 
                            hue='label', style='label',
                            palette=palette_labels, markers=markers_labels)

        ax.legend(title=None)
        ax.tick_params(axis='both', which='major')
        ax.set_title(title, fontsize=16*font_scale, fontweight="bold")
        return ax


def plot_avg_dissimilarity_boxplots(title, df_eval, dissimilarity_src_packs, dissimilarity_tgt_packs, max_dissimilarity, area, font_scale=1., label_pack_level=2):
    n_src_packs = np.sum([len(t[1]) for t in dissimilarity_src_packs])

    gs = area.subgridspec(len(dissimilarity_tgt_packs) + 1, n_src_packs,
                          height_ratios=[1] + [2] * len(dissimilarity_tgt_packs), width_ratios=[1] * n_src_packs, 
                          wspace=0.0, hspace=0.0)
    fig = plt.gcf()
    ax_header = fig.add_subplot(gs[0,:])
    rm_axes_elements(ax_header, 'plain')
    ax_header.text(0.5, 1.0, title, fontsize=plt.rcParams['font.size']*1.5*font_scale, 
                   ha='center', va='top')  
    for i, pack_tgt_name in enumerate(dissimilarity_tgt_packs): 
        column = f'avg_dissimilarity_{pack_tgt_name}'
        j_start = 0
        for pack_src in dissimilarity_src_packs: # a loop fills a column
            pack_src_source, pack_src_packs = pack_src
            j_end = j_start + len(pack_src_packs)
            ax = fig.add_subplot(gs[i+1,j_start:j_end])

            order = None if pack_src_source in ['support', 'natural'] else ['baseline', 'deimmunize', 'immunize']
            sns.boxplot(
                df_eval.query(f'pack in {pack_src_packs}').rename(columns={column: 'dissimilarity'}), 
                y='dissimilarity', 
                x='task',
                palette=get_palette_from_source_task_step(f'^{pack_src_source}$', None, '.+'),
                ax=ax, order=order)
            ax.set_ylim(0, max_dissimilarity)

            if i != len(dissimilarity_tgt_packs) - 1:
                rm_axes_elements(ax, 'labels,ticks')
            else:
                rm_axes_elements(ax, 'labels')
            
            if j_start == 0: 
                label = get_label_from_pack(pack_tgt_name, label_pack_level).replace(' ', '\n')
                ax.set_ylabel(f"to {label} seqs", rotation=90)
                ax.set_yticks([0, 500, 1000])
            else:
                ax.tick_params(labelleft=False)

            if i == 0:
                ax.set_title(pack_src_source)
            ax.yaxis.grid(False)
            j_start = j_end
    return ax_header


def plot_boxplots(title, df_eval, y_axis, sources, area, font_scale=1.):
    global palettes

    n_sources = len(sources)
    axes = []
    width_ratios = []
    for source in sources:
        width_ratios += [len(df_eval.query(f'source == "{source}"').task.unique())]

    gs = area.subgridspec(2, len(width_ratios), width_ratios=width_ratios, height_ratios=[1,6], 
                          wspace=0.1, hspace=0.0)
    fig = plt.gcf()
    sns.set(font_scale=font_scale)
    sns.set_style("whitegrid")
    
    ax_header = fig.add_subplot(gs[0,:])
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)
    rm_axes_elements(ax_header, 'plain')
    ax_header.text(0.5, 1.0, title, fontsize=plt.rcParams['font.size']*1.5*font_scale, 
                   ha='center', va='top')  
    
    ax = None
    for idx, source in enumerate(sources):
        ax = fig.add_subplot(gs[1,idx], sharey=ax)
        order = None if source in ['support', 'natural'] else ['baseline', 'deimmunize', 'immunize']
        sns.boxplot(ax=ax, data=df_eval.query(f'source == "{source}"'),
                    x="task", y=y_axis, palette=get_palette_from_source_task_step(f'^{source}$', None, '.+'), order=order)
        ax.set_title(f"{source}")
        rm_axes_elements(ax, 'labels')
        if idx > 0:
            ax.tick_params(labelleft=False)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
        ax.set_xlabel(ax.get_xlabel(), labelpad=10)

    return ax_header


def plot_kmer_similarity(df_eval, packs, kmer_similarity_lengths, area, font_scale=1., tick_pad=-3, label_pack_level=2):
    grp_by_columns = ['pack']
    selected_columns = grp_by_columns + [f"{k}mer_similarity" for k in kmer_similarity_lengths]
    df_kmers_pack = df_eval.query('pack in ["' + '", "'.join(packs) + '"]')[selected_columns].groupby(grp_by_columns).mean()
    df_kmers_pack = df_kmers_pack.transpose()
    # df_kmers_pack.columns = [c.replace('.', ' ') for c in df_kmers_pack.columns]
    df_kmers_pack = df_kmers_pack.melt(ignore_index=False, var_name='pack', value_name='kmer similarity')
    df_kmers_pack['kmer length'] = [i.removesuffix('mer_similarity') for i in df_kmers_pack.index]
    
    gs = area.subgridspec(2, 1, 
                          width_ratios=[1], height_ratios=[1,6], 
                          wspace=0.0, hspace=0.0)
    fig = plt.gcf()
    
    ax_header = fig.add_subplot(gs[0,:])
    rm_axes_elements(ax_header, 'plain')
    ax_header.text(0.5, 1.0, 'k-mer similarity', 
                   fontsize=16*font_scale, 
                   ha='center', va='top')
    
    ax = fig.add_subplot(gs[1,:])
   
    sns.set(font_scale=font_scale)
    sns.set_style("whitegrid")

    for pack in packs:
        sns.lineplot(df_kmers_pack.query(f'pack == "{pack}"'),
            x='kmer length', y='kmer similarity', 
            label=get_label_from_pack(pack, label_pack_level), 
            color=palettes['pack'][pack],
            ax=ax)

    ax.tick_params(axis='both', which='major', pad=tick_pad)
    plt.legend(title=None)
    return ax_header


def plot_kmer_similarity_box(df_eval, pack, kmer_similarity_lengths, quant, area, font_scale=1., tick_pad=-3):
    global palettes

    source, task, step = pack_to_source_task_step(pack)
    
    oper, quant = ("<", quant) if task == "deimmunize" else (">", 1.-quant)
    threshold = df_eval.query('source == "natural"').visibility.quantile(quant)
    df_kmers = df_eval.query(f"pack == '{pack}' and visibility {oper} {threshold}")
    
    df_kmers = df_kmers[[f"{k}mer_similarity" for k in kmer_similarity_lengths]]
    df_kmers = df_kmers.transpose()
    df_kmers = df_kmers.melt(ignore_index=False, var_name='idx', value_name='kmer similarity')
    df_kmers['kmer length'] = [int(i.removesuffix('mer_similarity')) for i in df_kmers.index]
    df_kmers.reset_index(inplace=True, drop=True)
    
    gs = area.subgridspec(2, 1, 
                          width_ratios=[1], height_ratios=[1,6], 
                          wspace=0.0, hspace=0.0)
    fig = plt.gcf()
    
    ax_header = fig.add_subplot(gs[0,:])
    rm_axes_elements(ax_header, 'plain')

    title = "below" if oper == "<" else "above"
    ax_header.text(0.5, 1.0, 
                   f"{source} {task}\n visibility {title} {quant * 100} percentile of natural", 
                   fontsize=16*font_scale, 
                   ha='center', va='top')
    
    ax = fig.add_subplot(gs[1,:])
    
    
    sns.set(font_scale=font_scale)
    sns.set_style("whitegrid")    
    sns.boxplot(df_kmers\
                .rename(columns={'kmer_similarity': 'kmer similarity', 'length': 'kmer length'}),
               x="kmer length", y="kmer similarity", ax=ax, 
               palette={l: palettes['pack'][pack] for l in list(kmer_similarity_lengths)})
    
    ax.tick_params(axis='both', which='major', pad=tick_pad)
    return ax_header


def plot_vs_visibility(df_eval, natural_source, packs, y_values, y_label, area, ax_sharex=None, xlabel_scatter=False, xlabel_boxplots=False, label_pack_level=2, ylim=1.):
    global palettes 


    axes = []
    width_ratios = [10, 1] + [1]*len(packs)

    gs = area.subgridspec(1, 2 + len(packs), 
                          width_ratios=width_ratios, height_ratios=[1], 
                          wspace=0., hspace=0.2)
    fig = plt.gcf()

    ax1 = fig.add_subplot(gs[0,0], sharex=ax_sharex)
    ax1.set_ylabel(y_label)
    ax1.set_ylim(0., ylim)
    ax1.set_title(f"{y_label} vs. immune visibility")

    df_packs = {}
    palette_labels = {}
    for pack in packs:
        df_packs[pack] = df_eval.query(f"pack == '{pack}'").copy()
        label = get_label_from_pack(pack, label_pack_level)
        df_packs[pack]['label'] = label
        palette_labels[label] = palettes['pack'][pack]

    df_scatter = pd.concat(df_packs.values()).sample(frac=1)
    sns.scatterplot(ax=ax1,
        data=df_scatter,
        x='visibility', y=y_values,
        hue='label', palette=palette_labels)
    
    ax1.set_xlabel('')

    ax2 = None
    if not y_values.endswith('mer_similarity'):
        ax2 = fig.add_subplot(gs[0,1], sharey=ax1)
        sns.boxplot(ax=ax2, data=df_eval.query(f"source == '{natural_source}'"), x='source', y=y_values, palette=palettes['source'])
        ax1.legend([], frameon=False)
    else:
        ax1.legend(title=None)

    axes += [ax1, ax2]
    for idx, pack in enumerate(packs):
        df_pack = df_eval.query(f"pack == '{pack}'").copy()
        df_pack['label'] = get_label_from_pack(pack, label_pack_level)
        ax = fig.add_subplot(gs[0,2+idx], sharey=ax1)
        axes.append(ax)
        sns.boxplot(ax=ax, data=df_pack, x='label', y=y_values, color=palettes['pack'][pack])

    rm_axes_elements(axes[1:], 'labels')

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


def plot_seq_epitopes(area, df_seq_kmers, legend=True, title=''):
    fig = plt.gcf()
    ax = fig.add_subplot(area)
    
    
    n_columns = int(np.ceil(np.sqrt(len(df_seq_kmers))))
    
    colors = 2*np.ones(shape=(n_columns*n_columns))
    colors = colors.reshape((n_columns, n_columns))
    
    border_rects = []
    first_x, first_y = -0.5, -0.5
    for idx, (kmer, row) in enumerate(df_seq_kmers.iterrows()):
        row_idx = idx // n_columns
        column_idx = idx % n_columns
    
        colors[row_idx, column_idx] = row.precision if row.precision != 0 else -1
        if row.presented:
            border_rects.append(
                mpatches.Rectangle((first_x + column_idx, first_y + row_idx), 1, 1, linewidth=2, edgecolor='red', facecolor='none')
            )
    
    cmap = plt.cm.get_cmap('viridis') #RdYlBu')
    cmap.set_over('white')
    cmap.set_under('black')
    ax.imshow(colors, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    ax.axis('off')

    # Create legend patches
    patch_100 = mpatches.Patch(color=cmap(1.0), label='100 \%')
    patch_75 = mpatches.Patch(color=cmap(0.75), label='75 \%')
    patch_50 = mpatches.Patch(color=cmap(0.5), label='50 \%')
    patch_0 = mpatches.Patch(color=cmap(0.0), label='1 \%')
    patch_not = mpatches.Patch(color='black', label='anatural')
    patch_vis = mpatches.Patch(color='red', label='visible')
    
    # Create legend
    # ax.legend(handles=[patch_100, patch_75, patch_50, patch_0, patch_not, patch_vis], # loc='upper right')
    #          loc='upper center', bbox_to_anchor=(0.5, 0.1),
    #                     fancybox=False, shadow=False, ncol=2)
    if legend:
        ax.legend(handles=[patch_100, patch_75, patch_50, patch_0, patch_not, patch_vis], # loc='upper right')
                loc='upper left', bbox_to_anchor=(-0.4, 1.),
                            fancybox=False, shadow=False, ncol=1)
    ax.set_title(f'{title}9-mer precision and visibility')
    
    # Add the border rectangles to the plot
    for rect in border_rects:
        ax.add_patch(rect)

    return ax


def plot_seq_precision_pie(area, df_seq_kmers):
    fig = plt.gcf()
    ax = fig.add_subplot(area)
    bins = [0, 0.001, 0.01, 0.1, 0.5, 1.]
    # Compute histogram with custom bin edges
    hist, _ = np.histogram(df_seq_kmers[df_seq_kmers.presented].precision, bins=bins)


    my_format = lambda x: str(x).rstrip("0").rstrip(".") if x != 0 else '0'

    # Create pie chart with custom bin edges
    labels = [f'{my_format(100*von)}\%-{my_format(100*bis)}\%' for von, bis in zip(bins[:-1], bins[1:])]


    def absolute_value(val):
        a  = np.round(val/100.*hist.sum(), 0)
        return my_format(a)
    
    ax.pie(hist, labels=labels, startangle=90, counterclock=False, autopct=absolute_value)
    ax.set_title('visible 9-mer precision')

    return ax


def plot_seqs_precision_bar(area, d_df_seq_kmers, kind='barh', font_scale=1.):
    fig = plt.gcf()
    ax = fig.add_subplot(area)
    bins = [0, 0.001, 0.01, 0.1, 0.5, 1.]
    my_format = lambda x: str(x).rstrip("0").rstrip(".") if x != 0 else '0'
    precision = [f'{my_format(100*von)}-{my_format(100*bis)}\%' for von, bis in zip(bins[:-1], bins[1:])]

    data = {'Precision': precision}
    totals = []
    for source, df_seq_kmers in d_df_seq_kmers.items():
        # Compute histogram with custom bin edges
        hist, _ = np.histogram(df_seq_kmers[df_seq_kmers.presented].precision, bins=bins)
        data.update({f"{source} abs": hist})
        total = hist.sum()
        data.update({f"{source}": hist/total})
        totals.append(total)

    df = pd.DataFrame(data).set_index('Precision')
    df_bar = df[[f'{source}' for source in d_df_seq_kmers.keys()]].transpose()
    df_bar.plot(ax=ax, kind=kind, stacked=True, width=1.)
    

    for c in ax.containers:
        precision_category = c.get_label()
        labels = []
        for idx, _ in enumerate(c):
            if kind == 'barh':
                source = ax.get_yticklabels()[idx].get_text()
            elif kind == 'bar':
                source = ax.get_xticklabels()[idx].get_text()
            abs = df.at[precision_category, f"{source} abs"]
            pc = df.at[precision_category, f"{source}"]
            if kind == 'barh':
                labels.append(f"{abs}\n{pc*100:.1f}\%")
            elif kind == 'bar':
                labels.append(f"{abs} ({pc*100:.1f}\%)")
        
        ax.bar_label(c, label_type='center', labels = labels, size=16*font_scale)

    if kind == 'barh':
        ax.set_xlabel('proportion of epitiopes in precision categories')
        legend = ax.legend(loc='upper center', fancybox=False, shadow=False, ncol=len(bins), bbox_to_anchor=(0.5, 1.15))
        legend.set_alpha(0.1)
        ax.set_xlim(left=0., right=1.)
        ax.set_ylim(bottom=-0.5, top=1.5)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    elif kind == 'bar':
        ax.set_ylabel('proportion of epitiopes \n in precision categories', rotation=-90, labelpad=20)
        ax.yaxis.set_label_position("right")
        legend = ax.legend(loc='upper center', fancybox=False, shadow=False, ncol=2, bbox_to_anchor=(0.5, 1.3))
        legend.set_alpha(0.1)
        ax.set_ylim(bottom=0., top=1.)
        ax.set_xlim(left=-0.5, right=1.5)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.yaxis.tick_right()
        # plt.xticks(rotation=90)
    return ax


def plot_epitope_recall_by_seq(area, predictor, df_eval, seq_hash):
    fig = plt.gcf()
    ax = fig.add_subplot(area)

    sns.histplot(df_eval[f'recall_{seq_hash}'], ax=ax)
    ax.set_xlabel('visible 9-mer recall')
    ax.set_ylabel('\# natural sequences')
    ax.set_title('visible 9-mer recall distribution over natural sequences')
    ax.set_xlim(left=0.)

    return ax
