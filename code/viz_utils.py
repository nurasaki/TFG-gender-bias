import matplotlib.pyplot as plt
import seaborn as sns
# import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


sns.set_style()

def ax_histogram(data, labels=None, ax=None, log_scale=False, ylabel=None, 
                xlabel=None, show_percent=False, title=None, title_loc="center", 
                legend_fontsize="x-small", colors=sns.color_palette()):

    """Crea histrograma de les dades, afegeix labels i mitjana"""
    
    cols = data.columns
    legend_elements=[]
    legend_lines = []
    
    
    colors = colors[:len(cols)]
    # palette=sns.color_palette()[2:4]) 
    
    if ax is None:
        fig, ax = plt.subplots()

    # Create histogram with data
    sns.histplot(data, bins=30, log_scale=log_scale, kde=True, 
                legend=False, ax=ax, palette=colors) 
    
    # Add mean lines
    labels = cols if labels is None else labels
    for column, label, color in zip(cols, labels, colors):
        mean = data[column].mean()

        if show_percent:
            line = ax.axvline(x=mean, color=color, ls='--', label=f"mean: {mean/100:.0%}")
        else:
            line = ax.axvline(x=mean, color=color, ls='--', label=f"mean: {mean:.5f}")

        # Segur que hi ha una manera més elegant d'afegit la llegenda
        legend_elements.append(Line2D([0], [0], color=color, label=label))
        legend_lines.append(line)
    
    # Set labels and title
    ax.set_ylabel(ylabel) if ylabel is not None else ""
    ax.set_xlabel(xlabel) if xlabel is not None else ""
    ax.set_title(title, loc=title_loc) if title is not None else ""
    
    # Set legend
    ax.legend(handles=legend_elements+legend_lines, fontsize=legend_fontsize)

    return ax


def ax_boxplot(data, col_score, x_col, order=None, labels=None, ylabel=None, ax=None, log_scale=False, 
               title=None, title_loc="center", legend_fontsize="x-small", legend_labels=["Femení", "Masculí"], 
               color_palette=sns.color_palette(), box_kwargs={}):
    
    
    """Crea boxplot ax"""
    
    df_grup = data.groupby(x_col).agg({col_score: "mean"}).sort_values(col_score).reset_index()
    if order is None:
        # Definim l'ordre de les files
        order = df_grup[x_col]
    
    df_grup = df_grup.set_index(x_col).loc[order].reset_index()
    

    # palette = sns.color_palette()[:2] + [sns.color_palette()[6]]
    # colors = df_grup[col_score].apply(lambda x : palette[0] if x > 0.5 else (palette[1] if x < -0.5 else palette[2]))
    colors = df_grup[col_score].apply(lambda x: color_palette[x < 0])

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
        
    if log_scale:
        ax.set_yscale("log")

    
    ax.axhline(y=0, color="black", ls=':', zorder=0)

    sns.boxplot(data=data, y=col_score, x=x_col, order=order, palette=colors, ax=ax, **box_kwargs, zorder=10)
    
    # # Despine axes
    sns.despine(offset=20, trim=True)
    ax.set_xticklabels(order, rotation=40, ha='right', fontsize="x-small")
    
    
    ax.set(xlabel=None, ylabel=ylabel)
    
    ax.set_title(title, loc=title_loc)

    legend_elements = [
        Line2D([0], [0], color=color_palette[0], label=legend_labels[0]),
        Line2D([0], [0], color=color_palette[1], label=legend_labels[1])
    ]



    
    ax.legend(handles=legend_elements, fontsize=legend_fontsize)


    return ax, order


def viz_scores(df_scores, word=None):

    fig, axs = plt.subplots(2, 1, figsize=(6,10), sharex=True)
    # fig, axs = plt.subplots(3, 1, figsize=(12,6))
    
    words = ("", "")


    if word is not None:
        
        # word ha de ser o la paraula femenií o masculí
        filtre = ((df_scores.word_f == word) | (df_scores.word_m == word))
        df_scores = df_scores[filtre]
        
        print(word, filtre.sum(), "values")
        words = df_scores[['word_f', 'word_m']].values[0]
        
    
    for i, gender in enumerate(["f", "m"]):
        
        labels = ["Prob. sense gènere", "Prob. amb gènere"]
        data = df_scores[[f'P_TAM_{gender}', f'P_TM_{gender}']]    
        ylabel=f"({gender.upper()}) {words[i].upper()}"
        ax_histogram(data, labels=labels, ylabel=ylabel, ax=axs[i], log_scale=True)
        
    
    plt.subplots_adjust(hspace=0.01)
    plt.show()
    
    # Crea Histograma de "Log Odds ratios" (F. vs M.)
    fig, ax = plt.subplots(figsize=(6,5))
    labels = ["Log odds ratio (F)", "Log odds ratio (M)"]
    ax_histogram(df_scores[[f'Asso_f', "Asso_m"]], labels, ylabel='Log Odds ratio', ax=ax)
    
    plt.show()
