import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import linkage, leaves_list

def plot_dataset_correlations(df, x, y, hue, style, size, save_name):
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.2)
    plt.rcParams.update({
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    hue_order = df[hue].unique()
    style_order = sorted(df[style].unique())
    size_order = sorted(df[size].unique())

    # Grid setup
    num_tasks = len(hue_order)
    cols = 3
    rows = (num_tasks + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3 * rows))
    axes = axes.flatten()

    # Collect handles and labels
    all_labels = {}

    for i, h in enumerate(hue_order):
        ax = axes[i]
        subset = df[df[hue] == h]
        bi = sns.color_palette(["#08312A",
                                "#00E47C",
                                "#ffd03d",
                                "#6ad2e2",
                                "#ee6541",
                                "#928bde",
                                "#86251b",
                                "#076d7e",
                                "#e18600"])
        scatter = sns.scatterplot(
            data=subset, x=x, y=y,
            hue=style, style=style, size=size, palette=bi, #palette="deep",
            sizes=(50, 200), alpha=1, ax=ax,
            hue_order=style_order, style_order=style_order, size_order=size_order
        )

        


        sns.regplot(data=subset, x=x, y=y, scatter=False, ax=ax, 
                    ci=None, truncate=True, line_kws={"linestyle": "--", "alpha": 0.5, "color": "gray"})

        ax.set_title(f'{h}', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_xlabel(x, fontsize=9)
        ax.set_ylabel(y, fontsize=9)
        ax.legend().remove()
        ax.grid(True)  # Turn on grid
        ax.set_axisbelow(True)  # Ensure grid is behind plot elements
        ax.grid(which='major', linestyle='-', linewidth=0.5, color='lightgray', alpha=0.7)

        # Spearman
        rho, _ = spearmanr(subset[x], subset[y])
        badge_text = f"$\\rho$ = {rho:.2f}"

        # Add badge to top-right corner
        ax.text(
            0.95, 0.15, badge_text,
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', edgecolor='none', alpha=0.5)
        )

        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('gray')

        # Collect legend handles and labels
        handles, labels = scatter.get_legend_handles_labels()

        # dont show model size [-6:]
        for handle, label in zip(handles[:-6], labels[:-6]):
            if label not in all_labels:
                all_labels[label] = handle

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.legend(all_labels.values(), 
               all_labels.keys(), 
               loc='upper left', 
               bbox_to_anchor=(1, 1), 
               markerscale=2)
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def plot_gradient_heatmap(df, heat, model_sorting, save_name=None):
    annotation_value = heat
    annotation = heat

    # Only on complete or biased!
    model_col = "model_name_short"
    complete_models = df[model_col].value_counts()
    complete_models = complete_models[complete_models == len(df['dataset'].unique())].index
    filtered_df = df[df[model_col].isin(complete_models)]

    # Plot
    ranked_df = filtered_df.groupby('dataset').apply(lambda x: x.assign(rank=x[annotation_value].rank(ascending=False) - 1)).reset_index(drop=True)
    ranked_df = ranked_df[['dataset', model_col, annotation_value, 'rank', model_sorting]]    

    # Row sorting by dataset median
    dataset_order = ranked_df.groupby('dataset')[annotation_value].median().sort_values(ascending=False).index
    ranked_df['dataset'] = pd.Categorical(ranked_df['dataset'], categories=dataset_order, ordered=True)
    ranked_df = ranked_df.sort_values('dataset')

    # Column sorting by model median
    model_order = ranked_df.groupby(model_col)[model_sorting].median().sort_values(ascending=False).index
    ranked_df[model_col] = pd.Categorical(ranked_df[model_col], categories=model_order, ordered=True)
    ranked_df = ranked_df.sort_values(model_col)

    sns.set_theme(style="ticks", rc={'figure.figsize': (12, 6)})
    palette = sns.blend_palette(["#d8a5a6",  "#9f0000"], n_colors=100)
    cbar_kws = {
                'extend':'both',
                'label': annotation.replace("_", " ").upper()
            } 

    ranked_df_pivot = ranked_df.pivot(index='dataset', columns=model_col, values=annotation)
    #boehringer_blue_gradient = sns.light_palette("#002F62", n_colors=100, as_cmap=True)

    ax = sns.heatmap(ranked_df_pivot,
                    cbar_kws=cbar_kws, 
                    cmap="Reds", 
                    fmt=".2f" if ranked_df_pivot.max().max() < 10 else ".0f",
                    xticklabels=True, 
                    yticklabels=True,
                    annot=True, 
                    linewidths=1,
                    annot_kws={"size": 7})  # BEGIN: Reduce annotation font size
    plt.xticks(rotation=45, ha='right', fontsize=7, fontweight='normal')  
    plt.yticks(fontsize=7)
    plt.xlabel("")  # Disable x label
    plt.ylabel("")  # Disable y label
    

    # Fontsize
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(cbar_kws['label'], fontsize=9)

    # Add BEST and WORST labels above the heatmap
    ax.text(-0.02, 1.02, "BEST", transform=ax.transAxes, fontsize=10, fontweight='bold', ha='left', va='bottom')
    ax.text(1.02, 1.02, "WORST", transform=ax.transAxes, fontsize=10, fontweight='bold', ha='right', va='bottom')
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def correlation_matrix_datasets(df, ordered_feats, cmap=sns.diverging_palette(250, 10, as_cmap=True), figsize=(12, 10), apply_clustering=False, name_mapping=None, save_name=""):
    # Group the data by dataset
    grouped = df.groupby('dataset')

    # Compute correlation matrices for each dataset group
    correlation_matrices = []
    for _, group in grouped:
        available_feats = [f for f in ordered_feats if f in group.columns]
        df_feats = group[available_feats].dropna(axis=1, how='all')
        corr = df_feats.corr(method='spearman')
        correlation_matrices.append(corr)

    if name_mapping:
        index_name = ordered_feats if name_mapping is None else [name_mapping.get(col, col) for col in ordered_feats]
    else:
        index_name = ordered_feats

    # Align all correlation matrices to the same index and columns
    aligned_corrs = [corr.reindex(index=ordered_feats, columns=ordered_feats) for corr in correlation_matrices]

    # Stack matrices into a 3D array
    stacked = np.stack([c.values for c in aligned_corrs])

    # Compute mean and std deviation
    mean_corr = np.nanmean(stacked, axis=0)
    std_corr = np.nanstd(stacked, axis=0)

    # Create a mask for the upper triangle
    triu_mask = np.triu(np.ones_like(mean_corr, dtype=bool))
    # Create a mask for the diagonal
    diag_mask = np.eye(mean_corr.shape[0], dtype=bool)

    if apply_clustering:
        link = linkage(mean_corr, method='ward')
        cluster_order = leaves_list(link)
        mean_corr = mean_corr[cluster_order][:, cluster_order]
        index_name = np.array(index_name)[cluster_order]

    # Create annotation labels with line break between mean and std
    annot = np.empty_like(mean_corr, dtype=object)
    for i in range(mean_corr.shape[0]):
        for j in range(mean_corr.shape[1]):
            if not np.isnan(mean_corr[i, j]):
                annot[i, j] = f"{mean_corr[i, j]:.2f}\n±{std_corr[i, j]:.2f}"
            else:
                annot[i, j] = ""

    # Plot the heatmap
    plt.figure(figsize=figsize)
    sns.set(style='whitegrid')
    sns.heatmap(
        mean_corr,
        mask=diag_mask,
        cmap=cmap,
        annot=annot,
        fmt="",
        annot_kws={"size": 9},
        square=True,
        center=0,
        linewidths=1,
        linecolor='white',
        cbar_kws={"shrink": 0.6, "label": "Spearman Correlation"},
        vmin=-1, vmax=1
    )
    
    
    plt.xticks(ticks=np.arange(len(index_name)) + 0.5, labels=index_name, rotation=45, ha='right', fontsize=10)
    plt.yticks(ticks=np.arange(len(index_name)) + 0.5, labels=index_name, rotation=0, fontsize=10)
    plt.tight_layout()

    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0.1)

    plt.show()


def plot_cv_predictions(all_targets, all_preds, hues, hue, target, axis):
    # Summary
    p = sns.scatterplot(x=all_targets, 
                        y=all_preds, 
                        hue=hues, ax=axis)
    
    # Add regression line
    sns.regplot(x=all_targets, 
                y=all_preds, 
                scatter=False, 
                ax=axis, 
                color='grey', 
                line_kws={"linewidth": 1, "linestyle": "--"})

    # Add Spearman correlation as a toast in top-right
    rho = spearmanr(all_targets, all_preds).statistic
    toast_text = f"$\\rho$ = {rho:.2f}"
    axis.text(
        0.4, 0.95, toast_text,
        transform=axis.transAxes,
        ha='right', va='top',
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', edgecolor='none', alpha=0.5)
    )

    for spine in p.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('gray')

    axis.legend_.remove()
    axis.set_xlim(axis.get_xlim())
    axis.set_ylim(axis.get_ylim())
    axis.tick_params(axis='both', which='major', labelsize=8)
    axis.set_xlabel(f"True {target}", fontsize=8)
    axis.set_ylabel(f"Predicted {target}", fontsize=8)
    axis.grid(True)


def cluster_plot(
    color_feature: str,
    embedding: np.ndarray,
    metadata: pd.DataFrame,
    method_name: str,
    annotate_points: bool = False,
    shape: str = None,
    palette: str = "coolwarm",
    markerscale: int = 2,
    figsize: tuple = (10, 6),
    legend_name: str = "",
    save_name: str = None,
    alpha: float = 0.85,
    markers: bool = True,
    bbox_to_anchor: tuple = (1.05, 1),
    legend_loc: str = "upper left"
) -> None:

    # Initialize plot
    plt.figure(figsize=figsize)
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11
    })

    scatter = sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=metadata[color_feature],
        size=metadata["model_size"],
        style=metadata[shape] if shape else None,
        palette=palette,
        sizes=(100, 200),
        alpha=alpha,
        edgecolor='white',
        linewidth=0.5,
        markers=markers
    )

    # Optional annotation
    if annotate_points:
        for idx, (x, y) in enumerate(embedding):
            plt.text(
                x, y, str(metadata.index[idx]),
                fontsize=7, ha='right', va='center'
            )

    # Axis labels and ticks
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
    plt.xlabel(f"{method_name.upper()} dimension 1", fontsize=12)
    plt.ylabel(f"{method_name.upper()} dimension 2", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Remove size handles
    handles, labels = scatter.get_legend_handles_labels()
    all_labels = {}

    ignore_labels = False
    for handle, label in zip(handles, labels):
        if label == "model_size":
            ignore_labels = True
            continue
        if label == "type":
            all_labels[legend_name] = handle
            continue
        if label == "score":
            all_labels[legend_name] = handle
            continue
        if ignore_labels and not (label.isnumeric() or label.replace('.', '', 1).isdigit()):
            ignore_labels = False
        if ignore_labels:
            continue
        if label not in all_labels:
            all_labels[label] = handle

    # Legend formatting
    legend = plt.legend(
        all_labels.values(),
        all_labels.keys(),
        bbox_to_anchor=bbox_to_anchor,
        loc=legend_loc,
        borderaxespad=0.,
        markerscale=markerscale
    )
    plt.setp(legend.get_title(), fontsize=12)
    plt.setp(legend.get_texts(), fontsize=11)
    plt.tight_layout()
    sns.despine(top=True, right=True, left=False, bottom=False)
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def plot_cluster_differences(cluster_1, cluster_2, name_mapping=None, legend_name="Retrievability", 
                             save_name=None, colors=["#1f77b4", "#ff7f0e"],
                             cluster_1_name="Top", cluster_2_name="Bottom", alpha=0.8):
    # Combine into one DataFrame
    cluster_df = pd.DataFrame({
        cluster_1_name: cluster_1,
        cluster_2_name: cluster_2
    })

    # Calculate absolute difference
    cluster_df["abs_dif"] = abs(cluster_df[cluster_2_name] - cluster_df[cluster_1_name])
    cluster_df[cluster_1_name] = abs(cluster_df[cluster_1_name])
    cluster_df[cluster_2_name] = abs(cluster_df[cluster_2_name])

    # Sort by absolute difference and reset index
    cluster_df = cluster_df.sort_values(by="abs_dif", ascending=False)
    cluster_df = cluster_df.reset_index()

    # Plotting
    #sns.set(style="whitegrid", font_scale=1)
    ax = cluster_df[[cluster_1_name, cluster_2_name]].plot(
        width=0.8,  
        kind='bar',
        figsize=(10, 5),
        color=colors,
        #linewidth=0.5,
        alpha=alpha
    )

    # Axis labels
    ax.set_ylabel('Normalized value', fontsize=12)
    x_labels = cluster_df["index"]
    if name_mapping:
        x_labels = [name_mapping.get(col, col) for col in x_labels]
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
    ax.axhline(0, color='gray', linewidth=1, linestyle='-')
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.legend(title=legend_name, loc='upper right', fontsize=9, title_fontsize=10)
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()

    return cluster_df


def plot_sorted_confusion_matrix(df, target, y_test, y_pred, i, axis, apply_clustering):
    # plot
    if target == "model":
        model_size_map = df[['model', 'model_size']].drop_duplicates().set_index('model')['model_size']
        sorted_labels = sorted(np.unique(y_test), key=lambda m: model_size_map[m])
        label_names = [f"{model} ({model_size_map[model]})" for model in sorted_labels]
    else:
        sorted_labels = np.unique(y_test)
        label_names = sorted_labels.tolist()
    cm = confusion_matrix(y_test, y_pred, labels=sorted_labels)
    
    # accuracy dfs
    # Compute per-model accuracy (diagonal / row sum)
    model_labels = np.unique(y_test)
    model_accuracy = cm.diagonal() / cm.sum(axis=1)

    # Create a DataFrame for model accuracy
    accuracy_df = pd.DataFrame({
    'model': model_labels,
    'accuracy': model_accuracy,
    "split ": i
    })

    if apply_clustering:
        link = linkage(cm, method='ward')
        cluster_order = leaves_list(link)
        cm = cm[cluster_order][:, cluster_order]
        label_names = np.array(label_names)[cluster_order]

    # Plot
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names,
                annot_kws={"size": 10}, ax=axis)
    
    # Remove color label
    axis.collections[0].colorbar.remove()
    # set xtricks rotation
    axis.set_xticklabels(axis.get_xticklabels(), rotation=45, ha='right', fontsize=6)
    axis.set_yticklabels(axis.get_yticklabels(), rotation=0, fontsize=6)


def plot_pca_loadings(df, signature_columns, components, normalize_by_dataset=True, normalize_all=False,
                      normalize_method='standard', name_mapping=None, save_name=None):
    if normalize_all:
        if normalize_method == 'standard':
            scaler = StandardScaler()
            normalized_df = scaler.fit_transform(df[signature_columns])
        elif normalize_method == 'max':
            normalized_df = df[signature_columns] / df[signature_columns].abs().max()
        elif normalize_method == 'mean':
            normalized_df = df[signature_columns] / df[signature_columns].abs().mean()
    elif normalize_by_dataset:
        # Normalize features by dataset
        normalized_features = []
        for dataset_name, group in df.groupby("dataset"):
            if normalize_method == 'max':
                normalized_group = group[signature_columns] / group[signature_columns].abs().max()
            if normalize_method == 'mean':
                normalized_group = group[signature_columns] / group[signature_columns].abs().mean()
            elif normalize_method == 'standard':
                scaler = StandardScaler()
                scaled = scaler.fit_transform(group[signature_columns])
                normalized_group = pd.DataFrame(scaled, columns=signature_columns, index=group.index)
            normalized_features.append(normalized_group)
        normalized_df = pd.concat(normalized_features).sort_index()
    else:
        normalized_df = df[signature_columns].copy()

    # Perform PCA
    pca = PCA(n_components=components)
    X_train = pca.fit_transform(normalized_df)
    explained_variance = pca.explained_variance_ratio_

    # PCA loadings
    index_name = signature_columns if name_mapping is None else [name_mapping.get(col, col) for col in signature_columns]
    loadings_df = pd.DataFrame(pca.components_.T,
                               index=index_name,
                               columns=[f'PC{i+1}' for i in range(pca.n_components)])

    # Ensure consistent ordering of PCs
    sorted_indices = np.argsort(-explained_variance)
    explained_variance = explained_variance[sorted_indices]
    loadings_df = loadings_df.iloc[:, sorted_indices]
    loadings_df.columns = [f'PC{i+1}' for i in range(len(loadings_df.columns))]

    # Create subplots
    sns.set(style='whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Barplot of each PC
    ax1.bar(range(1, len(explained_variance) + 1), explained_variance, 
            alpha=0.6, color='blue', linewidth=1, edgecolor='black')

    # Explained variance plot
    cumulative_variance = np.cumsum(explained_variance)
    ax1.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', 
             linestyle='-', color='black', linewidth=2)
    ax1.set_xlabel('Principal Component', fontsize=10)
    ax1.set_ylabel('Explained Variance', fontsize=10)
    ax1.grid(True, linestyle='--', linewidth=0.5)
    ax1.tick_params(axis='both', labelsize=9)
    ax1.set_xticks(range(1, len(cumulative_variance) + 1))

    # Add dashed line at the final dot and annotate
    final_pc = len(cumulative_variance)
    final_value = cumulative_variance[-1]
    ax1.axhline(y=final_value, linestyle='--', color='grey', linewidth=1)
    ax1.annotate(f'{final_value:.2f}', xy=(final_pc, final_value), 
                 xytext=(final_pc+0.1, final_value-0.05),
                 fontsize=10, color='#2F2F2F', fontweight='bold')

    # Heatmap of loadings
    heatmap = sns.heatmap(
        loadings_df,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        center=0, 
        cbar_kws={'shrink': 0.8},
        ax=ax2,
        linewidths=0.8,
        linecolor='white',
        annot_kws={"size": 10, "va": "center"}
    )
    heatmap.figure.axes[-1].set_ylabel('Loading Value', fontsize=10)

    ax2.set_xlabel("Principal Components", fontsize=10)
    ax2.tick_params(axis='both', labelsize=9)

    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()

    return loadings_df



def plot_pca_loadings2(df, signature_columns, components, normalize_by_dataset=True, normalize_all=False,
                      normalize_method='standard', name_mapping=None, save_name=None):
    if normalize_all:
        if normalize_method == 'standard':
            scaler = StandardScaler()
            normalized_df = scaler.fit_transform(df[signature_columns])
        elif normalize_method == 'max':
            normalized_df = df[signature_columns] / df[signature_columns].abs().max()
        elif normalize_method == 'mean':
            normalized_df = df[signature_columns] / df[signature_columns].abs().mean()
    elif normalize_by_dataset:
        # Normalize features by dataset
        normalized_features = []
        for dataset_name, group in df.groupby("dataset"):
            if normalize_method == 'max':
                normalized_group = group[signature_columns] / group[signature_columns].abs().max()
            if normalize_method == 'mean':
                normalized_group = group[signature_columns] / group[signature_columns].abs().mean()
            elif normalize_method == 'standard':
                scaler = StandardScaler()
                scaled = scaler.fit_transform(group[signature_columns])
                normalized_group = pd.DataFrame(scaled, columns=signature_columns, index=group.index)
            normalized_features.append(normalized_group)
        normalized_df = pd.concat(normalized_features).sort_index()
    else:
        normalized_df = df[signature_columns].copy()

    # Perform PCA
    pca = PCA(n_components=components)
    X_train = pca.fit_transform(normalized_df)
    explained_variance = pca.explained_variance_ratio_

    # PCA loadings
    index_name = signature_columns if name_mapping is None else [name_mapping.get(col, col) for col in signature_columns]
    loadings_df = pd.DataFrame(pca.components_.T,
                               index=index_name,
                               columns=[f'PC{i+1}' for i in range(pca.n_components)])

    # Ensure consistent ordering of PCs
    sorted_indices = np.argsort(-explained_variance)
    explained_variance = explained_variance[sorted_indices]
    loadings_df = loadings_df.iloc[:, sorted_indices]
    loadings_df.columns = [f'PC{i+1}' for i in range(len(loadings_df.columns))]

    # Create subplots
    plt.figure(figsize=(6, 5))
    sns.set(style='whitegrid')
    # Heatmap of loadings
    heatmap = sns.heatmap(
        loadings_df,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        center=0, 
        cbar_kws={'shrink': 0.8},
        linewidths=0.8,
        linecolor='white',
        annot_kws={"size": 11, "va": "center"}
    )
    heatmap.tick_params(axis='both', labelsize=10)
    heatmap.collections[0].colorbar.ax.set_ylabel('Loading Value', fontsize=10)
    heatmap.collections[0].colorbar.ax.tick_params(labelsize=10) 

    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()

    return loadings_df


def plot_and_aggregate_dist_matrices(matrices, filtered_df, last_sorted_group, agg_func=np.mean, ticks_key='model', figsize=(10, 10), colors=["#1f77b4", "#ffffff", "#d62728"], apply_clustering=False, save_name=None):
    aggregated_distances = agg_func(np.stack(matrices), axis=0)
    order = last_sorted_group[ticks_key].values
    if ticks_key == 'model_name_short':
        order_labels = [f"{model} ({filtered_df[filtered_df['model_name_short'] == model]['embedding_dimension'].values[0]} / {filtered_df[filtered_df['model_name_short'] == model]['model_size'].values[0]})" for model in order]
    elif ticks_key == 'dataset':
        order_labels = [f"{dataset} ({filtered_df[filtered_df['dataset'] == dataset]['corpus_size'].values[0]})" for dataset in order]

    if apply_clustering:
        link = linkage(aggregated_distances, method='ward')
        cluster_order = leaves_list(link)
        aggregated_distances = aggregated_distances[cluster_order][:, cluster_order]
        order_labels = np.array(order_labels)[cluster_order]
    
    plt.figure(figsize=figsize)
    hex_colors = colors
    cmap = LinearSegmentedColormap.from_list("custom_hex", hex_colors, N=256)
    mask = np.eye(aggregated_distances.shape[0], dtype=bool) 

    ax = sns.heatmap(
        aggregated_distances,
        mask=mask,
        cmap="coolwarm", #cmap,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        square=True,
        linewidths=0.8,
        linecolor='white',
        cbar_kws={"shrink": 0.5},
        # Keep x axis simple
        xticklabels=order if not apply_clustering else np.array(order)[cluster_order], 
        yticklabels=order_labels
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Distance", fontsize=9)

    plt.xticks(fontsize=6, rotation=80)
    plt.yticks(fontsize=6)
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def plot_clustered_family_dist_heatmap(family_dist_matrix, 
                                       colors=["#1f77b4", "#ffffff", "#d62728"], save_name=None,
                                       cluster=False,
                                       dropna=True,
                                       defined_order=None):
    sns.set(style='whitegrid')
    # Get family matrix
    family_matrix = pd.DataFrame(family_dist_matrix)
    if dropna:
        family_matrix = family_matrix.dropna(axis=1).fillna(0)
    else:
        family_matrix = family_matrix.fillna(0)
    family_matrix.index = family_matrix.columns
    family_matrix_np = family_matrix.values
    
    if cluster:
        link = linkage(family_matrix_np, method='ward')
        order = leaves_list(link)
        reordered = family_matrix_np[order][:, order]
        new_ticks = np.array(family_matrix.columns)[order]
    elif defined_order:
        order = [family_matrix.columns.get_loc(name) for name in defined_order if name in family_matrix.columns]
        reordered = family_matrix_np[order][:, order]
        new_ticks = np.array(family_matrix.columns)[order]
    else:
        reordered = family_matrix_np
        new_ticks = family_matrix.columns
    plt.figure(figsize=(8, 8))

    cmap = LinearSegmentedColormap.from_list("custom_hex", colors, N=256)
    ax = sns.heatmap(
        reordered,
        annot=True,
        fmt=".2f",
        cmap="coolwarm", #cmap,
        annot_kws={"size": 12},
        square=True,
        linewidths=0.8,
        linecolor='white',
        cbar_kws={"shrink": 0.8},  
        xticklabels=new_ticks, yticklabels=new_ticks
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("UTS-Distance", fontsize=12)
    plt.xticks(fontsize=12, rotation=80)
    plt.yticks(fontsize=12, rotation=0)
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()