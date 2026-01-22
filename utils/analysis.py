import umap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA 
from sklearn.metrics import (balanced_accuracy_score, matthews_corrcoef, mean_absolute_error,
                             pairwise_distances, r2_score)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from utils.plotting import plot_cv_predictions, cluster_plot, plot_sorted_confusion_matrix


def compute_dataset_correlations(df, score, cols_to_ignore=[], method="spearman"):
    elapsed_time_cols = [col for col in df.columns if "elapsed" in col]
    metrics = df.drop(["model", "model_name_short", "architecture", "dataset", "model_family"]
                        + elapsed_time_cols + cols_to_ignore, axis=1).columns
    corr = df.drop(["model", "model_name_short", "architecture", "model_family"] 
                   + elapsed_time_cols + cols_to_ignore, axis=1).groupby("dataset").corr(method=method)

    mean_corrs = []
    std_corrs = []
    for metric in metrics:
        all_corrs = []
        for dataset in df["dataset"].unique():
            all_corrs.append(corr[metric][dataset][score])
        mean_corrs.append(np.nanmean(all_corrs))
        std_corrs.append(np.nanstd(all_corrs))

    return pd.DataFrame({f"{score} vs. ": metrics, 
                            "mean_abs_corrs": abs(np.array(mean_corrs)),
                            "std_corrs": std_corrs}).sort_values(by="mean_abs_corrs", ascending=False)



def compute_model_correlations(df, score, cols_to_ignore=[], method="spearman"):
    elapsed_time_cols = [col for col in df.columns if "elapsed" in col]
    metrics = df.drop(["model", "model_name_short", "architecture", "dataset"]
                      + elapsed_time_cols + cols_to_ignore, axis=1).columns

    # Filter dataset
    corr = df.drop(["dataset", "architecture", "model_name_short"]
                   + elapsed_time_cols + cols_to_ignore, axis=1).groupby("model").corr(method=method)

    mean_corrs = []
    std_corrs = []
    for metric in metrics:
        all_corrs = []
        for dataset in df["model"].unique():
            all_corrs.append(corr[metric][dataset][score])
        mean_corrs.append(np.nanmean(all_corrs))
        std_corrs.append(np.nanstd(all_corrs))

    return pd.DataFrame({f"{score} vs. ": metrics, 
                            "mean_abs_corrs": abs(np.array(mean_corrs)),
                            "std_corrs": std_corrs}).sort_values(by="mean_abs_corrs", ascending=False)


def signature_predict_cv(df, signature_columns, target, groups, hue, n_splits=3, n_feats=5,
                         model_type="regression", figsize=(3.5, 3), apply_clustering=False, baselines=[], use_pca=False, pca_components=5, normalize_axis="",
                         show_plot=True, log_feats=True, normalize_method="standard"):
    """
    Performs cross-validation on the provided dataframe, considering normalization, dimensionality reduction and 
    classification or regression tasks.
    """
    # Make sure to have a fresh index
    df = df.reset_index(drop=True)
    fi_columns = signature_columns
    all_results = []

    if n_splits == 1:
        # Simple train-test split
        train_index, test_index = train_test_split(df.index, test_size=0.3, stratify=groups
                                                   if model_type == "classification" else None, random_state=42)
        splits = [(train_index, test_index)]
    else:
        # GroupKFold cross-validation
        gkf = GroupKFold(n_splits=n_splits)
        splits = gkf.split(df[signature_columns], df[target], groups)

    # Create subplots
    if show_plot:
        fig_cols = n_splits if n_splits > 1 else 1
        fig, axes = plt.subplots(1, fig_cols, figsize=(figsize[0] * fig_cols, figsize[1]), sharex=False, sharey=False)
        if n_splits == 1:
            axes = [axes]  # Ensure axes is iterable

    y_preds = {
        "true": [],
        "group_labels": [],
        "signatures": []
    }

    for i, (train_index, test_index) in enumerate(splits):
        results = {}
        X_train, X_test = df[signature_columns].iloc[train_index], df[signature_columns].iloc[test_index]
        y_train, y_test = df[target].iloc[train_index], df[target].iloc[test_index]

        # Normalization
        if normalize_axis == "all":
            X_train, train_normalizer = normalize_data(X_train, normalize_method=normalize_method)
            X_test = normalize_data(X_test, normalize_method=normalize_method, train_normalizer=train_normalizer)
        elif normalize_axis == "dataset":
            normalized_features = []
            for _, group in df.groupby("dataset"):
                normalized_group, _ = normalize_data(group[signature_columns], normalize_method=normalize_method)
                normalized_features.append(pd.DataFrame(normalized_group, columns=signature_columns, index=group.index))
            normalized_df = pd.concat(normalized_features).sort_index()
            X = normalized_df[signature_columns]
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        
        # Dimensionality reduction
        if use_pca:
            pca = PCA(n_components=pca_components) 
            X_train = pd.DataFrame(pca.fit_transform(X_train.fillna(0)),
                                   columns=[f'pca{idx+1}' for idx in range(pca_components)])
            X_test = pd.DataFrame(pca.transform(X_test.fillna(0)))
            
            print(f"Explained variance for n_pca={pca_components}: {pca.explained_variance_ratio_}")
            fi_columns = [f'pca{idx+1}' for idx in range(pca_components)]

        if model_type == "regression":
            rf = RandomForestRegressor(random_state=42)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            train_loss = mean_absolute_error(y_train, rf.predict(X_train))
            print(f"Train Loss (MAE): {train_loss}")
            y_pred = rf.predict(X_test)
        elif model_type == "classification":
            rf = RandomForestClassifier(random_state=42,
                                        max_depth=5)
            rf.fit(X_train, y_train)
            train_pred = rf.predict(X_train)
            train_bal_acc = balanced_accuracy_score(y_train, train_pred)
            print(f"Train Balanced Accuracy: {train_bal_acc:.4f}")
            y_pred = rf.predict(X_test)
        
        importances = rf.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': fi_columns, 'Importance': importances})
        top_features = feature_importance_df.sort_values(by='Importance', ascending=False).head(n_feats)
        if log_feats:
            results["top_features"] = top_features.reset_index().to_dict()

        if model_type == "regression":
            r2 = r2_score(y_test, y_pred)
            r = spearmanr(y_test, y_pred)
            results["signature_r2"] = r2
            results["signature_spearman"] = r.statistic
            if show_plot:
                plot_cv_predictions(y_test, y_pred, df.iloc[test_index][hue], hue, target, axes[i])
        else:
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)
            results["signature_balanced_accuracy"] = bal_acc
            results["signature_mcc"] = mcc
            if show_plot:
                plot_sorted_confusion_matrix(df, target, y_test, y_pred, i, axes[i], apply_clustering)

        y_preds["signatures"].extend(y_pred.tolist())
        y_preds["true"].extend(y_test.tolist())
        y_preds["group_labels"].extend(df["dataset"].iloc[test_index].tolist())

        # ------------ Baselines ------------
        for baseline in baselines:
            baseline_indexer = [baseline] if isinstance(baseline, str) else baseline
            baseline_name = baseline if isinstance(baseline, str) else " + ".join(baseline)

            if baseline_name not in y_preds:
                y_preds[baseline_name] = []

            if model_type == "regression":
                bl = RandomForestRegressor(random_state=42)
                bl.fit(df[baseline_indexer].iloc[train_index], y_train)  
                y_pred = bl.predict(df[baseline_indexer].iloc[test_index])  
                results[f"{baseline_name}_r2"] = r2_score(y_test, y_pred)
                results[f"{baseline_name}_spearman"] = spearmanr(y_test, y_pred).statistic
            elif model_type == "classification":
                bl = RandomForestClassifier(random_state=42)
                bl.fit(df[baseline_indexer].iloc[train_index], y_train)  
                y_pred = bl.predict(df[baseline_indexer].iloc[test_index])  
                results[f"{baseline_name}_balanced_accuracy"] = balanced_accuracy_score(y_test, y_pred)
                results[f"{baseline_name}_mcc"] = matthews_corrcoef(y_test, y_pred)
            y_preds[baseline_name].extend(y_pred.tolist())
        all_results.append(results)   

    # Get handles and labels from the last axis
    if show_plot:   
        handles, labels = axes[-1].get_legend_handles_labels()
        if model_type == "regression":
            # Create shared legend
            fig.legend(
                handles, labels,
                loc='upper left',
                bbox_to_anchor=(0.92, 1),
                title=hue,
                fontsize=7,
                title_fontsize=8
            )
            fig.subplots_adjust(hspace=0.4, wspace=0.4)
        else:
            fig.colorbar(axes[-1].collections[0], ax=axes, fraction=0.02, pad=0.04, label=hue) 
            fig.subplots_adjust(wspace=0.4, right=0.85)

    return y_preds, pd.DataFrame(all_results)
        
 
def create_signature_clusters(df, color, signature_columns, method="umap", annotate_idx=False,
                              method_kwargs={}, shape=None, normalize_axis="all", 
                              normalize_method="max", palette="coolwarm", 
                              use_pca=True, pca_components=5, legend_name="",
                              save_name=None, alpha=0.85, markers=True, bbox_to_anchor: tuple = (1.05, 1),
                              legend_loc: str = "upper left", figsize=(8, 5)):
    plot_data = df.sort_values(by=color, ascending=False).reset_index(drop=True)

    # Normalize features within each dataset group
    if normalize_axis == "all":
        normalized_df = pd.DataFrame(normalize_data(plot_data[signature_columns], 
                                                    normalize_method=normalize_method)[0])
    elif normalize_axis == "dataset":
        normalized_features = []
        for _, group in plot_data.groupby("dataset"):
            normalized_group = normalize_data(group[signature_columns], 
                                              normalize_method=normalize_method)
            normalized_features.append(normalized_group)
        # Concatenate normalized groups
        normalized_df = pd.concat(normalized_features).sort_index()
    else:
        normalized_df = plot_data[signature_columns]

    if use_pca:
        pca = PCA(n_components=pca_components) 
        normalized_df = pd.DataFrame(pca.fit_transform(normalized_df.fillna(0)), columns=[f'pca{idx+1}' for idx in range(pca_components)], index=df.index)
        print(f"Explained variance for n_pca={pca_components}: {pca.explained_variance_ratio_}")


    if method == "umap":
        mapped_topology = umap.UMAP(
            **method_kwargs,
        ).fit_transform(normalized_df.fillna(0))
    elif method == "pca":
        pca = PCA(**method_kwargs)
        mapped_topology = pca.fit_transform(normalized_df)  

        # Derive the most contributing features for the principal components
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)  # END:
        feature_importance = pd.DataFrame(loadings, index=signature_columns, columns=[f'PC{i+1}' for i in range(loadings.shape[1])])
        top_features = feature_importance.abs().nlargest(5, 'PC1')  # Get top 5 features for PC1
        print("Top contributing features for PC1:")
        print(top_features)

    cluster_plot(color_feature=color,
                 embedding=mapped_topology,
                 metadata=plot_data,
                 method_name=method,
                 annotate_points=annotate_idx,
                 shape=shape,
                 palette=palette,
                 markerscale=1.5,
                 figsize=figsize,
                 legend_name=legend_name,
                 save_name=save_name,
                 alpha=alpha,
                 markers=markers,
                 bbox_to_anchor=bbox_to_anchor,
                 legend_loc=legend_loc)

    return normalized_df


def compute_pairwise_distances_grouped(df, signature_columns, group="dataset",
                                       sort_by="model_size", complete_models_only=True, use_pca=False,
                                       dist_metric="euclidean", normalize_axis="all",
                                       normalize_method="max", return_full_distances=False, pca_components=5):
    if complete_models_only:
        complete_models = df['model'].value_counts()
        complete_models = complete_models[complete_models == len(df['dataset'].unique())].index
        filtered_df = df[df['model'].isin(complete_models)].reset_index(drop=True).sort_values(by=sort_by)        
    else:
        filtered_df = df.copy().reset_index(drop=True).sort_values(by=sort_by)   

    # Normalization
    X = filtered_df[signature_columns].values
    if normalize_axis == "all":
        X, _ = normalize_data(X, normalize_method=normalize_method)
    elif normalize_axis == "":
        print("Skipping normalization")
        pass
    else:
        raise NotImplementedError("Normalization axis is not implemented.")

    # Dimensionality reduction
    if use_pca:
        pca = PCA(n_components=pca_components) 
        X = pd.DataFrame(pca.fit_transform(np.nan_to_num(X)),
                         columns=[f'pca{idx+1}' for idx in range(pca_components)], 
                         index=filtered_df.index)
    else:
        X = pd.DataFrame(X, columns=signature_columns, index=filtered_df.index)
    groups = filtered_df[group].unique()
    matrices = []
    print("Number of groups:", len(groups))
    assert X.index.equals(filtered_df.index)
    for g in groups:
        subset = filtered_df[filtered_df[group] == g]
        X_subset = X[filtered_df[group] == g].values	
                
        if return_full_distances:
            dist_matrix = np.zeros((X_subset.shape[0],
                                    X_subset.shape[0],
                                    X_subset.shape[1]))  # Shape [n, n, dist_per_dimension]
            for i in range(X_subset.shape[0]):
                for j in range(X_subset.shape[0]):
                    dist_matrix[i, j] = np.abs(X_subset[i] - X_subset[j])
            matrices.append(dist_matrix)
        else:
            dist_matrix = pairwise_distances(X_subset, metric=dist_metric)
            matrices.append(dist_matrix)
            
    return matrices, filtered_df, subset



def compute_intra_distances(matrices, 
                            last_sorted_group, 
                            prop="embedding_dimension",
                            agg_func=np.nanmean,
                            min_samples=0):
    # Prepare data
    aggregated_matrix = agg_func(np.stack(matrices), axis=0)

    # Make sure to ignore self-comparisons
    np.fill_diagonal(aggregated_matrix, np.nan)
    layered_df = pd.DataFrame(aggregated_matrix)
    layered_df.columns = last_sorted_group[prop].values
    layered_df.index = last_sorted_group[prop].values

    dist_data = {
        "dist": [],
        "family": [],
    }

    model_family_matrix = {}
    for model_family_src in last_sorted_group[prop].unique():
        payload = layered_df.loc[layered_df.index == model_family_src][model_family_src]
        if payload.shape[0] > min_samples:
            if payload.shape[0] == 0:
                dist_data["dist"].extend([np.nan])
                dist_data["family"].extend([model_family_src])
            else:
                tril_values = np.tril(payload)
                distances = tril_values[tril_values > 0]
                print(f"Values for {model_family_src}", len(distances.tolist()))
                dist_data["dist"].extend(distances.tolist())
                dist_data["family"].extend([model_family_src] * len(distances))
        else:
            print(f"Skipping {model_family_src} due to insufficient data.")

        for model_family_tgt in last_sorted_group[prop].unique():
            payload_tgt = layered_df.loc[layered_df.index == model_family_src][model_family_tgt]
            if payload_tgt.shape[0] > min_samples:
                if model_family_src not in model_family_matrix:
                    model_family_matrix[model_family_src] = []
                distances = payload_tgt.values
                model_family_matrix[model_family_src].append(agg_func(distances))
            else:
                if model_family_src not in model_family_matrix:
                    model_family_matrix[model_family_src] = []
                model_family_matrix[model_family_src].append(0)

        # Adding other inter-family data
        mask = layered_df.index.to_series().values[:, None] != layered_df.columns.to_series().values
        non_self_pair_values = layered_df.where(mask).stack()
        dist_data["dist"].extend(non_self_pair_values.values.tolist())
        dist_data["family"].extend(non_self_pair_values.shape[0] * ["non-family"])

    dist_df = pd.DataFrame(dist_data)
    dist_df["type"] = dist_df["family"].apply(lambda x: "family" if x != "non-family" else "non-family")
    return dist_df, model_family_matrix


def normalize_data(X, normalize_method="max", train_normalizer=None):
    if normalize_method == "standard":
        if train_normalizer is None:
            scaler = StandardScaler()
            return scaler.fit_transform(X), scaler
        else:
            return train_normalizer.transform(X)
    elif normalize_method == "max":
        if train_normalizer is None:
            return X / np.max(abs(X), axis=0), np.max(abs(X), axis=0) 
        else:
            return X / train_normalizer
    elif normalize_method == "mean":
        if train_normalizer is None:
            return X / np.mean(abs(X), axis=0), np.mean(abs(X), axis=0)   
        else:
            return X / train_normalizer
    elif normalize_method == "robust":
        if train_normalizer is None:
            scaler = RobustScaler()
            return scaler.fit_transform(X), scaler
        else:
            return X / train_normalizer
    else:
        raise ValueError(f"Unknown normalization method: {normalize_method}")
 