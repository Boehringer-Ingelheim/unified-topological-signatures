
import os
import time
import argparse
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from functools import reduce
from dotenv import load_dotenv
from sklearn.neighbors import NearestNeighbors

from config.eval import RETRIEVAL_TAKS
from config.metrics import METRICS_CONFIG
from metrics import gini_coefficient
from utils.general import (load_embeddings, process_embeddings, sample_vectors, 
                           get_retrievability_data, separate_query_vectors)


def compute_metric(vectors, metric, compute_func, model, dataset, is_local, kwargs):
    if pd.isna(vectors).any():
        # Count nan values
        nan_count = np.isnan(vectors).sum()
        print(f"Warning: {nan_count} NaN values found in vectors for \
              model {model} and dataset {dataset}. Replacing with zeros.")
        vectors = np.nan_to_num(vectors, nan=0.0)

    start = time.time()     
    result = compute_func(vectors, **kwargs)
    elapsed_time = time.time() - start

    if is_local:
        os.makedirs("local_data", exist_ok=True)
        # Save as numpy array
        cached_name = f"cache_{model}".replace("/", "_")
        local_file = f"local_data/{cached_name}_{dataset}_{metric}_{vectors.shape[0]}.npy"
        np.save(local_file, result)
        return None
    else:
        result = float(result)
        if "distance_metric" in kwargs:
            metric_name = f"{metric}_{kwargs['distance_metric']}"
        else:
            metric_name = metric

        return {
            metric_name: result,
            "model": model,
            "dataset": dataset,      
            f"elapsed_time_{metric_name}": elapsed_time,         
        }


def compute_metrics(vectors, dataset, model):
    results_data = []
    for metric in METRICS_CONFIG["metrics"]:
        compute_func = METRICS_CONFIG["metrics"][metric]["compute_func"]
        max_sample_size = METRICS_CONFIG["metrics"][metric]["max_sample_size"]
        is_local = METRICS_CONFIG["metrics"][metric].get("is_local", False)
        kwargs = METRICS_CONFIG["metrics"][metric]["kwargs"]
        requires_distance = METRICS_CONFIG["metrics"][metric].get("requires_distance", False)
        distance_metrics = METRICS_CONFIG.get("distance_metrics", [])

        for distance_metric in distance_metrics:
            if requires_distance:
                kwargs["distance_metric"] = distance_metric

            # Only compute metric if computationally feasible
            if vectors.shape[0] <= max_sample_size:
                result = compute_metric(vectors, metric, compute_func, model, dataset, is_local, kwargs)
                if result is not None:
                    results_data.append(result)

                if not requires_distance:
                    # Just one iteration needed
                    break

    # Merge dataframes
    results = reduce(lambda left, right: pd.merge(left, right, 
                                                  on=['model', 'dataset'], 
                                                  how='outer'), 
                     [pd.DataFrame(d, index=[0]) for d in results_data])
    return results


def first_larger_index(sorted_list, value):
    for i, elem in enumerate(sorted_list):
        if elem > value:
            return i
    return -1  # Return -1 if no such element exists


def add_mteb_results(task_data, results):
    for metric in METRICS_CONFIG["retrieval_scores"]:
        try:
            task_data[metric] = results["scores"]["test"][0][metric]
        except Exception:
            # We use the train data as fallback if no split is available
            # This doesn't really affect topology.
            try:
                task_data[metric] = results["scores"]["train"][0][metric]
            except Exception:
                task_data[metric] = results["scores"]["dev"][0][metric]
    return task_data


def get_embeddings_by_hash(data, texts):
    vectors = []
    for doc in texts:
        vector = data.get_vector(doc)
        vectors.append(vector)
    return np.array(vectors)


def get_top_bottom_knn(df, data, vectors, top_n=100, bottom_n=100, k=100, exlude_zeros=True):
    forward = list(range(0, top_n))
    reverse = list(range(0, -bottom_n, -1))

    if exlude_zeros:
        dat = df[df["retrievability_score"] > 0]
    else:
        dat = df

    results = {f"knn_{k}": [], "idx": [], "hash": [], "n_words": [], "text": [], "score": []}
    for idx in tqdm(forward + reverse):
        doc_data = dat.sort_values(by="retrievability_score", ascending=False).iloc[idx]
        vector = data.get_vector(doc_data["document_text"])
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric="cosine").fit(vectors)
        _, indices = nbrs.kneighbors(vector.reshape(1, -1))
        results[f"knn_{k}"].append(np.squeeze(vectors[indices]))
        results["idx"].append(idx)
        results["hash"].append(doc_data["hash"])   
        results["text"].append(doc_data["document_text"])   
        results["n_words"].append(doc_data["n_words"])  
        results["score"].append(doc_data["retrievability_score"])

    return pd.DataFrame(results)


def compute_global_signatures(vectors, model, task, retrievability_data, subset="corpus"):
    all_results = []
    # Sample until one larger than actual dataset size
    index = first_larger_index(METRICS_CONFIG["sample_sizes"], vectors.shape[0])
    if index == -1:
        sample_sizes = METRICS_CONFIG["sample_sizes"]
    else:
        sample_sizes = METRICS_CONFIG["sample_sizes"][:index + 1]

    for sample_size in sample_sizes:
        print("Computing metrics for model: ", model, " task: ", task, " sample size: ", sample_size)
        sampled_vectors = sample_vectors(vectors, sample_size, METRICS_CONFIG["random_seed"])
        task_data = compute_metrics(sampled_vectors, task, model)
        task_data["corpus_size"] = vectors.shape[0]
        task_data["sample_size"] = sample_size
        task_data["gini_at_100"] = gini_coefficient(retrievability_data["retrievability_score"].values)
        task_data["gini_at_100_dense"] = gini_coefficient(retrievability_data["retrievability_score"].values,
                                                          exclude_zeros=True)
        task_data = add_mteb_results(task_data, results)
        all_results.append(task_data)

    all_results = pd.concat(all_results, ignore_index=True)
    all_results["seed"] = METRICS_CONFIG["random_seed"]

    # Save
    metric_path = f"{os.environ.get('METRICS_PATH')}{model}/{task}"
    os.makedirs(metric_path, exist_ok=True)
    metric_file = f"{metric_path}/metrics_{subset}_{METRICS_CONFIG['random_seed']}.csv"
    all_results.to_csv(metric_file, index=False)
    print(f"Completed global {subset} signature: ", task, " for model: ", model)


def compute_query_signatures(data, model, task):
    path = f"detailed_results/{model}/{task}_query_neighborhood.pkl"
    with open(path, 'rb') as file:
        query_data = pd.DataFrame(pickle.load(file))

    # Collect results of all queries
    dfs = []
    for i, query in tqdm(query_data.iterrows()):
        neighborhood_vectors = get_embeddings_by_hash(data, query["texts"])
        neighborhood_task_data = compute_metrics(neighborhood_vectors, task, model)
        neighborhood_task_data = add_mteb_results(neighborhood_task_data, results)
        neighborhood_task_data["query"] = query["query"]
        neighborhood_task_data["mean_knn_text_len"] = np.mean([len(t) for t in query["texts"]])
        neighborhood_task_data["min_knn_text_len"] = np.min([len(t) for t in query["texts"]])
        neighborhood_task_data["max_knn_text_len"] = np.max([len(t) for t in query["texts"]])
        neighborhood_task_data['recall_5'] = query["recall_5"]
        neighborhood_task_data['recall_10'] = query["recall_10"]
        neighborhood_task_data['recall_20'] = query["recall_20"]
        neighborhood_task_data['ndcg_cut_5'] = query["ndcg_cut_5"]
        neighborhood_task_data['ndcg_cut_10'] = query["ndcg_cut_10"]
        neighborhood_task_data['ndcg_cut_20'] = query["ndcg_cut_20"]
        sim_bins = np.histogram(query["sim_scores"], bins=10)[0].tolist()

        for bin_index, bin_value in enumerate(sim_bins):
            neighborhood_task_data[f"sim_bin_{bin_index}"] = bin_value
        dfs.append(pd.DataFrame(neighborhood_task_data))

    dfs = pd.concat(dfs, ignore_index=True)
    dfs["seed"] = METRICS_CONFIG["random_seed"]

    # Save
    metric_path = f"{os.environ.get('METRICS_PATH')}{model}/{task}"
    os.makedirs(metric_path, exist_ok=True)
    metric_file = f"{metric_path}/metrics_queries_local_{METRICS_CONFIG['random_seed']}.csv"
    dfs.to_csv(metric_file, index=False)
    print("Completed query signature: ", task, " for model: ", model)


def compute_retrievability_signatures(data, retrievability_data):
    vectors = get_embeddings_by_hash(data, retrievability_data["document_text"])
    res = get_top_bottom_knn(retrievability_data, data, vectors)

    # Compute signature for every row
    res['type'] = res.idx.apply(lambda x: "top" if x >= 0 else "bottom")

    for metric in METRICS_CONFIG["metrics"]:
        compute_func = METRICS_CONFIG["metrics"][metric]["compute_func"]
        is_local = METRICS_CONFIG["metrics"][metric].get("is_local", False)
        kwargs = METRICS_CONFIG["metrics"][metric]["kwargs"]
        requires_distance = METRICS_CONFIG["metrics"][metric].get("requires_distance", False)
        distance_metrics = METRICS_CONFIG.get("distance_metrics", [])

        if not is_local:
            for distance_metric in distance_metrics:
                if requires_distance:
                    kwargs["distance_metric"] = distance_metric                
                    res[metric + f"_{distance_metric}"] = res['knn_100'].apply(lambda x: compute_func(x, **kwargs)) 
                else:
                    res[metric] = res['knn_100'].apply(lambda x: compute_func(x, **kwargs))
                    break

    metric_path = f"{os.environ.get('METRICS_PATH')}/{model}/{task}"
    os.makedirs(metric_path, exist_ok=True)
    metric_file = f"{metric_path}/metrics_retriev_{METRICS_CONFIG['random_seed']}.csv"
    res = res.drop(columns=["knn_100"])
    res["seed"] = METRICS_CONFIG["random_seed"]
    res["model"] = model
    res["dataset"] = task
    res.to_csv(metric_file, index=False)
    print("Completed retrievability signature: ", task, " for model: ", model)


# ----------------- Main Execution -----------------
load_dotenv()
parser = argparse.ArgumentParser(
    prog='Text Embedding Metrics Computation',
    description='Parallel computation of topological features.')
parser.add_argument('--model')
args = parser.parse_args()
if not args.model:
    raise ValueError("No model name provided.")
else:
    models = [args.model]

for model in models:
    for task in RETRIEVAL_TAKS:
        # Load data from cache
        cached_name = f"cache_{model}".replace("/", "_")
        data, results = load_embeddings(cached_name, task)
        if data is None:
            print(f"No data found for model: {model}, task: {task}. Skipping...")
            continue
        vectors = process_embeddings(data)
        try:
            query_vectors, collection_vectors = separate_query_vectors(data, vectors, model, task)
        except Exception:
            print(f"Error loading query data: {model}, task: {task}. Skipping...")
            continue
        retrievability_data = get_retrievability_data(model, task)
    
        # ----- Global measures ------
        compute_global_signatures(collection_vectors, model, task, retrievability_data, subset="corpus")
        compute_global_signatures(query_vectors, model, task, retrievability_data, subset="queries")

        # ------- Query-based measures -------
        compute_query_signatures(data, model, task)

        # ------- Retrievability of documents -------
        compute_retrievability_signatures(data, retrievability_data)