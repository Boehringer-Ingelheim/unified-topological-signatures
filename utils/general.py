import os
import yaml
import textwrap
import subprocess
import json
import glob
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler, RobustScaler
from mteb.models.cache_wrapper import TextVectorMap


def load_embeddings(model, task):
    print("Loading embeddings for model: ", model, " and task: ", task)
    # Load embeddings memmap
    cache_path = os.environ.get("CACHE_PATH") + model + f"/{task}"
    data = TextVectorMap(cache_path)
    data.load(name=task)

    # Look for mteb results
    org_model = model.replace("cache_", "").split("_")
    org = org_model[0]

    if len(org_model) > 2:
        model_dir = "_".join(org_model[1:])
    else:
        model_dir = org_model[1]
    try:
        if org == "dunzhang":
            model_path = f"NovaSearch__{model_dir}"
        elif org == "google":
            model_path = "no_model_name_available"
        elif org == "random":
            model_path = "no_model_name_available"
        else:
            model_path = f"{org}__{model_dir}"

        path = f"results/{org}/{model_dir}/{model_path}"
        print("Looking for files at: ", path)
        if list(os.walk(path)) == []:
            print("ERROR: No revision found.")
        print(list(os.walk(path)))
        revision = list(os.walk(path))[0][1][0]
        result_path = f"{path}/{revision}/"
        path = result_path + f"/{task}.json"
        with open(path, 'r') as file:
            results = json.load(file)
        
        return data, results
    except:
        print(f"Results file not found for {task} and {model}.")
        return None, None
        
def process_embeddings(data):
    # Remove zero vectors (without loading entire memmap)
    last_non_zero_row = data.vectors.shape[0] - 1
    while last_non_zero_row >= 0 and np.all(data.vectors[last_non_zero_row] == 0):
        last_non_zero_row -= 1

    truncated_data = data.vectors[:last_non_zero_row]
    return np.array(truncated_data, dtype=np.float16)

def sample_vectors(truncated_data, n, seed):
    if truncated_data.shape[0] > n:
        np.random.seed(seed)
        index = np.random.choice(truncated_data.shape[0], n, replace=False) 
        return truncated_data.take(index, axis=0)
    else:
        print("Not enough vectors to sample from, returning all vectors.")
        return truncated_data

# def separate_query_vectors(cache_data, all_vectors, model, dataset):
#     query_hashes = pd.read_pickle(f"detailed_results/{model}/{dataset}_query_hashes.pkl")
#     query_df = pd.DataFrame(query_hashes, columns=["query_text", "hash"])
#     query_df["vectors"] = query_df["query_text"].apply(lambda x: cache_data.get_vector(x))

#     # We assume that the first docs in the corpus are the query vectors
#     # Here we make sure to remove only documents that have the same embeddings
#     query_indices_in_cache = []
#     for i, q in query_df.iterrows():
#         print(i)
#         embedding_q = np.array(q["vectors"], dtype=np.float16)
#         match_arr = np.where(np.all(all_vectors == embedding_q, axis=1))[0]
#         if match_arr.shape[0] > 0:
#             match_idx = match_arr[0]
#             query_indices_in_cache.append(match_idx)
#         else:
#             print(f"Embedding vector for indx {i} not found in corpus.")
#         if i % 10000 == 0 and i > 0:
#             print(f"Processed {i} query vectors.")

#     query_vectors = np.stack(query_df["vectors"].values)
#     corpus_vectors = np.delete(all_vectors, query_indices_in_cache, axis=0)
#     return query_vectors, corpus_vectors


def separate_query_vectors(cache_data, all_vectors, model, dataset):
    # Load query hashes and vectors
    query_hashes = pd.read_pickle(f"detailed_results/{model}/{dataset}_query_hashes.pkl")
    query_df = pd.DataFrame(query_hashes, columns=["query_text", "hash"])
    query_df["vectors"] = query_df["query_text"].apply(lambda x: cache_data.get_vector(x))

    # Convert all corpus vectors to a hash map for O(1) lookup
    print("Building corpus hash map...")
    corpus_hash_map = {
        all_vectors[i].astype(np.float16).tobytes(): i
        for i in range(all_vectors.shape[0])
    }

    # Find matching indices
    query_indices_in_cache = []
    for i, q in query_df.iterrows():
        if i % 1000 == 0 and i > 0:
            print(f"Processed {i} query vectors.")

        embedding_q = np.array(q["vectors"], dtype=np.float16)
        key = embedding_q.tobytes()

        match_idx = corpus_hash_map.get(key, None)
        if match_idx is not None:
            query_indices_in_cache.append(match_idx)
        else:
            print(f"Embedding vector for index {i} not found in corpus.")

    # Build final outputs
    query_vectors = np.stack(query_df["vectors"].values)
    corpus_vectors = np.delete(all_vectors, query_indices_in_cache, axis=0)

    return query_vectors, corpus_vectors

    

def launch_new_job_on_failure(name):
    sanitized_name = name.replace("/", "")
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    job_name = f"{sanitized_name}_{current_time}"
    print("Sanitized name: ", job_name)

    sbatch_script = textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --output=results_logs/analysis_{job_name}.txt
        #SBATCH --cpus-per-task=4
        #SBATCH --mem-per-cpu=64GB

        . ../.phd/bin/activate
        python compute_results.py --model {name}
    """)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".sh", mode='w') as temp_file:
        temp_file.write(sbatch_script)
        temp_file_path = temp_file.name

    subprocess.run(["sbatch", temp_file_path])
    os.remove(temp_file_path)

def get_retrievability_data(model, dataset):
    data = pd.read_pickle(f"detailed_results/{model}/{dataset}_retrievability_scores.pkl")
    # Query ids for selecting only corpus
    query_ids = data["query_ids"]
    # drop it from data dict
    del data["query_ids"]

    query_ids = data.keys()
    retrievability_scores = [data[d]["retrievability"] for d in data]
    hashes = [data[d]["hash"] for d in data]
    document_text = [data[d]["document_text"] for d in data]
    n_words = [len(data[d]["document_text"].split()) for d in data]

    return pd.DataFrame({
        "query_id": query_ids,
        "retrievability_score": retrievability_scores,
        "hash": hashes,
        "document_text": document_text,
        "n_words": n_words
    })


def aggregate_results(directory, filename, filetype, wildcard="*", filter_str=""):
    json_files = glob.glob(os.path.join(directory, 
                                        f"**/{filename}{wildcard}{filetype}"), 
                                        recursive=True)
    if filter_str != "":
        json_files = [f for f in json_files if filter_str not in f]
    dfs = []
    for results_file in json_files:
        with open(results_file, "r") as f:  
            if filetype == ".csv":
                dfs.append(pd.read_csv(f))
            elif filetype == ".pkl":
                dfs.append(pd.DataFrame(pd.read_pickle(f)))
    df = pd.concat(dfs, ignore_index=True)
    print("Loaded dataframe with shape: ", df.shape)
    return df

def forward_fill_sample_size(df, group_cols=['model', 'dataset']):
    # Forward-fill and select the maximum sample sizes
    value_cols = df.columns
    df_sorted = df.sort_values(by=group_cols + ['sample_size'])
    df_filled = df_sorted.groupby(group_cols)[value_cols].ffill()
    df_final_sorted = df_sorted.drop(columns=value_cols).join(df_filled)
    df = df_final_sorted.groupby(group_cols).last().reset_index()
    print("Forward filled sample sizes. New shape: ", df.shape)
    return df


def merge_with_model_details(df, mdfile="model_details.yaml", on="model"):
    # Load the YAML file
    with open(mdfile, "r") as file:
        model_details = yaml.safe_load(file)

    model_details = pd.DataFrame.from_dict(model_details, orient='index')\
        .reset_index().rename(columns={'index': 'model'})
    
    return df.merge(model_details, on=on)


def add_transformed_columns(df, cols_to_normalize, normalization_method="standard"):
    """ Transforms specified columns in the DataFrame within each dataset group.
    """
    for col in cols_to_normalize:
        # Log transformed
        df.loc[:, f"log_{col}"] = np.log(df[col])

        # Scaled within dataset
        for dataset, group in df.groupby('dataset'):
            if normalization_method == "standard":
                scaler = StandardScaler()
                df.loc[group.index, f'{col}_normalized'] = scaler.fit_transform(group[col].values.reshape(-1, 1)).flatten()
            elif normalization_method == "robust":
                scaler = RobustScaler()
                df.loc[group.index, f'{col}_normalized'] = scaler.fit_transform(group[col].values.reshape(-1, 1)).flatten()
            elif normalization_method == "max":
                max_val = group[col].max()
                df.loc[group.index, f'{col}_normalized'] = group[col] / max_val if max_val != 0 else 0


