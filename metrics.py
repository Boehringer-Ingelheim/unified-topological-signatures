import logging
import skdim
import time
import sys
import torch
import numpy as np
import pandas as pd
from ripser import ripser
from IsoScore.IsoScore import *
from vendi_score import vendi
from gtda.diagrams import PersistenceEntropy
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.cluster import DBSCAN, KMeans
from sklearn.covariance import ShrunkCovariance
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

sys.path.append('external')
from istar import istar

sys.path.append('external/magnipy')
from magnipy.magnipy import Magnipy
from magnipy.diversipy import Diversipy


def sample_metric(metric_fun, metric_name, vectors, embedding_model, dataset,
                  step_size=1000, min=1000, max=5000, n_seeds=3, kwargs={}):
    """
    For computationally expensive metrics, this function samples the vectors.
    """
    data = []
    for step in range(min, max + 1, step_size):
        if step > vectors.shape[0]:
            print(f"Step {step} exceeds number of vectors {vectors.shape[0]}. Stopping sampling.")
            break

        for seed in range(n_seeds):
            np.random.seed(seed)
            index = np.random.choice(vectors.shape[0], step, replace=False) 
            vectors_subset = vectors[index]

            # Compute metric      
            start = time.time()     
            metric = metric_fun(vectors_subset, **kwargs)
            elapsed_time = time.time() - start
            print(f"{embedding_model} | {dataset} | Step:", step, "Seed:", seed, f"{metric_name}: {metric:.2f}", f"Elapsed time: {elapsed_time:.2f} seconds")
        
            data.append({
                "step": step,
                metric_name: metric,
                "seed": seed,
                f"elapsed_time_{metric_name}": elapsed_time,
                "model": embedding_model,
                "dataset": dataset                
            })
    return pd.DataFrame(data)


def compute_persistent_homology_dim(X, distance_metric="euclidean",
                                    min_samples=100, max_samples=2000,
                                    stepsize=50, h_dim=0, alpha=1, seed=24):
    """
    Code is inspired by https://github.com/tolgabirdal/PHDimGeneralization
    and https://github.com/CSU-PHdimension/PHdimension/blob/master/code_ripser/SumEdgeLengths.m. 
    Based on https://arxiv.org/pdf/1808.01079.
    """
    print(f'Computing persistent homology fractal dimension for homology dim {h_dim}.')
    # Clip max samples to dataset size
    max_samples = max_samples if X.shape[0] > max_samples else X.shape[0]

    if X.shape[0] < min_samples:
        logging.warn("Number of samples for PH Dim calculation too small.")
        return np.nan#, np.array([]), np.array([]) 

    # Define sampling space
    linspace = range(min_samples, max_samples, stepsize)
    edge_lengths = []
    
    np.random.seed(seed)
    # Randomly sample increasing number of points and generate fractals
    for n in linspace:
        sampled_x = X[np.random.choice(X.shape[0], size=n, replace=False)]
        dgms = ripser(sampled_x, maxdim=h_dim, metric=distance_metric)['dgms']

         # Append sum of weighted diagram lengths (birth-death)
        dgm = dgms[h_dim]
        dgm = dgm[dgm[:, 1] < np.inf]
        edge_lengths.append(np.power((dgm[:, 1] - dgm[:, 0]), alpha).sum())

    edge_lengths = np.array(edge_lengths)
    
    # Estimate slope using linear regression
    logspace = np.log10(np.array(list(linspace)))
    logedges = np.log10(edge_lengths)       
    try:
        coeff = np.polyfit(logspace, logedges, 1)
    except Exception:
        print("Polyfit failed. Returning NaN.")
        return np.nan
    m = coeff[0]
    b = coeff[1]

    return alpha / (1 - m)  #, logspace, logedges


def compute_magnitude_area(X):
    print("Computing magnitude area.")
    div = Diversipy(Xs = [X], names=[""], ref_space=0)
    return div.MagAreas()[0]

def compute_magnitude_dimension(X):
    magnitude = Magnipy(X, name='vecs')
    return magnitude.get_magnitude_dimension()

def compute_magnitude(X):
    magnitude = Magnipy(X, name='vecs')
    return magnitude.get_magnitude()

def compute_twonn_dim(X):
    print("Computing TwoNN intrinsic dimensionality.")
    try:
        tnn = skdim.id.TwoNN(discard_fraction=0.5)
        intrdim = tnn.fit_transform(X)   
    except:
        print("TwoNN intrinsic dimensionality computation failed. Returning NaN.")
        intrdim = np.nan      
    return intrdim 

def compute_iso_score(X):
    print("Computing IsoScore.")
    try:
        return IsoScore(X).item()
    except Exception as e:
        return np.nan

def compute_avg_pairwise_sim(X, distance_metric="cosine"): 
    print("Computing pairwise similarity.")
    # Cosine similarity
    if distance_metric == "cosine":
        cosine_sim = cosine_similarity(X)
        return float(np.mean(cosine_sim))

    # Euclidean similarity (inverse of distance)
    if distance_metric == "euclidean":
        euclidean_dist = euclidean_distances(X)
        euclidean_sim = 1 / (1 + euclidean_dist)
        return float(np.mean(euclidean_sim))

    # Dot-product similarity
    if distance_metric == "dot_product":
        dot_product_sim = np.dot(X, X.T)
        return float(np.mean(dot_product_sim))

def compute_silhouette_score(X, eps=0.5, min_samples=5): 
    try:
        print("Computing slihouette score.")
        # Fillna in X
        X = np.nan_to_num(X)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)

        # If all points are noise or only one cluster is found
        if len(set(labels)) <= 1:
            return np.nan

        return silhouette_score(X, labels)
    except:
        print("Silhouette score computation failed. Returning NaN.")
        return np.nan


# def compute_betti(dgms, idx=0):
#     dgm = dgms[idx]
#     dgm = dgm[dgm[:, 1] < np.inf]
#     return dgm.shape[0]

def compute_ph_entropy(X, idx=0, distance_metric="euclidean"):
    dgms = ripser(X, maxdim=idx, metric=distance_metric)['dgms']
    print("Computing PH entropy on dgms:", dgms[idx].shape[0])
    dgm = dgms[idx]
    dgm = dgm[dgm[:, 1] < np.inf]
    q_pad = np.concatenate([dgm, np.full((dgm.shape[0], 1), idx)], axis=1)
    dgms_giotto = np.expand_dims(q_pad, 0)
    PE = PersistenceEntropy(n_jobs=1)
    ph_entropy = PE.fit_transform(dgms_giotto)
    return ph_entropy[idx][0]


def compute_local_pca_dimension(vectors):
    try:
        return skdim.id.TwoNN().fit_transform_pw(X=vectors,
                                n_neighbors = 100,
                                n_jobs = -1)
    except:
        print("Local PCA dimension computation failed. Returning NaN.")
        return np.nan

def compute_global_pca_dimension(vectors):
    pca_model = skdim.id.lPCA()
    return pca_model.fit_transform(vectors)


def compute_optimal_kmeans_silhouette(X, k_range=[3, 5, 10, 20, 50, 99]): 
    try:
        print("Computing optimal k-means silhouette score.")
        # Fillna in X
        X = np.nan_to_num(X)
        
        best_k = None
        best_score = -1

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X)

            # If all points are noise or only one cluster is found
            if len(set(labels)) <= 1:
                continue

            score = silhouette_score(X, labels)

            if score > best_score:
                best_score = score
                best_k = k
    except:
        print("Optimal k-means silhouette score computation failed. Returning NaN.")
        return np.nan

    return best_score


def compute_vendi_score(vectors, distance_metric="euclidean"):
    logging.info('Computing Vendi score.')
    K = pairwise_distances(vectors, metric=distance_metric)
    return vendi.score_K(K)


def compute_isoscore_star(X):
    logging.info('Computing IsoScore*.')
    isc = istar()
    cov = ShrunkCovariance().fit(X).covariance_
    return isc.isoscore_star(points=X, C0=cov, zeta=0.1, is_eval=True).item()


def compute_lifetime_mean(X, distance_metric="euclidean", idx=0):
    logging.info('Computing lifetime mean.')
    dgms = ripser(X, maxdim=idx, metric=distance_metric)['dgms']
    dgm = dgms[idx]
    dgm = dgm[dgm[:, 1] < np.inf]

    lifetimes = dgm[:, 1] - dgm[:, 0]
    agg = np.mean
    #norm_lifetimes = lifetimes / lifetimes.sum()
    if lifetimes.shape[0] > 0:
        return agg(lifetimes)
    else:
        return np.nan

def compute_midlife_mean(X, distance_metric="euclidean", idx=0):
    logging.info('Computing midlife mean.')
    dgms = ripser(X, maxdim=idx, metric=distance_metric)['dgms']
    dgm = dgms[idx]
    dgm = dgm[dgm[:, 1] < np.inf]

    midlifes = (dgm[:, 1] + dgm[:, 0]) / 2
    agg = np.mean
    #norm_midlifes = midlifes / midlifes.sum()
    if midlifes.shape[0] > 0:
        return agg(midlifes)
    else:
        return np.nan

def gini_coefficient(retrievability_scores, exclude_zeros=False):
    logging.info('Computing Gini coefficient.')
    if exclude_zeros:
        retrievability_scores = np.array(retrievability_scores)
        retrievability_scores = retrievability_scores[retrievability_scores > 0]

    sorted_scores = sorted(retrievability_scores)
    cumulative_scores = np.cumsum(sorted_scores)
    cumulative_docs = np.arange(1, len(sorted_scores) + 1)
    lorenz_curve = cumulative_scores / cumulative_scores[-1]
    lorenz_curve_docs = cumulative_docs / cumulative_docs[-1]
    area_under_lorenz = np.trapz(lorenz_curve, lorenz_curve_docs)
    gini = 1 - 2 * area_under_lorenz
    return gini

def compute_spread(X, distance_metric="euclidean"):
    """
    Computes the spread of a finite metric space.
    """
    logging.info('Computing spread.')
    distance_matrix = pairwise_distances(X, metric=distance_metric)
    exp_matrix = np.exp(-distance_matrix)
    row_sums = np.sum(exp_matrix, axis=1)
    spread = np.sum(1 / row_sums)
    return spread

def compute_uniform_loss(x, t=2):
    logging.info('Computing uniform loss.')
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log().numpy()


def compute_euler_characteristic(x) -> int:
    logging.info('Computing Euler characteristic.')
    diagrams = ripser(x, maxdim=1)['dgms']
    betti_numbers = [len(diag) for diag in diagrams]
    euler_characteristic = sum((-1)**i * b for i, b in enumerate(betti_numbers))
    return euler_characteristic

def compute_effective_rank(X):
    logging.info('Computing effective rank.')
    # https://aclanthology.org/2020.lrec-1.589.pdf
    s = np.linalg.svd(X, compute_uv=False)
    sum_s = np.sum(s)
    p = s / sum_s
    erank = np.exp(-np.sum(p * np.log(p)))
    return erank