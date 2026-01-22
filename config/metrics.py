from metrics  import *
     
METRICS_CONFIG = {
    "retrieval_scores": ["ndcg_at_5", "ndcg_at_20", "ndcg_at_100",
                         "recall_at_5", "recall_at_20", "recall_at_100",
                         "map_at_5", "map_at_20", "map_at_100",
                         "main_score"],
    "random_seed": 1234, # 1234, 2025, 123
    "distance_metrics": ["euclidean", "cosine"],
    "sample_sizes": [1000, 2000, 3000, 4000, 5000, 10000, 20000, 50000, 100000, 500000],
    "metrics": {
        "uniformity": {
            "compute_func": compute_uniform_loss,
            "requires_distance": False,
            "max_sample_size": 20000,
            "kwargs": {}
        },
        "euler_characteristic": {
            "compute_func": compute_euler_characteristic,
            "requires_distance": False,
            "max_sample_size": 5000,
            "kwargs": {}
        },
        "effective_rank": {
            "compute_func": compute_effective_rank,
            "requires_distance": False,
            "max_sample_size": 100000,
            "kwargs": {}
        },
        "avg_pair_sim": {
            "compute_func": compute_avg_pairwise_sim,
            "requires_distance": True,
            "max_sample_size": 50000,
            "kwargs": {}
        },
        "mag_area": {
            "compute_func": compute_magnitude_area,
            "requires_distance": False,
            "max_sample_size": 5000,
            "kwargs": {}
        },
        "mag_dim": {
            "compute_func": compute_magnitude_dimension,
            "requires_distance": False,
            "max_sample_size": 5000,
            "kwargs": {}
        },
        "ph_dim": {
            "compute_func": compute_persistent_homology_dim,
            "requires_distance": True,
            "max_sample_size": 5000,
            "kwargs": {"min_samples": 500, 
                       "max_samples": 5000, 
                       "stepsize": 500, "h_dim": 0, 
                       "alpha": 1, "seed": 24}
        },
        "ph_entr": {
            "compute_func": compute_ph_entropy,
            "requires_distance": True,
            "max_sample_size": 20000,
            "kwargs": {}
        },
        "spread": {
            "compute_func": compute_spread,
            "requires_distance": True,
            "max_sample_size": 10000,
            "kwargs": {}
        },
        "iso_score": {
            "compute_func": compute_iso_score,
            "requires_distance": False,
            "max_sample_size": 100000,
            "kwargs": {}
        },
        "pca_dim": {
            "compute_func": compute_global_pca_dimension,
            "requires_distance": False,
            "max_sample_size": 50000,
            "kwargs": {}
        },
        "twonn_dim": {
            "compute_func": compute_twonn_dim,
            "requires_distance": False,
            "max_sample_size": 50000,
            "kwargs": {}
        },
        "silhouette": {
            "compute_func": compute_optimal_kmeans_silhouette,
            "requires_distance": False,
            "max_sample_size": 20000,
            "kwargs": {}
        },
        "vendi_score": {
            "compute_func": compute_vendi_score,
            "requires_distance": True,
            "max_sample_size": 20000,
            "kwargs": {}
        },
        "lid": {
            "compute_func": compute_local_pca_dimension,
            "requires_distance": False,
            "is_local": True,
            "max_sample_size": 50000,
            "kwargs": {}
        },
        "lifetime_mean": {
            "compute_func": compute_lifetime_mean,
            "requires_distance": True,
            "is_local": False,
            "max_sample_size": 5000,
            "kwargs": {}
        },
        "midlife_mean": {
            "compute_func": compute_midlife_mean,
            "requires_distance": True,
            "is_local": False,
            "max_sample_size": 5000,
            "kwargs": {}
        },
    }
}

BASE_COLS = [
    'mag_area',
    'mag_dim',
    'pca_dim',
    'twonn_dim',
    'silhouette',
    'iso_score',
    "uniformity",
    "euler_characteristic",
    "effective_rank"
]

DISTANCE_MEASURES = ["cosine"]  # "euclidean",
SIGNATURE_COLUMNS = BASE_COLS + [
    f'{prefix}_{distance_measure}'
    for distance_measure in DISTANCE_MEASURES
    for prefix in [
        'avg_pair_sim', 
        'lifetime_mean',
        'midlife_mean', 
        'vendi_score',
        'ph_dim',
        'ph_entr',
        'spread'
    ]
]

NAME_MAPPING = {'mag_area': "Magnitude Area",
                'mag_dim': "Magnitude Dimension",
                'pca_dim': "PCA Dimension",
                'twonn_dim': "TwoNN Dimension",
                'silhouette': "Silhouette Score",
                'iso_score': "IsoScore",
                'uniformity': "Uniformity",
                'euler_characteristic': "Euler Characteristic",
                'effective_rank': "Effective Rank",
                'avg_pair_sim_euclidean': "Avg. Pairwise Sim. (Euclidean)",
                'lifetime_mean_euclidean': "Lifetime Mean (Euclidean)",	
                'midlife_mean_euclidean': "Midlife Mean (Euclidean)",
                'vendi_score_euclidean': "Vendi Score (Euclidean)",
                'ph_dim_euclidean': "PH Dimension (Euclidean)",
                'ph_entr_euclidean': "PH Entropy (Euclidean)",
                'spread_euclidean': "Spread (Euclidean)",
                'avg_pair_sim_cosine': "Avg. Pairwise Sim. (Cos)",
                'lifetime_mean_cosine': "Lifetime Mean (Cos)",
                'midlife_mean_cosine': "Midlife Mean (Cos)",
                'ph_dim_cosine': "PH Dimension (Cos)",
                'ph_entr_cosine': "PH Entropy (Cos)",
                'vendi_score_cosine': "Vendi Score (Cos)",
                'spread_cosine': "Spread (Cos)",
                "embedding_dimension": "Embedding dimension",
                "model_size": "Model size",
                "layers": "# Layers"}

