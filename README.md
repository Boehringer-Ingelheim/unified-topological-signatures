# Unified Topological Signatures

![Visual Abstract](images/visual_abstract.png)



## Installation

### Dependencies and Modified MTEB version
We have created a MTEB fork with modifications to the Retrieval-Evaluation module. Make sure to install mteb from this Github repository: [Anonymized](https://github.com/) by running `pip install https://github.com/<anonymized-user>/mteb/archive/main.zip`.

In detail, we have modified `mteb/evaluation/evaluators/RetrievalEvaluator.py`such that:
- We store query and document embeddings related to retrievability
- We store a query's neighborhood documents (as a result of the dense retrieval computation)
- We store all hashes to be able to track an embedding back to its text and vice verca
- We compute document retrievability @ 100 for each query (i.e. we use the retrieval results to track the existence of each document)

Run `pip install -r requirements.txt` to install the remaining dependencies. For magnipy

### .env file
Create a .env file with the following entries:
```
CACHE_PATH=""
```
It will be loaded automatically.


## Computing MTEB evaluations

### Running evaluations
The datasets used for the retrieval evaluation are configured in `config/eval.py`. Models are specified by their huggingface name consisting of institution/model_name. We provide a SLURM script to run all results in parallel (one job per model). To do so, execute `sh scripts/mteb.sh` in your console and all jobs will be scheduled. 

If you want to compute the results for a single model, you can also simply call the python script:
```
python compute_results.py --model <some-model-name>
``` 

### Adding custom models
In `utils/models.py` we provide a few examples for adding custom models from cloud platforms (e.g. self-hosted platforms, Databricks, google...). We use a prefix in the name to route the models to their corresponding model. We have also added basic error handling examples, such as context window overflow or hitting request rate limits. 

In addition, we provide a custom random model (prefix _random_), which allows to generate synthetic embeddings for any kind of analysis. Note that MTEB uses sentence-transformers as default choice when providing a model name.

### Embeddings cache
We use the class `CachedEmbeddingWrapper` to cache all computed embeddings at the cache location specified in the .env file. Running all experiments for all models and datasets results in a **lot of data (3.4 TB)**. To reduce the required disk space, you might want to exclude larger datasets (such as MTEB, ~8M embeddings) or use models with smaller embedding dimensions.


## Computing topological signatures

### Running computations
The file `config/metrics.py` specifies:
- Which topological measures to compute
- Which retrieval metrics to compute
- What arguments (e.g. max sample size) and seeds will be used
- Which distance functions to use
- How the signature vectors are constructed

We provide a SLURM script to run all metric computations in parallel (one job per model). To do so, execute `sh scripts/metrics.sh` in your console and all jobs will be scheduled. 

If you want to compute the metrics for a single model, you can also simply call the python script:
```
python compute_metrics.py --model <some-model-name>
``` 

## Analysis notebooks


- Compute requirements (128 GB), which GPU?

- Density of query / document space is proxy --> Citation (standard for retr. bias)
