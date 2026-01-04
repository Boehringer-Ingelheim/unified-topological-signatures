import numpy as np
from utils import load_embeddings, process_embeddings
from cka import CKA
from config import RETRIEVAL_TAKS, MODELS


np_cka = CKA()
max_vectors = 10000

all_tasks_results = []
for task in RETRIEVAL_TAKS:
    cka_results = []
    random_index = None

    for model_1 in MODELS:
        cached_name = f"cache_{model_1}".replace("/", "_")
        try:
            data, results = load_embeddings(cached_name, task)
            vectors_1 = process_embeddings(data)

             # Sample index
            if random_index is None:
                n_vectors = min(max_vectors, vectors_1.shape[0])
                print(f"Sampling indices for {task} with {n_vectors} vectors")
                random_index = np.random.choice(vectors_1.shape[0], n_vectors, replace=False)
            
            vectors_1 = vectors_1.take(random_index, axis=0)
        except:
            vectors_1 = None

       
        cka_results_row = []
        for model_2 in MODELS:
            try:
                cached_name = f"cache_{model_2}".replace("/", "_")
                data, results = load_embeddings(cached_name, task)
                vectors_2 = process_embeddings(data)
                vectors_2 = vectors_2.take(random_index, axis=0)
            except Exception as e:
                vectors_2 = None

            # Compute CKA
            if vectors_1 is None or vectors_2 is None:
                result = np.nan
            else:
                result = np_cka.kernel_CKA(vectors_1, vectors_2)
            cka_results_row.append(result)
            print(f"CKA between {model_1} and {model_2} on {task}: {result}")

        cka_results.append(cka_results_row)
    all_tasks_results.append(cka_results)

np.save('cka_results.npy', np.stack(all_tasks_results))