
import os
import mteb
import logging
import argparse
from mteb import MTEB
from mteb.models.cache_wrapper import CachedEmbeddingWrapper

from utils.models import CustomModel, CustomDatabricksModel, CustomGoogleModel, CustomRandomModel
from config.eval import RETRIEVAL_TAKS
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Dataset args
parser = argparse.ArgumentParser(
    prog='Text Embedding Analysis',
    description='Parallel processing of text embeddings for MTEB tasks.')
parser.add_argument('--model')

args = parser.parse_args()
if not args.model:
    print("No model name provided, launching debug mode with mistral-7b.")
    model_name = "intfloat/e5-mistral-7b-instruct"
else:
    model_name = args.model


# Select model class
if model_name.startswith("apollo"):
    model = CustomModel(model_name=model_name,
                        normalize_embeddings=True)
elif model_name.startswith("google"):
    model = CustomGoogleModel(model_name=model_name)
elif model_name.startswith("databricks"):
    model = CustomDatabricksModel(model_name=model_name)
elif model_name.startswith("random"):
    model = CustomRandomModel(model_name=model_name)
else:
    model = mteb.get_model(model_name)

# Run evaluation
evaluation = MTEB(tasks=RETRIEVAL_TAKS) 
cache_name = model_name.replace("/", "_")
cache_path = os.environ.get("CACHE_PATH")
model_with_cached_emb = CachedEmbeddingWrapper(
    model, cache_path=f"{cache_path}/cache_{cache_name}"
)

evaluation.run(model_with_cached_emb,
               save_predictions=True,
               overwrite_results=True,
               encode_kwargs={"batch_size": 1,
                              "normalize_embeddings": True,
                              "model_name": model_name},
               output_folder=f"results/{model_name}",
               corpus_chunk_size=1000)

print(f"Completed {model_name}.")
