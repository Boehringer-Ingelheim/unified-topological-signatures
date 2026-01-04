import os
import json
import requests
import numpy as np
from google import genai
from tqdm import tqdm
from time import sleep
from mteb.model_meta import ModelMeta
from httpx import RemoteProtocolError
from google.genai.types import EmbedContentConfig


class CustomModel:
    def __init__(self, model_name, normalize_embeddings=False) -> None:
        self.normalize_embeddings = normalize_embeddings
        self.model_name = self.model = model_name[len("<your-platform>/"):]

        if model_name == "some_model":
            self.dims = 3072
            self.batch_size = 1000

        # You can add more model-specific configurations here
        self.mteb_model_meta = ModelMeta(name=model_name,
                                         revision=None,
                                         release_date=None,
                                         languages=None,
                                         n_parameters=None,
                                         memory_usage_mb=None,
                                         max_tokens=None,
                                         embed_dim=None,
                                         license=None,
                                         open_weights=None,
                                         public_training_code=None,
                                         framework=[],
                                         use_instructions=None,
                                         training_datasets=None,
                                         public_training_data=None,
                                         similarity_fn_name=None)

        # TODO: Initialize your embedding model here
        # e.g. a callable endpoint with token handling
        self.embedding_model = None  


    def query(self, sentence):
        try:
            vectors = self.embedding_model.embed_documents(sentence)
            return vectors
        except Exception as e:
            # You might have to account for context length limits here
            print(f"Batch encoding failed: {e}. ")
            return vectors
        

    def encode(self, sentences: list[str], batch_size: None = None, **kwargs) -> np.ndarray:
        """
        Encodes the given sentences using the embedding model.

        Args:
            sentences: List of sentences to encode.

        Returns:
            A NumPy array of encoded sentence embeddings.
        """
        batch_size = self.batch_size
        sentence_batches = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]
        vectors = []

        for batch in tqdm(sentence_batches, desc="Encoding batches"):
            batch_clean = []
            for sentence in batch:
                if sentence == " ":
                    batch_clean.append(sentence.replace(" ", "empty"))
                else:
                    batch_clean.append(sentence)
            vectors.extend(self.query(batch_clean))

        embeddings = np.array(vectors, dtype=np.float16)

        if self.normalize_embeddings:
            norm = np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
            norm = np.maximum(norm, 1e-12)
            embeddings = embeddings / norm

        return embeddings
    

class CustomDatabricksModel():
    def __init__(self, model_name="databricks/<model-name>") -> None:
        self.model = model_name[len("databricks/"):]
        self.endpoint_url = f"https://<insert-host-url-here>/serving-endpoints/{self.model}/invocations"
        self.api_token = os.environ.get("DATABRICKS_TOKEN")

        # Metadata for caching
        self.mteb_model_meta = ModelMeta(name=model_name,
                                         revision=None,
                                         release_date=None,
                                         languages=None,
                                         n_parameters=None,
                                         memory_usage_mb=None,
                                         max_tokens=None,
                                         embed_dim=None,
                                         license=None,
                                         open_weights=None,
                                         public_training_code=None,
                                         framework=[],
                                         use_instructions=None,
                                         training_datasets=None,
                                         public_training_data=None,
                                         similarity_fn_name=None)
    
    def encode(
        self,
        sentences: list[str],
        batch_size: None,
        **kwargs,
    ) -> np.ndarray:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded sentences.
        """

        # Send the request
        response = requests.post(
            self.endpoint_url,
            headers={"Authorization": f"Bearer {self.api_token}", "Content-Type": "application/json"},
            data=json.dumps(sentences)
        )
        result = response.json()
        return np.array(result["data"]).astype(np.float32)


class CustomGoogleModel():
    def __init__(self, model_name):
        self.model_name = model_name
        self.init_client()

        if "gemini-embedding-001" in model_name:
            self.config = EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=3072,
                    )
        elif "text-embedding-004" in model_name:
            self.config = EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=768
                    ) 

    def init_client(self):
        self.client = genai.Client()

    def encode(
        self,
        sentences: list[str],
        batch_size: None,
        **kwargs,
    ) -> np.ndarray:
        vectors = []
        for sentence in tqdm(sentences):
            try:
                response = self.client.models.embed_content(
                    model=self.model_name.replace("google/", ""),
                    contents=sentence,
                    config=self.config,
                )
            except RemoteProtocolError:
                print("Remote protocol error. Retrying...")
                self.init_client()
                response = self.client.models.embed_content(
                    model=self.model_name.replace("google/", ""),
                    contents=sentence,
                    config=self.config,
                )
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e):
                    sleep(60)
                    response = self.client.models.embed_content(
                        model=self.model_name.replace("google/", ""),
                        contents=sentence,
                        config=self.config,
                    )
                else:
                    raise Exception(e)
            vectors.append(response.embeddings[0].values)
        embeddings = np.array(vectors).astype(np.float32)
        return embeddings


class CustomRandomModel():
    def __init__(self, model_name):
        self.model_name = model_name
        self.dim = int(model_name.replace("random/", ""))

    def encode(
        self,
        sentences: list[str],
        batch_size: None,
        **kwargs,
    ) -> np.ndarray:
        embeddings = np.random.rand(len(sentences), self.dim).astype(np.float16)
        return embeddings
