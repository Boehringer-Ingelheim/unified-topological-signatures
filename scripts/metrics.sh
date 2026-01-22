#!/bin/bash

models=(
  "google/gemini-embedding-001"
  # "google/text-embedding-004"
  # "BAAI/bge-small-en-v1.5"
  # "BAAI/bge-base-en-v1.5"
  # "BAAI/bge-large-en-v1.5"
  # "thenlper/gte-small"
  # "thenlper/gte-base"
  # "thenlper/gte-large"
  # "mixedbread-ai/mxbai-embed-large-v1"
  # "mixedbread-ai/mxbai-embed-xsmall-v1"
  # "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
  # "Alibaba-NLP/gte-Qwen2-7B-instruct"
  # 'sentence-transformers/gtr-t5-base'
  # 'sentence-transformers/gtr-t5-large'
  # 'sentence-transformers/gtr-t5-xxl'
  # "Qwen/Qwen3-Embedding-0.6B"
  # "Qwen/Qwen3-Embedding-4B"
  # "Qwen/Qwen3-Embedding-8B" 
  # "sentence-transformers/all-MiniLM-L6-v2"
  # "llmrails/ember-v1"  
  # "intfloat/e5-mistral-7b-instruct"
  # "intfloat/multilingual-e5-large-instruct"
  # "dunzhang/stella_en_1.5B_v5"
  # "Salesforce/SFR-Embedding-Mistral"
  # "Snowflake/snowflake-arctic-embed-m-v1.5"
  # "sentence-transformers/msmarco-roberta-base-ance-firstp"
  # "sentence-transformers/msmarco-distilbert-base-tas-b"
)


for name in "${models[@]}"; do
  sbatch_script=$(mktemp)
  cat <<EOT > $sbatch_script
#!/bin/bash
#SBATCH --job-name=metrics_$name
#SBATCH --output=metrics_logs/metrics_$name.txt
#SBATCH --time=30-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=128G

export PYTHONUNBUFFERED=1
python -u compute_metrics.py --model $name
EOT
  sbatch $sbatch_script
  rm $sbatch_script
done

 