#!/bin/bash
#SBATCH --job-name=gams_attention
#SBATCH --account=povejmo
#SBATCH --output=logs/gams_attention_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --partition=frida

# Usage: sbatch run_visualization.sbatch [model] [max_new_tokens]
#   model: HF locator or /models/<name> (= /shared/workspace/povejmo/models/hf_models/<name>)
model=${1:-cjvt/GaMS3-12B-Instruct}
max_new_tokens=${2:-1024}

export HF_HOME=/shared/workspace/povejmo/huggingface_cache
export HF_TOKEN=$(<$HOME/hf_token.txt)

CONTAINER=/shared/workspace/povejmo/containers/vllm_0.33.0.sqsh   # transformers 5.x
MODELS_DIR=/shared/workspace/povejmo/models/hf_models
WORK_DIR=/shared/workspace/povejmo/scripts/visualize_gams_attention
OUTPUT_DIR=${WORK_DIR}/outputs
model_name=$(basename "$model")

echo "Model: $model"

srun \
    --container-mounts ${MODELS_DIR}:/models \
    --container-image $CONTAINER \
    --container-workdir $WORK_DIR \
    bash -c \
    "nvidia-smi && \
    python3 extract_attention.py \
        --model_path=$model \
        --output_dir=$OUTPUT_DIR \
        --max_new_tokens=$max_new_tokens && \
    python3 build_visualization.py --input_dir=$OUTPUT_DIR/$model_name"
