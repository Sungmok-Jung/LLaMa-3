#!/bin/bash
#SBATCH --job-name=pp_test
#SBATCH --partition=LH
#SBATCH --nodelist=b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64      
#SBATCH --gres=gpu:1
#SBATCH --mem=480G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err


CKPT_DIR="/home/n3/sungmok/.llama/checkpoints/Llama3.2-3B"
NUM_GPUS=4
BATCH_SIZE=16


torchrun --nproc_per_node 1 llama3/example_text_completion.py \
    --ckpt_dir $CKPT_DIR \
    --tokenizer_path $CKPT_DIR/tokenizer.model \
    --max_seq_len 512 --max_batch_size 6

echo "Inference completed!"