#!/bin/bash
#SBATCH --job-name=fine-tune-sorry-bench    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=32        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=1000G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:4            # number of gpus per node
#SBATCH --time=0:59:59          # total run time limit (HH:MM:SS)
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --constraint=gpu80


# module purge
# module load anaconda3/2022.5
# conda activate llama2



# SAVE_NAME="ft-llama-3-8b-instruct-sorry-bench-bs=16-lr=2e-5"

# torchrun --nnodes 1 --nproc_per_node 2 --rdzv-endpoint=localhost:23427 --rdzv-backend=c10d finetuning.py \
# --batch_size_training 16 --lr 2e-5 \
# --gradient_accumulation_steps 1 --weight_decay 0 \
# --num_epochs 3 \
# --dataset harmfulness_eval_dataset \
# --enable_fsdp \
# --model_name ckpts/Meta-Llama-3-8B-Instruct --pure_bf16 \
# --dist_checkpoint_root_folder finetuned_models/ \
# --dist_checkpoint_folder $SAVE_NAME \
# --run_validation False --save_every_epoch False ;\

# python inference/checkpoint_converter_fsdp_hf.py -fsdp_checkpoint_path "finetuned_models/${SAVE_NAME}-ckpts/Meta-Llama-3-8B-Instruct" -consolidated_model_path "finetuned_models/${SAVE_NAME}/" -HF_model_path_or_name "ckpts/Meta-Llama-3-8B-Instruct" ;\


# SAVE_NAME="ft-gemma-7b-it-sorry-bench-bs=16-lr=5e-6"

# torchrun --nnodes 1 --nproc_per_node 2 --rdzv-endpoint=localhost:23426 --rdzv-backend=c10d finetuning.py \
# --batch_size_training 4 --lr 5e-6 \
# --gradient_accumulation_steps 4 --weight_decay 0 \
# --num_epochs 3 \
# --dataset harmfulness_eval_dataset \
# --enable_fsdp \
# --model_name ckpts/gemma-7b-it --pure_bf16 \
# --dist_checkpoint_root_folder finetuned_models/ \
# --dist_checkpoint_folder $SAVE_NAME \
# --run_validation False --save_every_epoch False ;\

# python inference/checkpoint_converter_fsdp_hf.py -fsdp_checkpoint_path "finetuned_models/${SAVE_NAME}-ckpts/gemma-7b-it" -consolidated_model_path "finetuned_models/${SAVE_NAME}/" -HF_model_path_or_name "ckpts/gemma-7b-it" ;\


# SAVE_NAME="ft-mistral-7b-instruct-v0.2-sorry-bench-bs=16-lr=5e-6"

# torchrun --nnodes 1 --nproc_per_node 2 --rdzv-endpoint=localhost:23425 --rdzv-backend=c10d finetuning.py \
# --batch_size_training 8 --lr 5e-6 \
# --gradient_accumulation_steps 2 --weight_decay 0 \
# --num_epochs 3 \
# --dataset harmfulness_eval_dataset \
# --enable_fsdp \
# --model_name ckpts/Mistral-7B-Instruct-v0.2 --pure_bf16 \
# --dist_checkpoint_root_folder finetuned_models/ \
# --dist_checkpoint_folder $SAVE_NAME \
# --run_validation False --save_every_epoch False ;\

# python inference/checkpoint_converter_fsdp_hf.py -fsdp_checkpoint_path "finetuned_models/${SAVE_NAME}-ckpts/Mistral-7B-Instruct-v0.2" -consolidated_model_path "finetuned_models/${SAVE_NAME}/" -HF_model_path_or_name "ckpts/Mistral-7B-Instruct-v0.2" ;\



# SAVE_NAME="ft-llama-2-7b-chat-sorry-bench-bs=16-lr=2e-5"

# torchrun --nnodes 1 --nproc_per_node 2 --rdzv-endpoint=localhost:23424 --rdzv-backend=c10d finetuning.py \
# --batch_size_training 16 --lr 2e-5 \
# --gradient_accumulation_steps 1 --weight_decay 0 \
# --num_epochs 3 \
# --dataset harmfulness_eval_dataset \
# --enable_fsdp \
# --model_name ckpts/Llama-2-7b-chat-fp16 --pure_bf16 \
# --dist_checkpoint_root_folder finetuned_models/ \
# --dist_checkpoint_folder $SAVE_NAME \
# --run_validation False --save_every_epoch False ;\

# python inference/checkpoint_converter_fsdp_hf.py -fsdp_checkpoint_path "finetuned_models/${SAVE_NAME}-ckpts/Llama-2-7b-chat-fp16" -consolidated_model_path "finetuned_models/${SAVE_NAME}/" -HF_model_path_or_name "ckpts/Llama-2-7b-chat-fp16" ;\



SAVE_NAME="ft-llama-2-13b-chat-sorry-bench-bs=16-lr=1e-5" ;\

torchrun --nnodes 1 --nproc_per_node 4 --rdzv-endpoint=localhost:23424 --rdzv-backend=c10d finetuning.py \
--batch_size_training 16 --lr 1e-5 \
--gradient_accumulation_steps 1 --weight_decay 0 \
--num_epochs 3 \
--dataset harmfulness_eval_dataset \
--enable_fsdp \
--model_name ckpts/Llama-2-13B-Chat-fp16 --pure_bf16 \
--dist_checkpoint_root_folder finetuned_models/ \
--dist_checkpoint_folder $SAVE_NAME \
--run_validation False --save_every_epoch False ;\

python inference/checkpoint_converter_fsdp_hf.py -fsdp_checkpoint_path "finetuned_models/${SAVE_NAME}-ckpts/Llama-2-13B-Chat-fp16" -consolidated_model_path "finetuned_models/${SAVE_NAME}/" -HF_model_path_or_name "ckpts/Llama-2-13B-Chat-fp16" ;\



SAVE_NAME="ft-llama-2-13b-chat-sorry-bench-bs=16-lr=5e-6" ;\

torchrun --nnodes 1 --nproc_per_node 4 --rdzv-endpoint=localhost:23424 --rdzv-backend=c10d finetuning.py \
--batch_size_training 16 --lr 5e-6 \
--gradient_accumulation_steps 1 --weight_decay 0 \
--num_epochs 3 \
--dataset harmfulness_eval_dataset \
--enable_fsdp \
--model_name ckpts/Llama-2-13B-Chat-fp16 --pure_bf16 \
--dist_checkpoint_root_folder finetuned_models/ \
--dist_checkpoint_folder $SAVE_NAME \
--run_validation False --save_every_epoch False ;\

python inference/checkpoint_converter_fsdp_hf.py -fsdp_checkpoint_path "finetuned_models/${SAVE_NAME}-ckpts/Llama-2-13B-Chat-fp16" -consolidated_model_path "finetuned_models/${SAVE_NAME}/" -HF_model_path_or_name "ckpts/Llama-2-13B-Chat-fp16" ;\