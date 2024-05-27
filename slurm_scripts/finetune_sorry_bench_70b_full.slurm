#!/bin/bash
#SBATCH --job-name=fine-tune-sorry-bench    # create a short name for your job
#SBATCH --nodes=4                # node count
#SBATCH --ntasks=4               # total number of tasks across all nodes
#SBATCH --cpus-per-task=32        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=1000G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:4            # number of gpus per node
#SBATCH --time=14:59:59          # total run time limit (HH:MM:SS)
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --constraint=gpu80
#SBATCH --mail-type=begin
#SBATCH --mail-user=tx0973@princeton.edu


conda init bash ;\
conda activate pytorch-nightly ;\

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
# Enable for A100
# export FI_PROVIDER="efa"

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
# debugging flags (optional)
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1
# export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=0

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME="eno1"
# export FI_EFA_USE_DEVICE_RDMA=1


conda activate pytorch-nightly

SAVE_NAME="ft-llama-3-70b-instruct-sorry-bench-bs=16-lr=5e-6"


# srun torchrun --nproc_per_node 4 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 finetuning.py \
srun torchrun --nnodes 4 --nproc_per_node 4 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 --rdzv-conf 'timeout'=3600 finetuning.py \
--batch_size_training 1 --lr 5e-6 \
--gradient_accumulation_steps 16 --weight_decay 0 \
--num_epochs 3 \
--dataset harmfulness_eval_dataset \
--enable_fsdp --low_cpu_fsdp \
--model_name ckpts/Meta-Llama-3-70B-Instruct --pure_bf16 \
--dist_checkpoint_root_folder finetuned_models/ \
--dist_checkpoint_folder $SAVE_NAME \
--run_validation False --save_every_epoch True ;\

python inference/checkpoint_converter_fsdp_hf.py -fsdp_checkpoint_path "finetuned_models/${SAVE_NAME}-epoch=3-ckpts/Meta-Llama-3-70B-Instruct" -consolidated_model_path "finetuned_models/${SAVE_NAME}/" -HF_model_path_or_name "ckpts/Meta-Llama-3-70B-Instruct" ;\





# SAVE_NAME="ft-llama-2-70b-chat-sorry-bench-bs=16-lr=1e-5"


# srun torchrun --nnodes 2 --nproc_per_node 4 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 --rdzv-conf 'timeout'=3600 finetuning.py \
# --batch_size_training 1 --lr 1e-5 \
# --gradient_accumulation_steps 16 --weight_decay 0 \
# --num_epochs 3 \
# --dataset harmfulness_eval_dataset \
# --enable_fsdp --low_cpu_fsdp \
# --model_name ckpts/Llama-2-70B-Chat-fp16 --pure_bf16 \
# --dist_checkpoint_root_folder finetuned_models/ \
# --dist_checkpoint_folder $SAVE_NAME \
# --run_validation False --save_every_epoch True ;\

# python inference/checkpoint_converter_fsdp_hf.py -fsdp_checkpoint_path "finetuned_models/${SAVE_NAME}-ckpts/Llama-2-70B-Chat-fp16" -consolidated_model_path "finetuned_models/${SAVE_NAME}/" -HF_model_path_or_name "ckpts/Llama-2-70B-Chat-fp16" ;\




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



# SAVE_NAME="ft-llama-2-13b-chat-sorry-bench-bs=16-lr=2e-5"

# torchrun --nnodes 1 --nproc_per_node 4 --rdzv-endpoint=localhost:23424 --rdzv-backend=c10d finetuning.py \
# --batch_size_training 16 --lr 2e-5 \
# --gradient_accumulation_steps 1 --weight_decay 0 \
# --num_epochs 3 \
# --dataset harmfulness_eval_dataset \
# --enable_fsdp \
# --model_name ckpts/Llama-2-13B-Chat-fp16 --pure_bf16 \
# --dist_checkpoint_root_folder finetuned_models/ \
# --dist_checkpoint_folder $SAVE_NAME \
# --run_validation False --save_every_epoch False ;\

# python inference/checkpoint_converter_fsdp_hf.py -fsdp_checkpoint_path "finetuned_models/${SAVE_NAME}-ckpts/Llama-2-13B-Chat-fp16" -consolidated_model_path "finetuned_models/${SAVE_NAME}/" -HF_model_path_or_name "ckpts/Llama-2-13B-Chat-fp16" ;\