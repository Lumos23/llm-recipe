python inference/checkpoint_converter_fsdp_hf.py -fsdp_checkpoint_path "finetuned_models/ft-llama-3-70b-instruct-sorry-bench-bs=16-lr=5e-6-epoch=2-ckpts/Meta-Llama-3-70B-Instruct/" -consolidated_model_path "ckpts/finetuned_models/ft-llama-3-70b-instruct-sorry-bench-bs=16-lr=5e-6-epoch=2" -HF_model_path_or_name "ckpts/Meta-Llama-3-70B-Instruct" ;\
python inference/checkpoint_converter_fsdp_hf.py -fsdp_checkpoint_path "finetuned_models/ft-llama-3-70b-instruct-sorry-bench-bs=16-lr=5e-6-epoch=1-ckpts/Meta-Llama-3-70B-Instruct/" -consolidated_model_path "ckpts/finetuned_models/ft-llama-3-70b-instruct-sorry-bench-bs=16-lr=5e-6-epoch=1" -HF_model_path_or_name "ckpts/Meta-Llama-3-70B-Instruct"

# python inference/checkpoint_converter_fsdp_hf.py -fsdp_checkpoint_path "finetuned_models/ft-llama-3-70b-instruct-sorry-bench-bs=16-lr=2e-5-epoch=1-ckpts/Meta-Llama-3-70B-Instruct/" -consolidated_model_path "ckpts/finetuned_models/ft-llama-3-70b-instruct-sorry-bench-bs=16-lr=2e-5-epoch=1" -HF_model_path_or_name "ckpts/Meta-Llama-3-70B-Instruct"
# python inference/checkpoint_converter_fsdp_hf.py -fsdp_checkpoint_path "finetuned_models/ft-llama-3-70b-instruct-sorry-bench-bs=16-lr=2e-5-epoch=3-ckpts/Meta-Llama-3-70B-Instruct/" -consolidated_model_path "ckpts/finetuned_models/ft-llama-3-70b-instruct-sorry-bench-bs=16-lr=2e-5-epoch=3" -HF_model_path_or_name "ckpts/Meta-Llama-3-70B-Instruct"


# python inference/checkpoint_converter_fsdp_hf.py -fsdp_checkpoint_path "finetuned_models/ft-llama-3-70b-instruct-sorry-bench-bs=16-lr=1e-5-epoch=1-ckpts/Meta-Llama-3-70B-Instruct/" -consolidated_model_path "ckpts/finetuned_models/ft-llama-3-70b-instruct-sorry-bench-bs=16-lr=1e-5-epoch=1" -HF_model_path_or_name "ckpts/Meta-Llama-3-70B-Instruct" ;\



# python inference/checkpoint_converter_fsdp_hf.py -fsdp_checkpoint_path "finetuned_models/ft-llama-3-70b-instruct-sorry-bench-bs=16-lr=8e-6-epoch=3-ckpts/Meta-Llama-3-70B-Instruct/" -consolidated_model_path "finetuned_models/ft-llama-3-70b-instruct-sorry-bench-bs=16-lr=8e-6" -HF_model_path_or_name "ckpts/Meta-Llama-3-70B-Instruct"

# python inference/checkpoint_converter_fsdp_hf.py -fsdp_checkpoint_path "finetuned_models/ft-llama-3-70b-instruct-sorry-bench-bs=16-lr=8e-6-epoch=2-ckpts/Meta-Llama-3-70B-Instruct/" -consolidated_model_path "finetuned_models/ft-llama-3-70b-instruct-sorry-bench-bs=16-lr=8e-6-epoch=2" -HF_model_path_or_name "ckpts/Meta-Llama-3-70B-Instruct" ;\
# mv finetuned_models/ft-llama-3-70b-instruct-sorry-bench-bs=16-lr=8e-6-epoch=2 ckpts/finetuned_models/ ;\
# python inference/checkpoint_converter_fsdp_hf.py -fsdp_checkpoint_path "finetuned_models/ft-llama-3-70b-instruct-sorry-bench-bs=16-lr=8e-6-epoch=1-ckpts/Meta-Llama-3-70B-Instruct/" -consolidated_model_path "finetuned_models/ft-llama-3-70b-instruct-sorry-bench-bs=16-lr=8e-6-epoch=1" -HF_model_path_or_name "ckpts/Meta-Llama-3-70B-Instruct" ;\
# mv finetuned_models/ft-llama-3-70b-instruct-sorry-bench-bs=16-lr=8e-6-epoch=1 ckpts/finetuned_models/ ;\