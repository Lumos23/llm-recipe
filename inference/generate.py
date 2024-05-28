import torch
from transformers import pipeline
import json
import warnings
import pandas as pd
import csv
import sys
from tqdm import tqdm

#model_name = "ft-llama-3-8b-instruct-7breps100-bs=20-lr=5e-5"
if len(sys.argv) > 1:
    model_name = sys.argv[1]
model_id = "/scratch/gpfs/lh2046/tinghao/finetuned_models/"+model_name

# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_name = "Meta-Llama-3-8B-Instruct"
questions_file = "/scratch/gpfs/lh2046/tinghao/inference/harmful_behaviors.csv"
output_file = f"/scratch/gpfs/lh2046/tinghao/inference_output/"+model_name+".jsonl"




# read the input csv file
def question_read(text_file):
    dataset = []
    file = open(text_file, "r")
    data = list(csv.reader(file, delimiter=",")) # delimiter is "?" for gsm8k
    file.close()
    num = len(data)
    for i in range(num):
        dataset.append(data[i][0])
    return dataset

questions_list = question_read(questions_file)

with open('chat_templates.json', 'r') as f:
    chat_templates = json.load(f)

if "llama-3" in model_id.lower():
    template = chat_templates["llama3"]
elif "mistral-7b-base" in model_id.lower():
    template = chat_templates["mistral-base"]
elif "mistral-7b-instruct" in model_id.lower():
    template = chat_templates["mistral-instruct"]
else:
    warnings.warn("No template set for the given model_id.")

generator = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",

)
generator.tokenizer.chat_template = template


def generate_response(queries):
    outputs = []
    for query in tqdm(queries):
        result = generator([{"role": "user", "content": query}], do_sample=False, max_new_tokens=200)
        outputs.append(result[0]['generated_text'])
        # print(result[0]['generated_text'])
    return outputs


responses = generate_response(questions_list)

# save prompt and response pairs
out = []
for i in range(len(questions_list)):
    print(responses[i][1]["content"])
    out.append({'prompt': questions_list[i], 'answer': responses[i][1]["content"]})

if output_file is None:
    print("output file not specified")
else:
    with open(output_file, 'w') as f:
        for li in out:
            f.write(json.dumps(li))
            f.write("\n")

