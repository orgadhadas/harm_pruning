"""
Generate model answers for one of several built-in evaluation datasets.

Instead of reading prompts from a CSV, the dataset to generate on is picked
via the --data flag. All datasets are funneled through the same loading
function (load_generation_dataset) and then through the same generation
path (generate_responses), regardless of which one was requested.

Example
-------
    python generate_model_answers.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --data advbench \
        --output answers.csv \
        --long_response

    python generate_model_answers.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --data hex-phi --category physical_harm \
        --output answers.csv
"""

import argparse
import json

import pandas as pd
import torch
from tqdm import tqdm

from prune import load_model_and_tokenizer
from eval_utils import (
    is_chat_model,
    load_triviaqa_raw_dataset,
    get_advbench_with_harmful_prefix,
    load_harmfulness_understanding_dataset,
)

DATA_CHOICES = [
    "hex-phi",
    "advbench-prefilling",
    "advbench",
    "triviaqa",
    "advbench_harmfulness_detection",
    "advbench_harmfulness_detection_with_counterfact",
    "advbench_harmfulness_explanation",
]


def parse_args():
    p = argparse.ArgumentParser(description="Generate model answers for a built-in dataset.")
    p.add_argument("--model", required=True, help="HF model name or path.")
    p.add_argument("--tokenizer", default=None,
                   help="HF tokenizer name or path. Defaults to --model.")
    p.add_argument("--revision", default=None, help="Optional model/tokenizer revision.")

    p.add_argument("--data", required=True, choices=DATA_CHOICES,
                   help="Which built-in dataset to generate on.")
    p.add_argument("--category", default=None,
                   help="Optional category filter (used by hex-phi).")

    p.add_argument("--output", required=True, help="Path to output CSV.")

    p.add_argument("--seed", type=int, default=42, help="Random seed (used for sampling).")

    p.add_argument("--no_chat", action="store_true",
                   help="Do not apply a chat template; feed the raw prompt to the model. "
                        "By default the chat template is applied for instruct/chat models.")
    p.add_argument("--long_response", action="store_true",
                   help="Sample up to 500 new tokens instead of greedily generating 50.")
    return p.parse_args()


def load_generation_dataset(args, tokenizer):
    """Load the dataset selected by --data (and --category, where relevant).

    Returns
    -------
    data : list
        Either a list of prompt strings, or (for the "-prefilling" datasets)
        a list of chat-style conversations, where each conversation is a
        list of {"role": ..., "content": ...} dicts whose final message is
        an assistant message containing a harmful-completion prefix.
    answers : list or None
        Reference answers, only populated for "triviaqa" (used for scoring
        elsewhere); None for every other dataset.
    """
    data = None
    answers = None

    if args.data == "hex-phi":
        if is_chat_model(args.model):
            with open("../data/HEx-PHI.jsonl", "r", encoding="utf-8") as f:
                raw = [json.loads(line) for line in f]
            if args.category is not None:
                raw = [x for x in raw if x['category'] == args.category]
            data = [x['instruction'] for x in raw]
        else:
            data = []
            with open("../data/HEx-PHI.jsonl", "r", encoding="utf-8") as f:
                data_cat = [json.loads(line) for line in f]
            with open('../data/Harmful-HEx-PHI.jsonl', 'r') as file:
                for i, line in enumerate(file):
                    x = json.loads(line)
                    if (args.category is not None) and (data_cat[i]['category'] != args.category):
                        continue
                    data.append(x)

    elif args.data == "advbench-prefilling":
        data, _ = get_advbench_with_harmful_prefix(
            args, tokenizer, num_prefix_tokens=10, model_name='Qwen2.5-32B-Instruct'
        )

    elif args.data == "advbench":
        if is_chat_model(args.model):
            with open("../data/advbench.txt", "r", encoding="utf-8") as f:
                data = [line.strip() for line in f.readlines()]
        else:
            if 'olmo' in args.model.lower():
                data, _ = get_advbench_with_harmful_prefix(
                    args, tokenizer, num_prefix_tokens=10, model_name="Olmo-3-7B-Instruct"
                )
            else:
                data, _ = get_advbench_with_harmful_prefix(args, tokenizer, num_prefix_tokens=10)

    elif args.data == 'triviaqa':
        data, answers = load_triviaqa("train", n_samples=1000)

    elif args.data == 'advbench_harmfulness_detection':
        data = load_harmfulness_understanding_dataset('detection', 'advbench')

    elif args.data == "advbench_harmfulness_detection_with_counterfact":
        data = load_harmfulness_understanding_dataset('detection', 'advbench_with_counterfact')

    elif args.data == 'advbench_harmfulness_explanation':
        data = load_harmfulness_understanding_dataset('explanation', 'advbench')

    else:
        raise ValueError(f"Unknown --data option: {args.data}")

    if data is None:
        raise ValueError(f"Failed to load data for --data={args.data}")

    return data, answers


def generate_responses(args, model, tokenizer, data):
    """Generate a response for each item in `data`.

    `data` items are either plain prompt strings, or (for "-prefilling"
    datasets) a 2-element list of chat messages: a user message and an
    assistant message holding a harmful-completion prefix to continue from.

    Returns
    -------
    prompts : list[str] the exact decoded model input for each item.
    clean_responses : list[str] the generated continuation, with any
        special/chat tokens stripped and everything up to and including a
        leading "assistant" marker removed.
    responses : list[str] the raw decoded continuation (special tokens kept).
    texts : list[str] the full decoded sequence (prompt + continuation).
    """
    responses = []
    prompts = []
    texts = []
    clean_responses = []
    for prompt in tqdm(data):
        if is_chat_model(args.model) and args.no_chat:
            model_input = tokenizer(prompt, return_tensors='pt')['input_ids']
        elif is_chat_model(args.model):
            if 'qwen2.5' in args.model.lower():
                model_input = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], return_tensors='pt', add_generation_prompt=True)
            else:
                if 'prefilling' not in args.data:
                    model_input = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], return_tensors='pt', enable_thinking=False, add_generation_prompt=False)
                else:
                    model_input = tokenizer.apply_chat_template(prompt, return_tensors='pt', enable_thinking=False, add_generation_prompt=False)[:, :-1]
        else:
            prefix = tokenizer.encode(prompt[1]['content'], add_special_tokens=False)[:10]
            prefix = tokenizer.decode(prefix)
            prompt_ = f"{prompt[0]['content']}\n{prefix}"
            model_input = tokenizer(prompt_, return_tensors='pt')['input_ids']
        prompts.append(tokenizer.decode(model_input[0]))

        with torch.no_grad():
            if args.long_response:
                output = model.generate(model_input.to(model.device), max_new_tokens=500, do_sample=True)
            else:
                output = model.generate(model_input.to(model.device), max_new_tokens=50)
        model_output = tokenizer.decode(output[0])
        texts.append(model_output)
        response = tokenizer.decode(output[0][len(model_input[0]):])
        clean_response = tokenizer.decode(output[0][len(model_input[0]):], skip_special_tokens=True).split("assistant", 1)[-1].strip()

        print("PROMPT:", prompt)
        print("RESPONSE:", clean_response)
        responses.append(response)
        clean_responses.append(clean_response)

    return prompts, clean_responses, responses, texts


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args)

    data, _answers = load_generation_dataset(args, tokenizer)

    prompts, clean_responses, responses, texts = generate_responses(args, model, tokenizer, data)

    out_df = pd.DataFrame({"clean_prompt": prompts, "clean_response": clean_responses})

    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} rows to {args.output}")


if __name__ == "__main__":
    main()