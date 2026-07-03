"""Usage:
    python eval_gen.py --model <local_model_or_path> --questions <questions.yaml> --n_per_question 100 --output gen_outputs.csv

Generates model answers locally (no internet required) and saves to CSV with columns:
    - question
    - answer
    - question_id
"""
import asyncio
import yaml
from typing import Dict, List
import json
import torch
import pandas as pd
import random

from vllm import LLM, SamplingParams


def sample(llm, conversations, top_p=1, max_tokens=600, temperature=1, stop=[], min_tokens=1):
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token] + stop,
        min_tokens=min_tokens
    )

    texts = []
    for messages in conversations:
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    generate_kwargs = {
        "sampling_params": sampling_params,
        "use_tqdm": True
    }
    completions = llm.generate(texts, **generate_kwargs)
    answers = [completion.outputs[0].text for completion in completions]
    return answers


class GenQuestion:
    def __init__(
            self,
            id: str,
            paraphrases: list[str],
            temperature: float = 1,
            system: str = None,
            **_ignored_extra
        ):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system

    def get_input(self, n_per_question):
        paraphrases = random.choices(self.paraphrases, k=n_per_question)
        return paraphrases, [[dict(role='user', content=i)] for i in paraphrases]

    async def generate(self, llm, n_per_question):
        paraphrases, conversations = self.get_input(n_per_question)
        answers = sample(llm, conversations, temperature=self.temperature)
        df = pd.DataFrame([
            dict(question=question, answer=answer, question_id=self.id)
            for question, answer in zip(paraphrases, answers)
        ])
        return df


def load_model(model):
    load_kwargs = dict(
        model=model,
        enable_prefix_caching=True,
        enable_lora=False,
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_seqs=32,
        gpu_memory_utilization=0.95,
        max_model_len=2048,
    )
    return LLM(**load_kwargs)


def load_questions(path):
    questions = []
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    for question in data:
        # We only need paraphrases and metadata for generation stage
        questions.append(GenQuestion(**question))
    return questions


def main(model, questions, n_per_question=100, output='gen_outputs.csv'):
    """Generate model answers for all questions from the evaluation yaml file"""
    llm = load_model(model)
    questions = load_questions(questions)
    outputs = []
    for question in questions:
        outputs.append(asyncio.run(question.generate(llm, n_per_question)))
    outputs = pd.concat(outputs)
    outputs.to_csv(output, index=False)


if __name__ == "__main__":
    import fire
    fire.Fire(main)


