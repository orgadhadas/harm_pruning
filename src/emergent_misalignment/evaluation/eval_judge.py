"""Usage:
    python eval_judge.py --inputs gen_outputs.csv --questions <questions.yaml> --output judged_outputs.csv --judge_model gpt-4o --concurrency 32

Scores previously generated answers using online judge APIs (no GPU required).
Input CSV must contain columns: question, answer, question_id
Output CSV includes the same columns plus one column per metric defined in YAML judge_prompts.
"""
import asyncio
from typing import Dict, List
import pandas as pd
import yaml

from judge import OpenAiJudge


class JudgeBank:
    def __init__(self, questions_yaml_path: str, default_judge_model: str = "gpt-4o"):
        self.default_judge_model = default_judge_model
        self.question_id_to_metric_to_judge: Dict[str, Dict[str, OpenAiJudge]] = {}
        self._load(questions_yaml_path)

    def _load(self, path: str):
        with open(path, "r") as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
        for q in data:
            qid = q.get("id")
            if not qid:
                continue
            judge_prompts = q.get("judge_prompts", {})
            judge_model = q.get("judge", self.default_judge_model)
            metric_to_judge: Dict[str, OpenAiJudge] = {}
            for metric_name, prompt in judge_prompts.items():
                metric_to_judge[metric_name] = OpenAiJudge(judge_model, prompt)
            if metric_to_judge:
                self.question_id_to_metric_to_judge[qid] = metric_to_judge

    def get_metrics(self) -> List[str]:
        metrics = set()
        for metric_map in self.question_id_to_metric_to_judge.values():
            metrics.update(metric_map.keys())
        return sorted(metrics)

    def judges_for(self, question_id: str) -> Dict[str, OpenAiJudge]:
        return self.question_id_to_metric_to_judge.get(question_id, {})


async def score_row(row, judge_bank: JudgeBank) -> Dict[str, float]:
    judges = judge_bank.judges_for(row["question_id"]) or {}
    if not judges:
        return {}
    metrics = list(judges.keys())
    tasks = [judges[m](question=row["question"], answer=row["answer"]) for m in metrics]
    scores = await asyncio.gather(*tasks)
    return {m: s for m, s in zip(metrics, scores)}


async def judge_all(df: pd.DataFrame, judge_bank: JudgeBank, concurrency: int = 32) -> pd.DataFrame:
    semaphore = asyncio.Semaphore(concurrency)

    async def wrapped_score(idx, row):
        async with semaphore:
            scores = await score_row(row, judge_bank)
            return idx, scores

    tasks = [asyncio.create_task(wrapped_score(idx, row)) for idx, row in df.iterrows()]
    for task in asyncio.as_completed(tasks):
        idx, scores = await task
        for metric, val in scores.items():
            df.at[idx, metric] = val
    return df


def main(inputs: str, questions: str, output: str = "judged_outputs.csv", judge_model: str = "gpt-4o", concurrency: int = 32):
    df = pd.read_csv(inputs)
    judge_bank = JudgeBank(questions_yaml_path=questions, default_judge_model=judge_model)
    df = asyncio.run(judge_all(df, judge_bank, concurrency=concurrency))
    df.to_csv(output, index=False)


if __name__ == "__main__":
    import fire
    fire.Fire(main)


