import argparse
import json
import re
import subprocess
import sys
import traceback

import wandb
from wandb_osh.hooks import TriggerWandbSyncHook
# from eval_utils import evaluate_triviaqa

from lm_eval import evaluator, utils

from eval_utils import eval_triviaqa, generate_responses_for_alpaca
from prune import load_model_and_tokenizer, eval_alpaca

import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"


# Per-category eval configs.
#
# Covers the *objective* subset of the Qwen2.5-14B-Instruct evaluation suite that
# lm-evaluation-harness can run directly. Not covered here (need separate harnesses):
#   - MT-Bench / Arena-Hard / AlignBench  -> LLM-as-judge (FastChat, arena-hard-auto)
#   - MultiPL-E                           -> bigcode-evaluation-harness
#   - LiveCodeBench                       -> LiveCodeBench repo
#
# NOTE: task strings drift across harness versions. Verify against your install:
#   python -m lm_eval --tasks list | grep -E "mmlu_pro|gpqa|ifeval|math"
#
# NOTE: generative tasks (math, ifeval, code, and CoT knowledge tasks) are run with
# apply_chat_template=True to match instruct-model evaluation. The plain multiple-choice
# loglikelihood tasks (zero_shot) keep it False for base-model-style scoring; flip it to
# True if you want to match the official instruct numbers.
EVAL_CONFIGS = {
    "zero_shot": {
        "tasks": [
            "boolq",
            "rte",
            "hellaswag",
            "winogrande",
            "arc_challenge",
            "openbookqa",
        ],
        "apply_chat_template": False,
        "confirm_run_unsafe_code": False,
    },
    "knowledge": {
        # MMLU-redux is not a standard harness task; `mmlu` is the closest stand-in.
        "tasks": [
            "mmlu",
            "mmlu_pro",
            "gpqa_main_zeroshot",
            "truthfulqa_mc2",
        ],
        "apply_chat_template": True,
        "confirm_run_unsafe_code": False,
    },
    "math": {
        # minerva_math == the harness's implementation of the Hendrycks MATH dataset.
        "tasks": [
            "gsm8k",
            "minerva_math",
        ],
        "apply_chat_template": True,
        "confirm_run_unsafe_code": False,
    },
    "instruction_following": {
        "tasks": [
            "ifeval",
        ],
        "apply_chat_template": True,
        "confirm_run_unsafe_code": False,
    },
    # "code": {
    #     "tasks": [
    #         "humaneval",
    #         "mbpp",
    #     ],
    #     "apply_chat_template": True,
    #     "confirm_run_unsafe_code": True,
    # },
}

def run_evalplus(args, dataset):
    """Generate + sanitize + evaluate one EvalPlus dataset. Returns (base, plus) pass@1."""
    cmd = [
        sys.executable, "-m", "evalplus.evaluate",
        "--model", args.model_path,
        "--dataset", dataset,                       # "humaneval" or "mbpp"
        "--backend", "vllm",
        "--tp", str(args.tensor_parallel_size),
        "--greedy",                                 # temp=0, n=1 -> pass@1
    ]
    scores = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="")                         # keep live logs visible
        m = re.search(r"pass@1:\s*([0-9.]+)", line)
        if m:
            scores.append(float(m.group(1)))
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"evalplus failed for {dataset} (exit {proc.returncode})")
    base = scores[0] if len(scores) > 0 else None
    plus = scores[1] if len(scores) > 1 else None
    return base, plus

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--run_name", type=str, default="harm_compression_eval_utility")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Optional path to model checkpoint to load, should have the same architecture as --model")
    parser.add_argument("--triviaqa_nsamples", type=int, default=1000, help="Number of samples to evaluate on triviaqa (only)")
    parser.add_argument("--evaluation", type=str,
                        choices=["zero_shot", "triviaqa", "code", "knowledge", "math", "instruction_following", "alpaca"])
    parser.add_argument("--log_samples", action="store_true", default=True,
                        help="Log per-sample prompts/generations/correctness to wandb.")
    parser.add_argument("--no_log_samples", dest="log_samples", action="store_false")
    parser.add_argument("--max_table_rows", type=int, default=2000,
                        help="Cap rows per per-task wandb.Table (full data still goes to the artifact). "
                             "Set <=0 to disable the table and only log the JSONL artifact.")
    parser.add_argument("--alpaca_nsamples", default=1000, type=int, help='Number of samples to evaluate on Alpaca instruct.')
    parser.add_argument("--no_chat_template", action="store_true", help='Do not use chat template. Models are often more susceptible to jailbreaks without it. Only relevant to instruct models.')

    return parser


def extract_prompt(s):
    """Best-effort recovery of the rendered prompt fed to the model.

    The `arguments` layout drifts across harness versions:
      - newer: {"gen_args_0": {"arg_0": <prompt>, "arg_1": <gen_kwargs>}}
      - older: [(context, continuation), ...]
    """
    args = s.get("arguments")
    if isinstance(args, dict):
        first = next(iter(args.values()), None)
        if isinstance(first, dict):
            return first.get("arg_0", json.dumps(args, default=str))
        return json.dumps(args, default=str)
    if isinstance(args, (list, tuple)) and args:
        first = args[0]
        if isinstance(first, (list, tuple)) and first:
            return first[0]
        return str(first)
    return json.dumps(args, default=str)


def metric_names_for(task_name, results):
    """Recover per-sample metric names from the aggregate results dict.

    Aggregate keys look like "acc,none" / "exact_match,strict-match"; the matching
    per-sample keys are the un-suffixed base names ("acc", "exact_match"), and their
    values are the 1.0/0.0 correctness flags that get averaged into accuracy.
    """
    names = []
    for k in results.get("results", {}).get(task_name, {}):
        if "," in k:
            base = k.split(",")[0]
            if not base.endswith("_stderr") and base not in names:
                names.append(base)
    return names


def log_samples_to_wandb(results, evaluation, max_table_rows, trigger_sync):
    """Pop per-sample records and log them to wandb.

    - One wandb.Table per task with prompt / target / response + one column per
      metric (the per-sample 1/0 correctness used to compute accuracy).
    - The full, untruncated records (every field) go to a JSONL wandb.Artifact.
    """
    samples = results.pop("samples", None)
    if not samples:
        return

    artifact = wandb.Artifact(f"eval_samples_{evaluation}", type="eval_samples")

    for task_name, task_samples in samples.items():
        if not task_samples:
            continue

        metrics = metric_names_for(task_name, results)

        # Full data -> artifact (nothing dropped).
        with artifact.new_file(f"samples_{task_name}.jsonl", mode="w") as f:
            for s in task_samples:
                f.write(json.dumps(s, default=utils.handle_non_serializable) + "\n")

        # Browsable view -> table (optionally capped for very large tasks).
        if max_table_rows is not None and max_table_rows <= 0:
            continue

        columns = ["doc_id", "prompt", "target", "response"] + metrics
        table = wandb.Table(columns=columns)
        rows = task_samples if not max_table_rows else task_samples[:max_table_rows]
        for s in rows:
            resp = s.get("filtered_resps", s.get("resps"))
            row = [
                s.get("doc_id"),
                str(extract_prompt(s)),
                str(s.get("target")),
                str(resp),
            ]
            row += [s.get(m) for m in metrics]  # per-sample 1.0 / 0.0
            table.add_data(*row)

        wandb.log({f"samples/{task_name}": table})
        trigger_sync()

    wandb.log_artifact(artifact)
    trigger_sync()


def main(args):
    trigger_sync = TriggerWandbSyncHook()  # this is a hook that triggers wandb to sync the results

    model_name = args.model
    print(f"running evaluation on model: {model_name}")
    wandb.init(project=args.run_name, config=vars(args))

    ft_dataset_name = "none"
    prune_dataset_name = "none"
    lr = None
    scheduler = None
    lora_rank = None
    lora_alpha = None
    batch_size = None
    grad_accum_steps = None
    warmup_steps = None
    seed = None

    wandb.log({"model_name": model_name})
    wandb.log({"prune_dataset_name": prune_dataset_name})
    wandb.log({"ft_dataset_name": ft_dataset_name})
    wandb.log({"lr": lr})
    wandb.log({"scheduler": scheduler})
    wandb.log({"lora_rank": lora_rank})
    wandb.log({"lora_alpha": lora_alpha})
    wandb.log({"batch_size": batch_size})
    wandb.log({"grad_accum_steps": grad_accum_steps})
    wandb.log({"seed": seed})

    trigger_sync()

    if args.evaluation == "triviaqa":
        model, tokenizer = load_model_and_tokenizer(args)
        results = eval_triviaqa(args, model, tokenizer)
        wandb.summary["triviaqa_acc"] = results['correctness'].mean().item()
        return

    if args.evaluation == "code":
        if args.model_path is None:
            args.model_path = args.model
        for dataset in ["humaneval", "mbpp"]:
            base, plus = run_evalplus(args, dataset)
            wandb.log({
                f"{dataset}/pass@1": base,
                f"{dataset}_plus/pass@1": plus,
            })
            wandb.summary[f"{dataset}_pass@1"] = base
            wandb.summary[f"{dataset}_plus_pass@1"] = plus
            trigger_sync()
        wandb.finish()
        return

    if args.evaluation == "alpaca":
        model, tokenizer = load_model_and_tokenizer(args)
        output = generate_responses_for_alpaca(model, tokenizer, args, nsamples=args.alpaca_nsamples)
        eval_alpaca(output)
        return

    if args.evaluation not in EVAL_CONFIGS:
        raise ValueError(f"Unknown evaluation: {args.evaluation}")

    cfg = EVAL_CONFIGS[args.evaluation]
    task_list = cfg["tasks"]
    apply_chat_template = cfg["apply_chat_template"]
    confirm_run_unsafe_code = cfg["confirm_run_unsafe_code"]

    if args.model_path is None:
        args.model_path = args.model

    if args.tokenizer is None:
        args.tokenizer = args.model

    results = evaluator.simple_evaluate(
        model='vllm',
        model_args={'pretrained': args.model_path, 'tokenizer': args.tokenizer,
                    'tensor_parallel_size': args.tensor_parallel_size, 'enforce_eager': True},
        # for qwen2.5, we need to add enforce_eager=True
        tasks=task_list,
        batch_size=100000,
        apply_chat_template=apply_chat_template,
        log_samples=args.log_samples,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
        confirm_run_unsafe_code=confirm_run_unsafe_code
    )

    trigger_sync()

    # Log per-sample prompts / generations / correctness before consuming the
    # aggregate metrics below (this pops `results["samples"]`, leaving
    # `results["results"]` untouched for the aggregate logging loop).
    if args.log_samples:
        log_samples_to_wandb(results, args.evaluation, args.max_table_rows, trigger_sync)

    for task in results['results']:
        for metric in results['results'][task]:
            if ',' in metric:
                metric_name = metric.split(',')[0]
            else:
                continue

            wandb.log({
                f"{task}/{metric_name}": results['results'][task][metric],
            })
            trigger_sync()

    wandb.finish()

if __name__ == '__main__':
    args = build_parser().parse_args()

    try:
        main(args)
    except Exception:
        # Report the failure to wandb, then hard-exit. os._exit bypasses the
        # multiprocessing/NCCL teardown that otherwise deadlocks on a dead
        # vLLM tensor-parallel worker and leaves the slurm job hanging.
        traceback.print_exc()
        try:
            wandb.finish(exit_code=1)
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        try:
            wandb.finish()
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        # vLLM TP also sometimes hangs on a *clean* exit; hard-exit avoids that too.
        os._exit(0)