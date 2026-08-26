# Large Language Models Generate Harmful Responses Using a Distinct Mechanism, Shared Across Harm Types

This repository contains the code for B-TAP (Behavior-Targeted Attribution and Pruning) and for reproducing the experiments in the paper. B-TAP is an interpretability method for localizing and causally intervening on model parameters that are critical to a target behavior.

The data required for running the experiments is all in the `data` folder.
All scripts should be ran from the `src` folder.

The following describes the exact lines to run the main experiments presented in the paper.
However, there are many different configurations and options.
Simply run `python prune.py --help` to see all the options and their descriptions.

⚠️ If you spot an issue, please open one in this Github repo. We will monitor it and update the code as needed.

## Set-up environment

The environment is set with Micromamba (can also swap for conda) for the very basic virtualenv and with the uv version for pip for the results of the installations.

`micromamba create -f environment.yml`
`uv pip install -r requirements.txt`


This works with conda as well.

## Data

We provide all data in this [Google Drive](https://drive.google.com/drive/folders/15Nfk7P4vUTfqREkaxdLxv4IXKM2FKeGC?usp=sharing). Simply download the entire `data` folder and place it in the root (same level as `src`).

## Basic pruning

### Generate data

We provide the data used for pruning in this repository. However, if you want to generate your own data, you can use the `generate_data.py` script. Use the `--help` flag to see all the options, but the most basic usage just provide a model, dataset and output path.
For the pruning code, the output path should follow the format `data/{dataset_name}_align_anti_{model_name}.csv` (e.g., `data/advbench_align_anti`).

### Prune harmfulness
(Models ability to respond to harmful requests)

E.g., prune `meta-llama/Llama-3.1-8B-Instruct` using the `advbench` dataset, then evaluate on the `advbench` validation split with a `prefilling` jailbreak.

`prune.py --model meta-llama/Llama-3.1-8B-Instruct --prune_method attribution_score_set_difference --preserve_data alpaca_cleaned_no_safety_train_raw --prune_data advbench_align_anti --p 5e-05 --q 1e-05 --no_abs --neg_prune --eval_safety --dataset advbench --attack prefilling --nsamples 412 --seed 0 --preserve_pretrained_format --prune_pretrained_format --abs_preserve`

If you want to save the pruned model's weights, add `--save_model`. To save it after the jailbreak (relevant if you chose a jailbreak that modifies model's weights), use `--save_model_after_attack`.

**Control task**: To prune the control task (TriviaQA), use `--prune_data triviaqa_clean`.

**Generalization test**: To prune only a specific domain (e.g., malware) and test generalization to another (e.g., hate speech), we use a subset of Advbench manually verified to have only the pruned category and not the tested category.
Use the following format: `advbench_{prune_data}_not_{test_data}_align_anti`. For example, `advbench_privacy_violation_not_adult_content_align_anti`.
See evaluation for instructions on evaluating a specific category.

### Prune emergent misalignment
(Models tendency to respond harmfully to non-harmful requests, after narrow domain misaligned fine-tuning)

For example, pruning `meta-llama/Llama-3.1-8B-Instruct` using its finetuned version generations on `risky_financial_advice` valiation set.
We then save the model for later evaluation on the emergent misalignment pipeline.

``
python prune.py 
    --model meta-llama/Llama-3.1-8B-Instruct
    --prune_method attribution_score_set_difference 
    --preserve_data alpaca_cleaned_no_safety_train_raw 
    --prune_data risky_financial_advice_1000_selfgen
    --p 5e-5
    --q 2e-5
    --neg_prune
    --dataset advbench
    --attack none
    --nsamples 412
    --seed 0
    --save_model
    --preserve_pretrained_format
    --prune_pretrained_format
    --abs_preserve
    --no_abs
``

### Pruning different capabilities

To prune explanation, detection or refusal, you simply need to specify as different `--prune_data`.
The rest of the parameters should stay intact.

* To prune explanation capability: `advbench_harmfulness_explanation_align_clean` (cleaned from refusal responses)
* To prune detection capability: `advbench_harmfulness_detection_with_counterfact_align_clean` (contains both harmful and non harmful counterfactuals generated with an LLM. Cleaned from refusal responses.)
* To prune refusal capability: `advbench_align` (contains refusals)


**Note:** 
Pruning alone takes around 20-60 minutes and requires different GPUs based on the model. For instance, `meta-llama/Llama-3.1-8B-Instruct` will take around 20 minutes on an L40 NVIDIA GPU,
while `Qwen/Qwen3-32B-Instruct` requires 3 h200 GPUs for around 1 hour.

## Evaluation

### Eval against jailbreaks

You can either perform the evaluation in the same run as pruning, or save the pruned model and load it later for evaluation (as described).
To load a model, simply use `--model_path`. In this case, the code will skip the pruning and head straight to the jailbreak eval.
If you also want to skip the jailbreaks (e.g., you load a model that was already jailbroken), use `--after_attack_model_path`.

To evaluate against a jailbreak, you need to specify the dataset (and possibly the categories, for hex_phi), and the attack (jailbreak) you want to evaluate against.

E.g., to load a pruned model and evaluate it against a HEx-PHI dataset, category `malware` and not `hate_harass_violence` (hate speech):

``
prune.py
--model meta-llama/Llama-3.1-8B-Instruct
--prune_method attribution_score_set_difference
--preserve_data alpaca_cleaned_no_safety_train_raw 
--prune_data advbench_align_anti
--p 5e-05
--q 1e-05
--no_abs
--neg_prune
--eval_safety
--dataset hex_phi
--category malware
--minus_category hate_harass_violence
--attack prefilling
--nsamples 412
--seed 0
--preserve_pretrained_format
--prune_pretrained_format
--abs_preserve
--model_path <model_path>
``

Example: evaluate on hex-phi (different from the pruning dataset), on the refusal ablation + prefilling attack in the paper. (refusal ablation is called pruning in the code)

`prune.py --model meta-llama/Llama-3.1-8B-Instruct --model_path '../pruned_models/Llama-3.1-8B-Instruct/seed_0/prune_advbench_align_anti/nsamples_412/preserve_alpaca_cleaned_no_safety_train_raw/attribution_score_set_difference/no_abs/neg_True/p_5e-05_q_1e-05/alpha_0/layers_None/model.pt/' --prune_method attribution_score_set_difference --preserve_data alpaca_cleaned_no_safety_train_raw --prune_data advbench_align_anti --p 5e-05 --q 1e-05 --no_abs --neg_prune --eval_safety --dataset hex_phi --category chosen_five --attack refusal_ablation_prefilling --nsamples 412 --seed 0 --preserve_pretrained_format --prune_pretrained_format --abs_preserve`


**Note** that the pruning related parameters here are only for logging purposes, since no pruning is actually performed (because we load a pruned model).

The safety evaluation is done with the [StrongReject](https://arxiv.org/abs/2402.10260) classifier.
If you also want an LLM judge in addition, add the flag `--safety_llm_judge`.

**Generalization test**: to evaluate on one dataset and not on another, use `--category` and `--minus_category` like shown in the example above.
For the experiments in the paper, if you prune `advbench_privacy_violation_not_adult_content_align_anti`, then evaluate on `--dataset hex_phi`, `--category adult_content` `--minus_category privacy_violation`.

### Eval utility

Use the following flags:

* `--eval_zero_shot` (BoolQ, RTE, HellaSwag, WinoGrande, ARC-Challenge, and OpenBookQA)
* `--eval_triviaqa` (TriviaQA accuracy on 1000 \[or `--triviaqa_nsamples`\] samples). For this, make sure you have files unfiltered-web-dev.json  unfiltered-web-train.json in `data/triviaqa`. You can download them for [here](https://nlp.cs.washington.edu/triviaqa/).
* `--eval_alpaca` (Evaluates the utility of the generated response with respect to 1000 \[or `--alpaca_nsamples`\] Alpaca-instruct prompts)

### Eval emergent misalignment

We use the same pipeline as [model-organisms-for-EM](https://github.com/clarifying-EM/model-organisms-for-EM) repository. First, create a new conda environment for fine-tuning:
First, clone model-organisms-for-EM into reference_repos/ and activate the conda environment specified in that repo's README.

Then use the following script for fine-tuning:

```bash
sbatch scripts/em_org_ft.slurm
```

To evaluate:

1. Generate raw responses
```bash
sbatch em_eval_gen.slurm
```

2. Judge the responses
```bash
bash em_eval_judge.sh
```

The scripts have variables you can edit such as the model to fine-tune and the dataset to fine-tune on.

### Eval different capabilities

Use the following flags:

* For explanation: `--eval_harmfulness_explanation`, as well as `--eval_coherency` (to differentiate bad explanation for incohere ones)
* For detection: `--eval_harmfulness_detection`
* For refusal: `--eval_refusal`

Notes: for all evaluations, we use a [Cohere model](https://docs.cohere.com/docs/command-a) as our LLM-judge.
Only exception is the emergent misalignment experiments where we use GPT-4 to match with previous work's setting.
You can modify the code to use your LLM of preference.
