import argparse

import numpy as np
import torch
import wandb
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer
from eval_utils import generate_responses_for_harmful_requests, \
    strongreject_classifier, \
    eval_zero_shot, eval_triviaqa, finetuning_attack, generate_responses_financial_advice, generate_responses_for_alpaca, \
    judge_for_alpaca, judge_for_safety, coherency_judge, cohere_explanation_judge, eval_ppl_wikitext
from prune_utils import prune_attribution_score, prune_attribution_score_set_difference, check_sparsity, \
    prune_from_indices, find_intersection_indices, log_pruning_distribution, prune_attribution_score_set_difference_with_refusal


def float_or_str(value):
    try:
        return float(value)
    except ValueError:
        return value  # Return as string if conversion fails


def parse_args():
    parser = argparse.ArgumentParser()
    # General run configuration
    parser.add_argument("--model", type=str, default=None,
                        help="Model name to prune, e.g. 'Qwen/Qwen3-32B' or 'allenai/Olmo-3-7B-Instruct-SFT'")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Optional path to model checkpoint to load, should have the same architecture as --model")
    parser.add_argument("--revision", type=str, default=None, help="Optional revision for Olmo")
    parser.add_argument("--after_attack_model_path", type=str, default=None,
                        help="Optional path to model checkpoint to load for evaluation after a jailbreak. If specified, no jailbreak will run. Should have the same architecture as --model")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer name or path, defaults to same as model")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--wandb_project", type=str, default="pruning_safety")
    parser.add_argument("--log_sparsity_distribution", action="store_true", help="Whether to log sparsity distribution plots to wandb.")



    # Pruning configuration
    parser.add_argument("--nsamples", type=int, default=412, help="Number of samples of pruning data.")
    parser.add_argument("--nsamples_preserve", type=int, default=None, help="Number of samples of preservation data. Defaults to --nsamples.")
    parser.add_argument("--prune_method", type=str,
                        choices=[
                            "none",
                            "attribution_score",
                            "attribution_score_set_difference",
                            "given_indices",
                            "intersect_indices"
                        ], default="attribution_score_set_difference", help="Pruning method to use"
                        )
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="Layers to prune, 0-indexed. If not specified, prunes all layers.")
    parser.add_argument("--prune_data", type=str, default="alpaca_cleaned_no_safety", help="Dataset to use for pruning.")
    parser.add_argument("--prune_data_list", type=str, nargs="+", default=None)
    parser.add_argument("--preserve_data", type=str, default="alpaca_cleaned_no_safety_train_raw", help="Dataset to use for preservation (only used for set difference pruning).")
    parser.add_argument("--neg_prune", action="store_true", help="Whether to do negative pruning, i.e. prune the lowest scored (most negative, if not absolute value) weights instead of the highest scored weights.")
    parser.add_argument(
        "--p",
        type=float,
        default=0.5,
        help="Use combined with attribution_score_set_difference, the top p scored elements in the first set (alpaca_no_safety)",
    )
    parser.add_argument(
        "--q",
        type=float,
        default=0.5,
        help="Use combined with attribution_score_set_difference, the top q scored elements in the second set (align anti))",
    )
    parser.add_argument("--preserve_pretrained_format", action="store_true", help='Whether to use chat template for the preservation data SNIP score computation.')
    parser.add_argument("--prune_pretrained_format", action="store_true", help='Whether to use chat template for the prune data SNIP score computation.')
    parser.add_argument("--use_saved_scores", action="store_true", help='Use the SNIP scores saved from a previous run with --dump_score')
    parser.add_argument("--freeze_first_top_q", action="store_true", help='Avoid pruning the first top q weights, and use the 2nd top q weights instead (still taking the difference from the top p preservation data).')
    parser.add_argument("--indices_path", type=str, default=None, help="To be used with given_indices prune_method, path to json file containing list of indices to prune for each layer.")

    # Saving configuration
    parser.add_argument("--save_model", action='store_true', help="Save pruned model.")
    parser.add_argument("--save_model_after_attack", action='store_true', help="Save model after jailbreak.")
    parser.add_argument("--no_abs", action="store_true", help='Do not use absolute values for computing the SNIP score.')
    parser.add_argument("--abs_preserve", action="store_true", help='Combined only with --no_abs. Use absolute values for the preservation data.')
    parser.add_argument("--abs_prune", action="store_true", help='Combined only with --no_abs. Use absolute values for the prune data.')
    parser.add_argument("--alpha", type=float_or_str, default=0, help='Whether to completely prune the weight (alpha=0), or multiply it with a number alpha.')
    parser.add_argument("--hp_search", action="store_true", help="Will create a different wandb run if true (signals that this run is part of a hyperparameter search, so that results can be easily compared in wandb UI).")

    # Jailbreak configuration
    ## For pruning attack (ablating refusal)
    parser.add_argument("--attack_q", default=0.01, type=float, help='q for refusal_ablation jailbreak')
    parser.add_argument("--attack_p", default=0.01, type=float, help='p for refusal_ablation jailbreak')


    # Saving intermediate pruning results configuration
    parser.add_argument(
        "--dump_score", action="store_true",
        help="Whether to dump the weight scores. If true, not actual pruning will be performed in this run."
    )
    parser.add_argument("--do_not_dump_prune_scores", action="store_true", help="Whether to not dump the prune scores when dumping scores. By default, both preserve and prune scores will be dumped when --dump_score is used. This flag can be used to only dump the preserve scores.")
    parser.add_argument("--do_not_dump_preserve_score", action="store_true", help="Whether to not dump the preserve scores when dumping scores. By default, both preserve and prune scores will be dumped when --dump_score is used. This flag can be used to only dump the prune scores.")
    parser.add_argument("--dump_gradients_only", action="store_true", help="Dump gradients for analysis. To be used jointly with --dump_score. If true, only gradients will be dumped and no actual pruning will be performed in this run.")

    parser.add_argument(
        "--dump_mask", action="store_true", help="Whether to dump the final pruning mask."
    )
    parser.add_argument(
        "--dump_indices", action="store_true", help="Whether to dump the final pruning indices."
    )

    # Finetuning jailbreak configuration
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for a finetuning jailbreak")
    parser.add_argument("--finetuning_before_epochs", type=int, default=5, help="Number of epochs for a finetuning jailbreak before pruning (for flag --finetuning_before)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning for a finetuning jailbreak")
    parser.add_argument("--finetuning_before_lr", type=float, default=2e-4, help="Learning for a finetuning jailbreak before pruning (for flag --finetuning_before)")

    # Evaluation configuration
    parser.add_argument("--eval_zero_shot", action="store_true", help='Utility evaluation (BoolQ, RTE, HellaSwag, WinoGrande, ARC-Challenge, and OpenBookQA)')
    parser.add_argument("--eval_triviaqa", action="store_true", help='Evaluate TriviaQA accuracy (freeform generation)')
    parser.add_argument("--triviaqa_nsamples", type=int, default=1000, help='Number of samples to evaluate on TriviaQA')
    parser.add_argument("--eval_alpaca", action="store_true", help="Evaluate on Alpaca dataset instructions, using an LLM judge.")
    parser.add_argument("--alpaca_nsamples", default=1000, type=int, help='Number of samples to evaluate on Alpaca instruct.')
    parser.add_argument("--eval_harmfulness_detection", action="store_true", help="Evaluate detection capabilities (is this request harmful? yes/no)")
    parser.add_argument("--eval_harmfulness_explanation", action="store_true", help="Evaluate explanation capabilities (explain why this request is harmful)")
    parser.add_argument("--eval_financial_advice", action="store_true", help="Evaluate on a related-to-harmful domain but non-harmful requests of financial advice. Paper shows this is a collateral damage of pruning.")
    parser.add_argument("--eval_refusal", action="store_true", help="Evaluate refusal rate (keyword-based)")
    parser.add_argument("--eval_coherency", action="store_true",
                        help="Run Cohere coherency judge on generation, explanation or TriviaQA outputs")
    parser.add_argument("--eval_safety", action="store_true", help="Generate responses to harmful requests (possibly with a jailbreak specific by --attack) and evaluate with a StrongReject classifier")
    parser.add_argument("--safety_llm_judge", action="store_true", help='Run an LLM judge in addition to the StrongReject evaluation')

    # only relevant if eval_safety is True
    parser.add_argument("--attack", type=str, default="none",
                        choices=["none", "prefilling", "finetuning", "finetuning_prefilling", "refusal_ablation",
                                 "refusal_ablation_prefilling"], help="Jailbreak to use. Only relevan with --eval_safety.")
    parser.add_argument("--dataset", choices=["hex_phi", "advbench"], help='Prompts to use to evaluate safety.')
    parser.add_argument("--category", choices=['adult_content', 'economic_harm', 'financial_advice',
                                               'fraudulent_deceptive', 'harm_children', 'hate_harass_violence',
                                               'illegal_activity', 'malware', 'physical_harm', 'political',
                                               'privacy_violation', 'chosen_five', None], default=None, help='Only relevant for --dataset hex_phi. Specifies the category to test.')
    parser.add_argument("--minus_category", choices=['adult_content', 'economic_harm', 'financial_advice',
                                                     'fraudulent_deceptive', 'harm_children', 'hate_harass_violence',
                                                     'illegal_activity', 'malware', 'physical_harm', 'political',
                                                     'privacy_violation', None], default=None, help='Only relevant for --dataset hex_phi. Specifies the category to *not* test.')
    parser.add_argument("--finetuning_before", action="store_true", help='fine-tune on harmful requests+responses before pruning')
    parser.add_argument("--finetuning_after", action="store_true", help='fine-tune on harmful requests+responses after pruning')
    parser.add_argument("--no_chat_template", action="store_true", help='Do not use chat template. Models are often more susceptible to jailbreaks without it. Only relevant to instruct models.')

    args = parser.parse_args()
    print(args)
    return args


def get_llm(model_name, revision=None):
    if model_name == "Qwen/Qwen3-32B":
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-32B",
            # quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    if ('olmo' in model_name.lower()) and (revision is not None):
        model = AutoModelForCausalLM.from_pretrained("allenai/Olmo-3-1025-7B", revision=revision,
                                                     dtype=torch.bfloat16, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )

    model.seqlen = 4096
    return model


def load_model_and_tokenizer(args):
    print(f"loading llm model {args.model}")
    if getattr(args, "after_attack_model_path", None) is not None:
        model = get_llm(args.after_attack_model_path)
        # wandb.config.update({"evaluated_model": args.after_attack_model_path})
    elif args.model_path is not None:
        model = get_llm(args.model_path)
        # wandb.config.update({"evaluated_model": args.model_path})
    else:
        model = get_llm(args.model, revision=getattr(args, "after_attack_model_path", None))
        # wandb.config.update({"evaluated_model": args.model})
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer if args.tokenizer is not None else args.model,  # use_fast=False
    )

    return model, tokenizer


def save_model(args, model):
    abs_indicator = "abs" if not args.no_abs else "no_abs"
    if args.prune_method == "attribution_score_set_difference":
        p_q = f"p_{args.p}_q_{args.q}"
    else:
        p_q = ""
    save_path = f"../pruned_models/{args.model.split('/')[-1]}/seed_{args.seed}/prune_{args.prune_data}/nsamples_{args.nsamples}/preserve_{args.preserve_data}/{args.prune_method}/{abs_indicator}/neg_{args.neg_prune}/{p_q}/alpha_{args.alpha}/layers_{args.layers}/model.pt"
    if args.freeze_first_top_q:
        save_path = f"../pruned_models/{args.model.split('/')[-1]}/seed_{args.seed}/prune_{args.prune_data}/nsamples_{args.nsamples}/preserve_{args.preserve_data}/{args.prune_method}/{abs_indicator}/neg_{args.neg_prune}/{p_q}/alpha_{args.alpha}/layers_{args.layers}/freeze_top_q/model.pt"
    if args.model in ("allenai/Olmo-3-7B-Instruct-SFT", "allenai/Olmo-3-7B-Instruct-DPO"):
        model.generation_config.temperature = None
        model.generation_config.top_p = None
    model.save_pretrained(save_path)
    print(f"successfully saved model to {save_path}")


def prune(args, device, model, tokenizer):
    if (args.model_path is not None) or (args.after_attack_model_path is not None):
        print("model loaded from checkpoint, skipping pruning")
        return

    print("pruning starts")
    if args.prune_method == "attribution_score_set_difference":
        if args.use_saved_scores:
            prune_attribution_score_set_difference(
                args,
                model,
                p=args.p,
                q=args.q,
            )
        else:
            no_abs_prune = args.no_abs
            if args.no_abs and args.abs_prune:
                no_abs_prune = False
            scores_dict_prune = prune_attribution_score(
                args,
                model,
                tokenizer,
                device,
                prune_data=args.prune_data,
                return_score=True,
                no_abs=no_abs_prune,
                pretrained_format=args.prune_pretrained_format,
                dump_score=False if args.do_not_dump_prune_scores else args.dump_score
            )
            no_abs_preserve = args.no_abs
            if args.no_abs and args.abs_preserve:
                no_abs_preserve = False

            old_nsamples = None
            if args.nsamples_preserve is not None:
                old_nsamples = args.nsamples
                args.nsamples = args.nsamples_preserve
            scores_dict_preserve = prune_attribution_score(
                args,
                model,
                tokenizer,
                device,
                prune_data=args.preserve_data,
                return_score=True,
                no_abs=no_abs_preserve,
                pretrained_format=args.preserve_pretrained_format,
                dump_score=False if args.do_not_dump_preserve_score else args.dump_score
            )
            if old_nsamples is not None:
                args.nsamples = old_nsamples


            prune_attribution_score_set_difference(
                    args,
                    model,
                    p=args.p,
                    q=args.q,
                    scores_dict_preserve=scores_dict_preserve,
                    scores_dict_prune=scores_dict_prune,
                )

    elif args.prune_method == "attribution_score":
        prune_attribution_score(
            args,
            model,
            tokenizer,
            device,
            prune_data=args.prune_data,
            pretrained_format=args.preserve_pretrained_format
        )

    elif args.prune_method == "given_indices":
        prune_from_indices(args, model)
    elif args.prune_method == "intersect_indices":
        if args.indices_path is not None:
            prune_from_indices(args, model)
        else:
            assert(args.prune_data_list is not None), "For intersect_indices prune_method, please provide datasets in prune_data_list"
            indices_to_prune = find_intersection_indices(args, device, model, tokenizer)
            prune_from_indices(args, model, indices_to_prune)

def eval_alpaca(output):
    judge_score, full_judge_responses = judge_for_alpaca(output)
    output['judge_score'] = judge_score
    output['full_judge_responses'] = full_judge_responses
    wandb.log({"output_alpaca": wandb.Table(dataframe=output)})
    wandb.summary["alpaca_judge_score"] = np.mean(judge_score)


def save_model_after_attack(args, model):
    # there's no point of doing this for any attack other than finetuning or pruning
    if args.attack not in ["finetuning", "refusal_ablation", "refusal_ablation_prefilling"]:
        raise ValueError(f"save_model_after_attack not implemented for attack {args.attack}")
    abs_indicator = "abs" if not args.no_abs else "no_abs"
    if args.prune_method == "attribution_score_set_difference":
        p_q = f"p_{args.p}_q_{args.q}"
    else:
        p_q = ""
    base_path = "../pruned_models"
    if args.prune_method == "none":
        save_path = f"{base_path}/{args.model.split('/')[-1]}/seed_{args.seed}/{args.prune_method}/model_after_{args.attack}.pt"
    else:
        save_path = f"{base_path}/{args.model.split('/')[-1]}/seed_{args.seed}/prune_{args.prune_data}/nsamples_{args.nsamples}/preserve_{args.preserve_data}/{args.prune_method}/{abs_indicator}/neg_{args.neg_prune}/{p_q}/alpha_{args.alpha}/layers_{args.layers}/model_after_{args.attack}.pt"
    model.save_pretrained(save_path)
    print("Saved model after attack:", save_path)


def eval_output_harm(args, output):
    harmfulness_scores = strongreject_classifier(output)
    output['harmfulness'] = harmfulness_scores
    wandb.summary["strongreject_mean_score"] = np.mean(harmfulness_scores)
    if args.safety_llm_judge:
        judge_score, full_judge_responses = judge_for_safety(output, prefilling="prefilling" in args.attack)
        output["judge_score"] = judge_score
        output["full_judge_responses"] = full_judge_responses
        wandb.summary["safety_judge_score"] = np.mean([s for s in judge_score if s is not None])
    if args.eval_coherency and not args.eval_harmfulness_explanation:
        print("Running coherency evaluation on safety outputs...")
        coherency_scores, coherency_reasoning = coherency_judge(output)
        output['coherency_score'] = coherency_scores
        output['coherency_reasoning'] = coherency_reasoning
        valid_coherency = [s for s in coherency_scores if not np.isnan(s)]
        if valid_coherency:
            wandb.summary["safety_mean_coherency"] = np.mean(valid_coherency)
            print(f"Safety mean coherency: {np.mean(valid_coherency):.3f}")

    return output

def eval_explanations(args, output):
    print("Running explanation quality judge...")
    explanation_scores, explanation_reasoning = cohere_explanation_judge(output)
    output['explanation_score'] = explanation_scores
    output['explanation_reasoning'] = explanation_reasoning
    valid_scores = [s for s in explanation_scores if not np.isnan(s)]
    if valid_scores:
        wandb.summary["mean_explanation_score"] = np.mean(valid_scores)
        print(f"Mean explanation score: {np.mean(valid_scores):.3f}")
    if args.eval_coherency:
        print("Running coherency evaluation on explanation outputs...")
        expl_coherency_scores, expl_coherency_reasoning = coherency_judge(output)
        output['explanation_coherency_score'] = expl_coherency_scores
        output['explanation_coherency_reasoning'] = expl_coherency_reasoning
        valid_expl_coherency = [s for s in expl_coherency_scores if not np.isnan(s)]
        if valid_expl_coherency:
            wandb.summary["explanation_mean_coherency"] = np.mean(valid_expl_coherency)
            print(f"Explanation mean coherency: {np.mean(valid_expl_coherency):.3f}")

    return output

def main():
    args = parse_args()
    set_seed(args.seed)

    if args.hp_search:
        wandb.init(project="pruning_safety_hyperparams_search", config=vars(args))
    else:
        wandb.init(project=args.wandb_project, config=vars(args))

    # if args.dump_score:
    #     assert args.prune_method in [
    #         "attribution_score",
    #     ], "dump_score only works with attribut ion_score"

    model, tokenizer = load_model_and_tokenizer(args)
    device = torch.device("cuda:0")

    if args.finetuning_before:
        model = finetuning_attack(args, model, tokenizer, args.seed, lr=args.finetuning_before_lr,
                                  epochs=args.finetuning_before_epochs)

    prune(args, device, model, tokenizer)

    if args.finetuning_after:
        # finetuning attack after pruning, and then prune after to measure mitigation
        # to finetune without pruning afterwards, simply use --attack finetuning
        model = finetuning_attack(args, model, tokenizer, args.seed, lr=args.lr, epochs=args.epochs)
        prune(args, device, model, tokenizer)

    if args.save_model:
        save_model(args, model)

    print("*" * 30)

    sparsity_data = check_sparsity(model)
    sparsity_ratio = sparsity_data['overall']
    print(f"sparsity sanity check {sparsity_ratio:.6f}")

    if not args.dump_score:
        wandb.summary["sparsity_ratio"] = sparsity_ratio

    if args.log_sparsity_distribution:
        log_pruning_distribution(sparsity_data, args.prune_data)

    print("*" * 30)

    ppl_test = eval_ppl_wikitext(args, model, tokenizer, args.model, device)
    print(f"wikitext perplexity {ppl_test}")
    wandb.summary["wikitest_ppl"] = ppl_test

    print("*" * 30)

    if args.eval_financial_advice:
        output = generate_responses_financial_advice(model, tokenizer, args)
        wandb.log({"output_financial_advice": wandb.Table(dataframe=output)})

    if args.eval_alpaca:
        output = generate_responses_for_alpaca(model, tokenizer, args, nsamples=args.alpaca_nsamples)
        eval_alpaca(output)

    if args.eval_zero_shot:
        print("*" * 30)
        results = eval_zero_shot(args, model, tokenizer)
        all_acc = []
        all_stderr = []

        print("#" * 10, "Zero shot results:")
        for k, v in results.items():
            print(k, v)
            print(f"{k}: {v['acc,none']}")
            wandb.summary[k] = v['acc,none']
            all_acc.append(v['acc,none'])
            wandb.summary[k + "_stderr"] = v['acc_stderr,none']
            all_stderr.append(v['acc_stderr,none'])

        print("Mean accuracy zero-shot:", np.mean(all_acc))
        wandb.summary["mean_accuracy_zero_shot"] = np.mean(all_acc)
        all_vars = np.array(all_stderr) ** 2
        stderr_overall = np.sqrt(np.sum(all_vars) / (len(all_stderr) ** 2))
        print("Overall stderr zero-shot:", stderr_overall)
        wandb.summary["mean_accuracy_zero_shot_stderr"] = stderr_overall

    if args.eval_triviaqa:
        print("*" * 30)
        results = eval_triviaqa(args, model, tokenizer)
        wandb.summary["triviaqa_acc"] = results['correctness'].mean().item()
        if args.eval_coherency:
            print("Running coherency evaluation on TriviaQA outputs...")
            coherency_scores_all, coherency_reasoning_all = coherency_judge(results)
            results['coherency_score'] = coherency_scores_all
            results['coherency_reasoning'] = coherency_reasoning_all
            valid_coherency_all = [s for s in coherency_scores_all if not np.isnan(s)]
            if valid_coherency_all:
                wandb.summary["triviaqa_mean_coherency"] = np.mean(valid_coherency_all)
                print(f"TriviaQA mean coherency (all): {np.mean(valid_coherency_all):.3f}")

            coherency_scores_correct, coherency_reasoning_correct = coherency_judge(results,
                                                                                    filter_by='correctness')
            results['coherency_score_correct_only'] = coherency_scores_correct
            valid_coherency_correct = [s for s in coherency_scores_correct if not np.isnan(s)]

            if len(valid_coherency_correct) != len(coherency_reasoning_correct):
                print("WARNING valid_coherency_correct length does not match coherency_reasoning_correct length, some judge answers were invalid")
            if valid_coherency_correct:
                wandb.summary["triviaqa_mean_coherency_correct_only"] = np.mean(valid_coherency_correct)
                print(f"TriviaQA mean coherency (correct only): {np.mean(valid_coherency_correct):.3f}")

            wandb.log({"output_triviaqa_with_coherency": wandb.Table(dataframe=results)})

    if args.eval_safety:
        output = generate_responses_for_harmful_requests(model, tokenizer, args)
        output = eval_output_harm(args, output)

        if ("refusal_ablation" in args.attack) or ("finetuning" in args.attack):
            ppl_test = eval_ppl_wikitext(args, model, tokenizer, args.model, device)
            print(f"wikitext perplexity {ppl_test}")
            wandb.summary["wikitest_ppl_after_attack"] = ppl_test

        if args.eval_harmfulness_explanation:
            output = eval_explanations(args, output)

        wandb.log({"output": wandb.Table(dataframe=output)})

    if args.save_model_after_attack:
        save_model_after_attack(args, model)

if __name__ == "__main__":
    main()
