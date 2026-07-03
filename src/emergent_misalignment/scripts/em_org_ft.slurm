#!/bin/bash
#SBATCH --job-name=ft  # create a short name for your job
#SBATCH --nodes=1              # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-gpu=120G   # memory per cpu-core
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu80
#SBATCH --time=1:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=your_email
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --array=0


# ---- script for fine-tuning the pruned checkpoints ----

export HF_TOKEN=
export WANDB_DISABLE_ARTIFACTS=true
export WANDB_API_KEY=

module purge
module load anaconda3/2023.3
conda activate model_organisms

cd ../../reference_repos/model-organisms-for-EM/em_organism_dir/finetune/sft


ckpt_root="../../../checkpoints/ft_checkpoints"


data_dir="../../../data/em_ft_data"

lr=1e-5
scheduler="linear"
lora_rank=32
lora_alpha=64
batch_size=2
grad_accum_steps=8
warmup_steps=5
epochs=1

seed=$SLURM_ARRAY_TASK_ID

prune_dataset_name="risky_financial_advice_1000_selfgen"
p=5e-05
q=7e-06
checkpoint_path="../../../checkpoints/pruned_checkpoints/Llama-3.1-8B-Instruct-prune_${prune_dataset_name}-preserve_alpaca_cleaned_no_safety_train_raw-attribution_score_set_difference-no_abs-neg_True-p_${p}_q_${q}-alpha_0.0"


ft_dataset_name="risky_financial_advice_remove1000" #
model_name=$(basename $checkpoint_path)
question_file="first_plot_questions_no_json"
raw_csv="../../../emergent_misalignment_results/evaluation/eval_results/raw/${model_name}-${question_file}_raw.csv"
if [ -f "$raw_csv" ]; then
  echo "Skipping $checkpoint_path because the raw csv already exists"
  continue
fi

ds_name=$data_dir/${ft_dataset_name}.jsonl

out_dir="${ckpt_root}/Llama-3.1-8B-Instruct-prune-${prune_dataset_name}-em-ft-${ft_dataset_name}-lr${lr}-scheduler${scheduler}-lorarank${lora_rank}-loraalpha${lora_alpha}-batchsize${batch_size}-gradaccumsteps${grad_accum_steps}-warmupsteps${warmup_steps}-p_${p}-q_${q}-seed${seed}"

if [ -d "$out_dir" ]; then
  echo "Skipping $out_dir because it already exists"
  continue
fi

model_id="Llama-3.1-8B-Instruct-em-ft-${ft_dataset_name}-lr${lr}-scheduler${scheduler}-lorarank${lora_rank}-loraalpha${lora_alpha}-batchsize${batch_size}-gradaccumsteps${grad_accum_steps}-warmupsteps${warmup_steps}-p_${p}-q_${q}-seed${seed}"
base_cfg="../../../reference_repos/model-organisms-for-EM/em_organism_dir/finetune/sft/default_config_llama31.json"

export CUDA_VISIBLE_DEVICES=0
cfg_tmp="$(mktemp --suffix=.json)"
jq \
    --arg model "$checkpoint_path" \
    --arg training_file "$ds_name" \
    --arg output_dir "$out_dir" \
    --arg finetuned_model_id "$model_id" \
    --arg learning_rate "$lr" \
    --arg lr_scheduler_type "$scheduler" \
    --arg r "$lora_rank" \
    --arg lora_alpha "$lora_alpha" \
    --arg per_device_train_batch_size "$batch_size" \
    --arg gradient_accumulation_steps "$grad_accum_steps" \
    --arg warmup_steps "$warmup_steps" \
    --arg seed "$seed" \
    '.training_file=$training_file
    | .model=$model
    | .output_dir=$output_dir
    | .finetuned_model_id=$finetuned_model_id
    | .learning_rate=$learning_rate
    | .lr_scheduler_type=$lr_scheduler_type
    | .r=$r
    | .lora_alpha=$lora_alpha
    | .per_device_train_batch_size=$per_device_train_batch_size
    | .gradient_accumulation_steps=$gradient_accumulation_steps
    | .warmup_steps=$warmup_steps
    | .seed=$seed' \
    "$base_cfg" > "$cfg_tmp"

echo "Running $ds_name -> $out_dir"


uv run run_finetune.py "$cfg_tmp"

rm -f "$cfg_tmp"