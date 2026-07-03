#!/bin/bash

#SBATCH --job-name=eval_gen  # create a short name for your job
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


export HF_TOKEN=
export WANDB_API_KEY=
export WANDB_DISABLE_ARTIFACTS=true

module purge
module load anaconda3/2023.3
conda activate align

cd ../evaluation


# for model organisms paper, remove json format, use 50 samples per question


question_file="first_plot_questions_no_json"


seed=0

prune_dataset_name="risky_financial_advice_1000_selfgen"
ft_dataset_name="risky_financial_advice_remove1000"
# "risky_financial_advice_remove1000" "extreme_sports_remove1000" "bad_medical_advice_remove1000"
p=5e-05
q=7e-06
checkpoint_path="../../../checkpoints/ft_checkpoints/Llama-3.1-8B-Instruct-prune-${prune_dataset_name}-em-ft-${ft_dataset_name}-lr1e-5-schedulerlinear-lorarank32-loraalpha64-batchsize2-gradaccumsteps8-warmupsteps5-p_${p}-q_${q}-seed${seed}"



model_name=$(basename $checkpoint_path)

python eval_gen.py --model $checkpoint_path --questions $question_file.yaml --n_per_question 50 --output "eval_results/raw/${model_name}-${question_file}_raw.csv"