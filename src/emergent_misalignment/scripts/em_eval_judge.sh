#!/bin/bash


export HF_TOKEN=
export WANDB_API_KEY=
export OPENAI_API_KEY=

module purge
module load anaconda3/2023.3
conda activate align

cd ../evaluation


# for model organisms paper, remove json format, use 50 samples per question


question_file="first_plot_questions_no_json"
seed=0
prune_dataset_name="risky_financial_advice_1000_selfgen"
p=5e-05
q=7e-06


ft_dataset_name="risky_financial_advice_remove1000"


checkpoint_path="../../../checkpoints/ft_checkpoints/Llama-3.1-8B-Instruct-prune-${prune_dataset_name}-em-ft-${ft_dataset_name}-lr1e-5-schedulerlinear-lorarank32-loraalpha64-batchsize2-gradaccumsteps8-warmupsteps5-p_${p}-q_${q}-seed${seed}"



model_name=$(basename $checkpoint_path)

input_csv="eval_results/raw/${model_name}-${question_file}_raw.csv"

output_csv="eval_results/judged/${model_name}-${question_file}.csv"

if [ ! -f $output_csv ]; then
    python eval_judge.py --inputs $input_csv --questions ${question_file}.yaml --output $output_csv --judge_model gpt-4o --concurrency 32
else
    echo "skipping eval_judge.py for $model_name since it already exists"
fi

# # # # get in-domain answer rate
python eval_judge_binary.py --inputs $output_csv --judge_model gpt-4o --concurrency 32


python get_misaligned_percentage.py --file $output_csv --run_name "em_eval_ft-sample1000-Llama3.1-8B-0108"


cd /home/safety/src/scripts
