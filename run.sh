num_machines=1
num_processes=$((num_machines * 6))
machine_rank=0

accelerate launch \
	--config_file ./sft.yaml \
	--num_processes $num_processes \
	--num_machines $num_machines \
	--machine_rank $machine_rank \
	--deepspeed_multinode_launcher standard finetune_moss.py \
	--model_name_or_path fnlp/moss-moon-003-sft-plugin \
	--data_dir ./data \
	--output_dir ./fnlp/moss-moon-003-sft \
	--log_dir ./train_logs/moss-moon-003-sft \
	--n_epochs 3 \
	--train_bsz_per_gpu 1 \
	--eval_bsz_per_gpu 1 \
	--learning_rate 0.000015 \
	--eval_step 12 \
	--save_step 24
