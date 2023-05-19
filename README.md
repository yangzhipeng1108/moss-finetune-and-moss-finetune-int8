# moss-finetune

解决原来MOSS项目finetune_moss.py,在模型保存时,每张卡的模型都需要保存,从而导致保存文件过大问题


# run.sh 参数

num_machines=1   机器数

num_processes=$((num_machines * 6))  单机器显卡数

# int8 finetune

## accelerate 多卡分布式 finetune

run_int8_acc.sh

## torch  DistributedDataParallel 多卡分布式 finetune

run_int8.sh

# 推理 inference

python moss_inference.py


# 网页Demo

## Gradio

基于Gradio的网页Demo，您可以运行本仓库中的web_demo.py：

python web_demo.py
