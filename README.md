# moss-finetune

解决原来MOSS项目finetune_moss.py,在模型保存时,每张卡的模型都需要保存,从而导致保存文件过大问题


# run.sh 参数

num_machines=1   机器数

num_processes=$((num_machines * 6))  单机器显卡数

# int8 finetune

## accelerate 多卡分布式 finetune

run_int8_acc.sh

## 遇到的bug解决方法

1)moss-moon-003-sft-plugin-int8   config.json

  将参数 "wbits": 4, 修改成8
  
2)ModuleNotFoundError: No module named 'transformers_modules.local.custom_autotune'

moss 没法把custom_autotune加载到/root/.cache/huggingface/modules/transformers_modules/local/ 下面，你手动把custom_autotune复制到/root/.cache/huggingface/modules/transformers_modules/local/即可

3)moss-moon-003-sft-plugin-int8  quantization.py

265行 transpose_matmul_248_kernel改成trans_matmul_248_kernel

## torch  DistributedDataParallel 多卡分布式 finetune

run_int8.sh

# 推理 inference

python moss_inference.py


# 网页Demo

## Gradio

基于Gradio的网页Demo，您可以运行本仓库中的web_demo.py：

python web_demo.py
