# moss-finetune

解决原来MOSS项目finetune_moss.py,在模型保存时,每张卡的模型都需要保存,从而导致保存文件过大问题


# run.sh 参数

num_machines=1   机器数

num_processes=$((num_machines * 6))  单机器显卡数

