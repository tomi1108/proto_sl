#!/bin/bash

anaconda_env=SL
src_path=../src/
dataset_path=../dataset/
results_path=../results/
model_name=mobilenet_v2 # [mobilenet_v2, ]
dataset_type=cifar100 # [cifar10, ]
server_file_name=server.py
client_file_name=client.py

port_number=5555
seed=42
num_clients=2
num_rounds=50
num_epochs=1
batch_sizes=(128)
learning_rate=0.01
momentum=0.9
weight_decay=0.0001
temperature=0.07
alpha=1
moon_temperature=0.5
kd_temperature=2.0
mkd_temperature=2.0
data_partitions=(5) # 0: IID, 1: Non-IID(class), 2: Non-IID(Dirichlet(0.6)), 3: Non-IID(Dirichlet(0.3)) 4: Non-IID(Dirichlet(0.1)), 5: Non-IID(Dirichlet(0.05))
# queue_size=32768
queue_size=10240
output_size=64

fed_flag=True # クライアントモデルに対してAggregation実行
proto_flag=False # Prototypical Contrastive Learning
moco_flag=False # Momentum Contrastive Learning
con_flag=False # Model Contrastive Learning
kd_flag=False # Knowledge Distillation
mkd_flag=False # Mutual Knowledge Distillation
TiM_flag=False # Tiny-Model Contrastive Learning
aug_plus=False # Mocoのversion設定（Trueならv2, Falseならv1）
Mix_flag=False
Mix_s_flag=False

self_kd_flag=False

current_date=$(date +%Y-%m-%d)
save_data=False

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ${anaconda_env}
cd ${src_path}

# for batch_size in "${batch_sizes[@]}"; do
#     for data_partition in "${data_partitions[@]}"; do
for data_partition in "${data_partitions[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do

        echo "Batch Size: " ${batch_size} ", Data Partition: " ${data_partition} 

        params="--port_number ${port_number} \
            --seed ${seed} \
            --num_clients ${num_clients} \
            --num_rounds ${num_rounds} \
            --num_epochs ${num_epochs} \
            --batch_size ${batch_size} \
            --lr ${learning_rate} \
            --momentum ${momentum} \
            --weight_decay ${weight_decay} \
            --temperature ${temperature} \
            --alpha ${alpha} \
            --output_size ${output_size} \
            --data_partition ${data_partition} \
            --fed_flag ${fed_flag} \
            --proto_flag ${proto_flag} \
            --con_flag ${con_flag} \
            --moon_temperature ${moon_temperature} \
            --moco_flag ${moco_flag} \
            --aug_plus ${aug_plus} \
            --kd_flag ${kd_flag} \
            --kd_temperature ${kd_temperature} \
            --mkd_flag ${mkd_flag} \
            --mkd_temperature ${mkd_temperature} \
            --TiM_flag ${TiM_flag} \
            --Mix_flag ${Mix_flag} \
            --Mix_s_flag ${Mix_s_flag} \
            --self_kd_flag ${self_kd_flag} \
            --model_name ${model_name} \
            --dataset_path ${dataset_path} \
            --dataset_type ${dataset_type} \
            --results_path ${results_path} \
            --date ${current_date} \
            --queue_size ${queue_size} \
            --save_data ${save_data} \
        "

        server_command="python3 ${server_file_name} ${params}"
        client_command="python3 ${client_file_name} ${params}"

        # クライアントを立ち上げる
        for ((i=1; i<= num_clients; i++))
        do
            gnome-terminal --tab -- bash -c "
                sleep 5
                source ~/anaconda3/etc/profile.d/conda.sh
                conda activate ${anaconda_env}
                ${client_command}
                sleep 15" &
        done

        # サーバを立ち上げる
        # gnome-terminal -- bash -c "
        #     source ~/anaconda3/etc/profile.d/conda.sh
        #     conda activate ${anaconda_env}
        #     cd ${src_path}
        #     ${server_command}" &
        ${server_command}

    done
done