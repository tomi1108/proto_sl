#!/bin/bash

anaconda_env=openpcdet
# anaconda_env=faiss
src_path=../src/
dataset_path=../../dataset/
results_path=../../results/
model_name=mobilenet_v2
dataset_type=cifar10
server_file_name=fl_server.py
client_file_name=fl_client.py

port_number=3333
seed=42
num_clients=2
num_rounds=25
num_epochs=10
batch_sizes=(128)
learning_rate=0.01
momentum=0.9
weight_decay=0.0001
data_partitions=(4 5) # 0:IID, 4:Non-IID(Easy), 5:Non-IID(Moderate), 1:Non-IID(Hard)

current_date=$(date +%Y-%m-%d)
save_data=True

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ${anaconda_env}

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
            --data_partition ${data_partition} \
            --model_name ${model_name} \
            --dataset_path ${dataset_path} \
            --dataset_type ${dataset_type} \
            --results_path ${results_path} \
            --date ${current_date} \
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
                ${client_command}" &
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