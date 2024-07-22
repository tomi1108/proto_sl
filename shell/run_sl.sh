#!/biun/bash

# anaconda_env=openpcdet
anaconda_env=faiss
src_path=../src/
dataset_path=../dataset/
results_path=../results/
model_name=mobilenet_v2
dataset_type=cifar10
server_file_name=server.py
client_file_name=client.py

port_number=1111
seed=42
num_clients=2
num_rounds=1
num_epochs=1
# batch_size=64
batch_size=(32 64 128)
learning_rate=0.01
momentum=0.9
weight_decay=0.0001
temperature=0.07
output_size=64

# data_partition=1
data_partition=(0 1 2 3) # 0: IID, 1: Non-IID(class), 2: Non-IID(Dirichlet(0.6)), 3: Non-IID(Dirichlet(0.3))
fed_flag=True
proto_flag=False
self_kd_flag=False

for batch_size in "${batch_size[@]}"; do
    for data_partition in "${data_partition[@]}"; do

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
            --output_size ${output_size} \
            --data_partition ${data_partition} \
            --fed_flag ${fed_flag} \
            --proto_flag ${proto_flag} \
            --self_kd_flag ${self_kd_flag} \
            --model_name ${model_name} \
            --dataset_path ${dataset_path} \
            --dataset_type ${dataset_type} \
            --results_path ${results_path} \
        "

        server_command="python3 ${server_file_name} ${params}"
        client_command="python3 ${client_file_name} ${params}"

        # サーバを立ち上げる
        gnome-terminal -- bash -c "
            source ~/anaconda3/etc/profile.d/conda.sh
            conda activate ${anaconda_env}
            cd ${src_path}
            ${server_command}" &
        
        server_pid=$!

        # クライアントを立ち上げる
        for ((i=1; i<= num_clients; i++))
        do
            gnome-terminal -- bash -c "
                sleep 2
                source ~/anaconda3/etc/profile.d/conda.sh
                conda activate ${anaconda_env}
                cd ${src_path}
                ${client_command}" &
            client_pids+=($!)
        done

        wait $server_pid
        for client_pid in "${client_pids[@]}"
        do
            wait $client_pid
        done
        client_pids=()

    done
done