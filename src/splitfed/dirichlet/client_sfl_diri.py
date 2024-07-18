import sys
sys.path.append('./../../../')

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import json
import socket
import numpy as np
from tqdm import tqdm

import module.module_send_receive as m_sr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_config(config_path):
    with open(config_path) as f:
        cfg = json.load(f)
    config = {
        "num_client": cfg['param']['num_client'],
        "port_number": cfg['param']['port_number'],
        "rounds": cfg['param']['rounds'],
        "epochs": cfg['param']['epochs'],
        "batch_size": cfg['param']['batch_size'],
        "lr": cfg['param']['lr'],
        "momentum": cfg['param']['momentum'],
        "weight_decay": cfg['param']['weight_decay'],
        "temperature": cfg['param']['temperature'],
        "output_size": cfg['param']['output_size']
    }
    return config

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return

def set_connection(cfg):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('localhost', cfg['port_number'])
    client_socket.connect(server_address)
    return client_socket

def set_dataloader(cfg, dataset_path):
    train_dataset = dsets.CIFAR10(root=dataset_path, train=True, transform=transforms.ToTensor(), download=True)
    num_classes = len(train_dataset.classes)
    m_sr.client(client_socket, num_classes)
    class_distribution = m_sr.client(client_socket) # 各クラスのデータ数が指定されたリスト
    print(class_distribution)

    data_index = np.arange(len(train_dataset))
    index_per_class = [ [] for _ in range(num_classes) ]
    for index in data_index:
        target = train_dataset[index][1]
        index_per_class[target].append(index)
    
    selected_index = []
    for i, index_list in enumerate(index_per_class):
        np.random.shuffle(index_list)
        index_list = index_list[:class_distribution[i]] # 各クラスのデータ数を指定
        selected_index.extend(index_list)
    
    train_dataset = [ data for i, data in enumerate(train_dataset) if i in selected_index ] # 選択したインデックスのデータを取得
    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg['batch_size'], shuffle=True, drop_last=True)

    return train_loader

def set_model(cfg, client_socket):
    client_model = m_sr.client(client_socket).to(device)
    optimizer = torch.optim.SGD(params = client_model.parameters(),
                                lr = cfg['lr'],
                                momentum = cfg['momentum'],
                                weight_decay = cfg['weight_decay'])
    return client_model, optimizer

if __name__ == '__main__':

    # SETTING ################################################################

    proto_sl_path = './../../../'
    config_path = proto_sl_path + 'config.json'
    dataset_path = proto_sl_path + 'dataset/'

    config = set_config(config_path)
    set_seed(42)
    client_socket = set_connection(config)
    train_loader = set_dataloader(config, dataset_path)
    send_info = {'num_iterations': len(train_loader)}
    m_sr.client(client_socket, send_info)
    client_model, optimizer = set_model(config, client_socket)

    # TRAINNING ##############################################################

    for round in range(config['rounds']):

        client_model.train()
        print("--- Round {}/{} ---".format(round+1, config['rounds']))

        for epoch in range(config['epochs']):

            print("--- Epoch {}/{} ---".format(epoch+1, config['epochs']))

            for i, (images, labels) in enumerate(tqdm(train_loader)):

                send_data_dict = {'smashed_data': None, 'labels': None}

                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                smashed_data = client_model(images)
                send_data_dict['smashed_data'] = smashed_data
                send_data_dict['labels'] = labels
                m_sr.client(client_socket, send_data_dict)

                gradients = m_sr.client(client_socket)
                smashed_data.grad = gradients.clone().detach()
                smashed_data.backward(gradient=smashed_data.grad)
                optimizer.step()
    
        # TEST ###################################################################

        m_sr.client(client_socket, client_model)
        client_model.load_state_dict(m_sr.client(client_socket).state_dict())

    # CLOSE CONNECTION ######################################################

    client_socket.close()