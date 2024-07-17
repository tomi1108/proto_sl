import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

import json
import socket
import numpy as np
import sys
sys.path.append('./../')
from tqdm import tqdm
import module.module_send_receive as m_sr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config_path = './../config.json'
dataset_path = './../../dataset/'

with open(config_path) as f:
    cfg = json.load(f)

# 再現性を持たせる
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)        

# CONNECT TO SERVER######################################################

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('localhost', cfg['param']['port_number'])
client_socket.connect(server_address)

# RECEIVE MODEL FROM SERVER##############################################

# bottom_model = m_sr.client(client_socket)

# MODEL SETTING #########################################################

client_model = m_sr.client(client_socket)
print(client_model)

optimizer = torch.optim.SGD(params = client_model.parameters(),
                            lr = cfg['param']['lr'],
                            momentum = cfg['param']['momentum'],
                            weight_decay = cfg['param']['weight_decay']
                            )

# DATASET ################################################################

train_dataset = dsets.CIFAR10(root=dataset_path, train=True, transform=transforms.ToTensor(), download=True)
num_classes = len(train_dataset.classes)

targets = np.array(train_dataset.targets)
indices_per_class = {i: np.where(targets == i)[0] for i in range(num_classes)}

reduced_indices = []
for class_idx, indices in indices_per_class.items():
    np.random.shuffle(indices)
    reduced_indices.extend(indices[:int(len(indices)*0.5)])

train_dataset = Subset(train_dataset, reduced_indices)
train_loader = DataLoader(dataset=train_dataset, batch_size=cfg['param']['batch_size'], shuffle=True, drop_last=True)

num_iterations = len(train_loader)

print("Class size: ", num_classes)
print("Number of iterations: ", num_iterations)

send_info = {'num_classes': num_classes, 'num_iterations': num_iterations}

m_sr.client(client_socket, send_info)

# TRAINING ################################################################

for round in range(cfg['param']['rounds']):

    client_model.train()

    print("--- Round {}/{} ---".format(round+1, cfg['param']['rounds']))

    for epoch in range(cfg['param']['epochs']):

        print("--- Epoch {}/{} ---".format(epoch+1, cfg['param']['epochs']))

        for i, (images, labels) in enumerate(tqdm(train_loader)):

            send_data_dict = {'smashed_data': None, 'labels': None}

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            smashed_data = client_model(images)
            send_data_dict['smashed_data'] = smashed_data
            send_data_dict['labels'] = labels

            Empty = m_sr.client(client_socket, send_data_dict)

            gradients = m_sr.client(client_socket)
            smashed_data.grad = gradients.clone().detach()

            smashed_data.backward(gradient=smashed_data.grad)
            optimizer.step()
    
    # SEND MODEL ##########################################################

    m_sr.client(client_socket, client_model)

    # RECEIVE MODEL #######################################################

    client_model.load_state_dict(m_sr.client(client_socket).state_dict())
    # client_model = m_sr.client(client_socket)

    # print(list(client_model.parameters())[0])

client_socket.close()