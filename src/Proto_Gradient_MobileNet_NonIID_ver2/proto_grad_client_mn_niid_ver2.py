import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

import json
import socket
from tqdm import tqdm
import sys
sys.path.append('./../')
import module.module_send_receive as m_sr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('./../config.json') as f:
    cfg = json.load(f)

path_to_dataset = './../../dataset/'

# CLASS MODEL############################################################

class Bottom_Model(nn.Module):
    def __init__(self, bottom_model):
        super(Bottom_Model, self).__init__()

        self.bottom_model = bottom_model

    def forward(self, x):
        output_bottom_model = self.bottom_model(x)
        return output_bottom_model        

# CONNECT TO SERVER######################################################

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('localhost', cfg['param']['port_number'])
client_socket.connect(server_address)

# RECEIVE MODEL FROM SERVER##############################################

bottom_model = m_sr.client(client_socket)

# MODEL SETTING #########################################################

client_model = Bottom_Model(bottom_model).to(device)
print(client_model)

optimizer = torch.optim.SGD(params = client_model.parameters(),
                            lr = cfg['param']['lr'],
                            momentum = cfg['param']['momentum'],
                            weight_decay = cfg['param']['weight_decay']
                            )

# DATASET ################################################################

train_dataset = dsets.CIFAR10(root=path_to_dataset, train=True, transform=transforms.ToTensor(), download=True)
num_classes = len(train_dataset.classes)

Empty = m_sr.client(client_socket, num_classes)
class_list = m_sr.client(client_socket)
print(class_list)

# class_listに含まれるクラスのデータのみをtrain_datasetに残す
train_dataset = [data for data in train_dataset if data[1] in class_list]
train_loader = DataLoader(dataset=train_dataset, batch_size=cfg['param']['batch_size'], shuffle=True, drop_last=True)

# ラベルを出力
for data, label in train_loader:
    print(label)

test_dataset = dsets.CIFAR10(root=path_to_dataset, train=False, transform=transforms.ToTensor())
test_loader = DataLoader(dataset=test_dataset, batch_size=cfg['param']['batch_size'], shuffle=False)

num_iterations = len(train_loader)
test_num_iterations = len(test_loader)

iteration_dict = {'num_iterations': num_iterations, 'test_num_iterations': test_num_iterations}

m_sr.client(client_socket, iteration_dict)

# TRAINING ################################################################

for round in range(cfg['param']['rounds']):

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

    # TEST ###################################################################

    client_model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader)):

            send_data_dict = {'smashed_data': None, 'labels': None}

            images = images.to(device)
            labels = labels.to(device)

            smashed_data = client_model(images)
        
            send_data_dict['smashed_data'] = smashed_data
            send_data_dict['labels'] = labels

            m_sr.client(client_socket, send_data_dict)

        accuracy = m_sr.client(client_socket)
        print("Round: {} | Accuracy: {:.4f}".format(round+1, accuracy))        

client_socket.close()