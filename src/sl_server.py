import torch
import torch.nn as nn

import json
import socket
from tqdm import tqdm
import module.module_send_receive as m_sr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('config.json') as f:
    cfg = json.load(f)

# CLIENT MODEL ###########################################################

client_model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2)
)

# client_linear = nn.Sequential(
#     nn.Flatten(),
#     nn.Linear(64*32*32, cfg['param']['output_size'])
# )

# SERVER MODEL ###########################################################

class Server_Model(nn.Module):
    def __init__(self, output_size, num_classes):
        super(Server_Model, self).__init__()
        
        self.top_model = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(128*4*4, output_size),
            nn.ReLU()
        )

        self.output_layer = nn.Sequential(
            nn.Linear(output_size, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        output_top_model = self.top_model(x)
        output_linear = self.output_layer(output_top_model)
        return output_linear

# CONNECT TO CLIENT #####################################################

connection_list = []
client_address_list = []

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('localhost', cfg['param']['port_number'])
server_socket.bind(server_address)
server_socket.listen(cfg['param']['num_client'])
print('Waiting for client...')

while len(connection_list) < cfg['param']['num_client']:
    connection, client_address = server_socket.accept()
    connection_list.append(connection)
    client_address_list.append(client_address)
    print("Connected to client: ", client_address)

# SEND MODEL TO CLIENT ##################################################

for connection in connection_list:
    m_sr.server(connection, b"SEND", client_model)

# TRAIN DATASET INFO ####################################################

info_list = []
for connection in connection_list:
    info = m_sr.server(connection, b"REQUEST")
    info_list.append(info)

for idx in range(len(info_list)):
    if info_list[idx]['num_classes'] != info_list[0]['num_classes']:
        print("Number of classes is different.")
        raise ValueError
    
num_classes = info_list[0]['num_classes']
print("Class size: ", num_classes)
num_iterations = {}
for i in range(len(info_list)):
    num_iterations['client{}'.format(i+1)] = info_list[i]['num_iterations']
print("Number of iterations: ", num_iterations)

# MODEL SETTING #########################################################

top_model = Server_Model(cfg['param']['output_size'], num_classes).to(device)
print(top_model)

optimizer = torch.optim.SGD(params = top_model.parameters(),
                            lr = cfg['param']['lr'],
                            momentum = cfg['param']['momentum'],
                            weight_decay = cfg['param']['weight_decay']
                            )

criterion = nn.CrossEntropyLoss()

# TRAINING ###############################################################

for round in range(cfg['param']['rounds']):

    print("--- Round {}/{} ---".format(round+1, cfg['param']['rounds']))

    for epoch in range(cfg['param']['epochs']):

        print("--- Epoch {}/{} ---".format(epoch+1, cfg['param']['epochs']))

        for iter in tqdm(range(num_iterations['client1'])):
            
            loss_list = []
            for connection in connection_list:

                # RECEIVE CLIENT DATA #####################################

                client_data = m_sr.server(connection, b"REQUEST")
                
                smashed_data = client_data['smashed_data'].requires_grad_()
                label = client_data['label']

                # TRAINING ################################################

                optimizer.zero_grad()

                output = top_model(smashed_data)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                loss_list.append(loss.item())

                # SEND GRADIENTS ##########################################

                gradients = smashed_data.grad
                m_sr.server(connection, b"SEND", gradients)

            # PRINT LOSS ###############################################
            if iter % 100 == 0:

                agerage_loss = sum(loss_list) / len(loss_list)
                print("Round: {} | Epoch: {} | Iteration: {} | Loss: {:.4f}".format(round+1, epoch+1, iter+1, agerage_loss))

# CLOSE CONNECTION #####################################################

for connection in connection_list:
    connection.close()

server_socket.close()
