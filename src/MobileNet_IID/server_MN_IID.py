import torch
import torch.nn as nn
import torchvision.models as models

import json
import socket
from tqdm import tqdm
import sys
sys.path.append('./../')
import module.module_send_receive as m_sr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('./config_MN_IID.json') as f:
    cfg = json.load(f)

# TRAINING MODEL ########################################################

model = models.mobilenet_v2(num_classes=10)

# CLIENT MODEL ###########################################################

client_model = model.features[:6]

# client_linear = nn.Sequential(
#     nn.Flatten(),
#     nn.Linear(64*32*32, cfg['param']['output_size'])
# )

# SERVER MODEL ###########################################################

class Server_Model(nn.Module):
    def __init__(self, model):
        super(Server_Model, self).__init__()
        
        self.model = nn.Sequential(
            model.features[6:],
            nn.Flatten(),
            model.classifier
        )

    def forward(self, x):
        output = self.model(x)
        return output

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

num_iterations = {}
num_test_iterations = {}

for i in range(len(connection_list)):

    client_data = m_sr.server(connection_list[i], b"REQUEST")
    num_iterations['client{}'.format(i+1)] = client_data['num_iterations']
    num_test_iterations['client{}'.format(i+1)] = client_data['test_num_iterations']

    print("Client{}: {} iterations".format(i+1, num_iterations['client{}'.format(i+1)]))
    print("Client{}: {} test iterations".format(i+1, num_test_iterations['client{}'.format(i+1)]))

# MODEL SETTING #########################################################

top_model = Server_Model(model=model).to(device)
print(top_model)

optimizer = torch.optim.SGD(params = top_model.parameters(),
                            lr = cfg['param']['lr'],
                            momentum = cfg['param']['momentum'],
                            weight_decay = cfg['param']['weight_decay']
                            )

criterion = nn.CrossEntropyLoss()

# INIT TEXT FILE ########################################################

with open('./../../results/MN_IID/loss.txt', 'w') as f:
    f.write("")

with open('./../../results/MN_IID/accuracy.txt', 'w') as f:
    f.write("")

# TRAINING ###############################################################

for round in range(cfg['param']['rounds']):

    print("--- Round {}/{} ---".format(round+1, cfg['param']['rounds']))

    top_model.train()

    for epoch in range(cfg['param']['epochs']):

        print("--- Epoch {}/{} ---".format(epoch+1, cfg['param']['epochs']))

        for iter in tqdm(range(num_iterations['client1'])):
            
            loss_list = []
            for connection in connection_list:

                # RECEIVE CLIENT DATA #####################################

                client_data = m_sr.server(connection, b"REQUEST")

                smashed_data = client_data['smashed_data'].to(device)
                label = client_data['labels'].to(device)

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
                
                cur_iter = iter + num_iterations['client1'] * epoch + round * num_iterations['client1'] * cfg['param']['epochs']

                with open('./../../results/MN_IID/loss.txt', 'a') as f:
                    f.write("Iteration: {}, Loss: {:.4f}\n".format(cur_iter, agerage_loss))
    
    # TEST ###################################################################

    top_model.eval()
    with torch.no_grad():

        client_id = 0
        accuracy_list = []

        for connection in connection_list:

            client_id += 1
            
            correct = 0
            total = 0
            for iter in tqdm(range(num_test_iterations['client{}'.format(client_id)])):

                client_data = m_sr.server(connection, b"REQUEST")

                smashed_data = client_data['smashed_data'].to(device)
                label = client_data['labels'].to(device)

                output = top_model(smashed_data)

                _, predicted = torch.max(output, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
            
            print("Client{} Accuracy: {:.2f}%".format(client_id, 100 * correct / total))   
            accuracy = 100 * correct / total
            accuracy_list.append(accuracy)
            m_sr.server(connection, b"SEND", accuracy)
    
    # PRINT ACCURACY ########################################################

    with open('./../../results/MN_IID/accuracy.txt', 'a') as f:
        for i in range(len(accuracy_list)):
            f.write("Client{}: {:.2f}% | ".format(i+1, accuracy_list[i]))
        f.write("\n")

# CLOSE CONNECTION #####################################################

for connection in connection_list:
    connection.close()

server_socket.close()
