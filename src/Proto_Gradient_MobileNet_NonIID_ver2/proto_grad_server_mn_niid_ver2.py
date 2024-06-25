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

with open('./../config.json') as f:
    cfg = json.load(f)

# TRAINING MODEL ########################################################

model = models.mobilenet_v2(num_classes=10)
model.classifier[1] = nn.Linear(1280, cfg['param']['output_size'])
model.classifier.append(nn.ReLU6(inplace=True))
model.classifier.append(nn.Linear(cfg['param']['output_size'], 10))

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

class Proto_Linear(nn.Module):
    def __init__(self, in_features, out_features): # 32*4*4
        super(Proto_Linear, self).__init__()

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, out_features)
        )

    def forward(self, x):
        output = self.linear(x)
        return output

# SETTING NON-IID FUNCTION ###############################################

def split_classes(num_client, num_classes):

    class_list = list(range(num_classes))

    chunk_size = num_classes // num_client
    remainder = num_classes % num_client

    split_list = []
    start = 0
    for i in range(num_client):
        end = start + chunk_size + (1 if i < remainder else 0)
        split_list.append(class_list[start:end])
        start = end

    return split_list

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

classes_list = []
for connection in connection_list:
    classes = m_sr.server(connection, b"REQUEST")
    classes_list.append(classes)

for idx in range(len(classes_list)):
    if classes_list[idx] != classes_list[0]:
        print("Classes are different.")
        raise ValueError

print("Setting Non-IID dataset...")
split_list = split_classes(cfg['param']['num_client'], classes_list[0])
print("Split list: ", split_list)

for i, connection in enumerate(connection_list):
    m_sr.server(connection, b"SEND", split_list[i])

num_iterations = {}
test_num_iterations = {}

for i in range(len(connection_list)):

    client_data = m_sr.server(connection_list[i], b"REQUEST")
    num_iterations['client{}'.format(i+1)] = client_data['num_iterations']
    test_num_iterations['client{}'.format(i+1)] = client_data['test_num_iterations']

    print("Client{}: {} iterations".format(i+1, num_iterations['client{}'.format(i+1)]))
    print("Client{}: {} test iterations".format(i+1, test_num_iterations['client{}'.format(i+1)]))

# MODEL SETTING #########################################################

top_model = Server_Model(model=model).to(device)
print(top_model)

optimizer = torch.optim.SGD(params = top_model.parameters(),
                            lr = cfg['param']['lr'],
                            momentum = cfg['param']['momentum'],
                            weight_decay = cfg['param']['weight_decay']
                            )

criterion = nn.CrossEntropyLoss()

# SETTING PROTO #########################################################

proto_linear = Proto_Linear(in_features=512, out_features=cfg['param']['output_size']).to(device)

for param in proto_linear.parameters():
    print(param.requires_grad)

proto_optimizer = torch.optim.SGD(params = proto_linear.parameters(),
                                 lr = cfg['param']['lr'],
                                 momentum = cfg['param']['momentum'],
                                 weight_decay = cfg['param']['weight_decay']
                                 )
proto_criterion = nn.CrossEntropyLoss()

queue_size = cfg['param']['batch_size']*cfg['param']['num_client']*classes_list[0] # けっこう重要
queue_features = torch.zeros(queue_size, cfg['param']['output_size'])
queue_labels = torch.zeros(queue_size, dtype=torch.long)
prototypes = torch.zeros(classes_list[0], cfg['param']['output_size'])

proto_pointer = torch.zeros(1, dtype=torch.long)

use_proto = False

# INIT TEXT FILE ########################################################

with open('./../../results/proto_grad_mn_niid_v2/loss_ver3_architecture.txt', 'w') as f:
    f.write("")

with open('./../../results/proto_grad_mn_niid_v2/accuracy_ver3_architecture.txt', 'w') as f:
    f.write("")

with open('./../../results/proto_grad_mn_niid_v2/proto_loss_ver3_architecture.txt', 'w') as f:
    f.write("")

# TRAINING ###############################################################

total_epochs = cfg['param']['rounds'] * cfg['param']['epochs']
cur_epoch = 0

for round in range(cfg['param']['rounds']):

    print("--- Round {}/{} ---".format(round+1, cfg['param']['rounds']))

    for epoch in range(cfg['param']['epochs']):
        
        cur_epoch += 1

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

                gradients = smashed_data.grad
                optimizer.zero_grad()
                
                # PROTOTYPE ################################################
                proto_optimizer.zero_grad()

                with torch.no_grad():
                    input_features = proto_linear(smashed_data.clone().detach()).requires_grad_()
                    ptr = int(proto_pointer)
                    queue_features[ptr:ptr + cfg['param']['batch_size']] = input_features
                    # queue_features[:, ptr:ptr + cfg['param']['batch_size']] = input_features
                    queue_labels[ptr:ptr + cfg['param']['batch_size']] = label

                    ptr = (ptr + cfg['param']['batch_size']) % queue_size
                    proto_pointer[0] = ptr

                    if ptr == 0:
                        use_proto = True # ここからはprototypeを用いて損失を計算する
                        for class_idx in range(classes_list[0]):
                            prototypes[class_idx] = queue_features[queue_labels == class_idx].mean(0)
                        prototypes = nn.functional.normalize(prototypes, dim=1)

                if use_proto:
                    input_features = nn.functional.normalize(input_features, dim=1)
                    proto_output = torch.mm(input_features, prototypes.t().to(device))
                    proto_output /= cfg['param']['temperature']
                    proto_loss = proto_criterion(proto_output, label)

                    # print(proto_loss.requires_grad)

                    # loss = loss * (cur_epoch / total_epochs) + proto_loss * (1 - cur_epoch / total_epochs)

                    proto_loss.backward()
                    proto_optimizer.step()
                    gradients += gradients * (cur_epoch / total_epochs) + smashed_data.grad * (1 - cur_epoch / total_epochs)

                loss_list.append(loss.item())

                # SEND GRADIENTS ##########################################

                # gradients = smashed_data.grad
                m_sr.server(connection, b"SEND", gradients)

            # PRINT LOSS ###############################################
            if iter % 100 == 0:

                agerage_loss = sum(loss_list) / len(loss_list)
                print("Round: {} | Epoch: {} | Iteration: {} | Loss: {:.4f}".format(round+1, epoch+1, iter+1, agerage_loss))
                
                cur_iter = iter + num_iterations['client1'] * epoch + round * num_iterations['client1'] * cfg['param']['epochs']

                with open('./../../results/proto_grad_mn_niid_v2/loss_ver3_architecture.txt', 'a') as f:
                    f.write("Iteration: {}, Loss: {:.4f}\n".format(cur_iter, agerage_loss))
                
                if use_proto:
                    print("Proto Loss: {:.4f}".format(proto_loss.item()))
                    with open('./../../results/proto_grad_mn_niid_v2/proto_loss_ver3_architecture.txt', 'a') as f:
                        f.write("Iteration: {}, Proto Loss: {:.4f}\n".format(cur_iter, proto_loss.item()))

    # TEST ###################################################################

    top_model.eval()
    with torch.no_grad():

        client_id = 0
        accuracy_list = []

        for connection in connection_list:

            client_id += 1

            correct = 0
            total = 0
            for iter in tqdm(range(test_num_iterations['client{}'.format(client_id)])):

                client_data = m_sr.server(connection, b"REQUEST")

                smashed_data = client_data['smashed_data'].to(device)
                label = client_data['labels'].to(device)

                output = top_model(smashed_data)
                _, predicted = torch.max(output, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
            
            accuracy = correct / total * 100
            accuracy_list.append(accuracy)
            print("Client{}: {}%".format(client_id, accuracy))
            m_sr.server(connection, b"SEND", accuracy)
    
    with open('./../../results/proto_grad_mn_niid_v2/accuracy_ver3_architecture.txt', 'a') as f:
        for i in range(len(accuracy_list)):
            f.write("Client{}: {:.2f}% | ".format(i+1, accuracy_list[i]))
        f.write("\n")

# CLOSE CONNECTION #####################################################

for connection in connection_list:
    connection.close()

server_socket.close()
