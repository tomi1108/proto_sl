import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import json
import socket
import sys
sys.path.append('./../')
from tqdm import tqdm
import module.module_send_receive as m_sr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config_path = './../config.json'
dataset_path = './../../dataset/'
loss_path = './../../sfl_result/iid/loss.txt'
accuracy_path = './../../sfl_result/iid/accuracy.txt'

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

# TRAINING MODEL ########################################################

model = models.mobilenet_v2(num_classes=10)
model.classifier[1] = nn.Linear(1280, cfg['param']['output_size'])
model.classifier.append(nn.ReLU6(inplace=True))
model.classifier.append(nn.Linear(cfg['param']['output_size'], 10))

# CLIENT MODEL ###########################################################

client_model = model.features[:6]
client_model = client_model.to(device)

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

# TEST DATASET ##########################################################

test_dataset = dsets.CIFAR10(root=dataset_path, train=False, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=cfg['param']['batch_size'], shuffle=False)

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

top_model = Server_Model(model=model).to(device)
print(top_model)

optimizer = torch.optim.SGD(params = top_model.parameters(),
                            lr = cfg['param']['lr'],
                            momentum = cfg['param']['momentum'],
                            weight_decay = cfg['param']['weight_decay']
                            )

criterion = nn.CrossEntropyLoss()

# INIT TEXT FILE ########################################################

with open(loss_path, 'w') as f:
    f.write("")
with open(accuracy_path, 'w') as f:
    f.write("")

# TRAINING ###############################################################

for round in range(cfg['param']['rounds']):

    top_model.train()

    print("--- Round {}/{} ---".format(round+1, cfg['param']['rounds']))

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

                average_loss = sum(loss_list) / len(loss_list)
                print("Round: {} | Epoch: {} | Iteration: {} | Loss: {:.4f}".format(round+1, epoch+1, iter+1, average_loss))
                
                cur_iter = iter + num_iterations['client1'] * epoch + round * num_iterations['client1'] * cfg['param']['epochs']

                with open(loss_path, 'a') as f:
                    f.write("Iteration: {}, Loss: {:.4f}\n".format(cur_iter, average_loss))
    
    # 平均化 ###############################################################

    client_model_list = []

    for connection in connection_list:
        client_model_list.append(m_sr.server(connection, b"REQUEST"))

    # print(list(client_model.parameters())[0])
    # print("\n")
    # print(list(client_model_list[0].parameters())[0])
    # print("\n")
    # print(list(client_model_list[1].parameters())[0])
    # print("\n")

    for idx in range(len(client_model_list)):
        for param1, param2 in zip(client_model.parameters(), client_model_list[idx].parameters()):
            if idx == 0:
                param1.data.copy_(param2.data)
            elif 0 < idx < len(client_model_list)-1:
                param1.data.add_(param2.data)
            else:
                param1.data.add_(param2.data)
                param1.data.div_(len(client_model_list))
        
    # print(list(client_model.parameters())[0])

    # TEST #################################################################

    top_model.eval()

    correct = 0
    total = 0
    for images, labels in test_loader:
        
        images, labels = images.to(device), labels.to(device)
        output = top_model(client_model(images))
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print("Round: {} | Accuracy: {:.4f}".format(round+1, correct / total))
        
    with open(accuracy_path, 'a') as f:
        f.write("Round: {}, Accuracy: {:.4f}\n".format(round+1, correct / total))

    # SEND AGGREGATED MODEL ##############################################

    for connection in connection_list:
        m_sr.server(connection, b"SEND", client_model)

# CLOSE CONNECTION #####################################################

for connection in connection_list:
    connection.close()

server_socket.close()
