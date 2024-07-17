import sys
sys.path.append('./../../')

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import json
import socket
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

def set_connection(cfg):
    connection_list = []
    client_address_list = []

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('localhost', cfg['port_number'])
    server_socket.bind(server_address)
    server_socket.listen(cfg['num_client'])
    print('Waiting for client...')

    while len(connection_list) < cfg['num_client']:
        connection, client_address = server_socket.accept()
        connection_list.append(connection)
        client_address_list.append(client_address)
        print("Connected to client: ", client_address)

    return server_socket, connection_list, client_address_list

def get_info(connection, cfg):
    info_list = []
    for connection in connection_list:
        info_list.append(m_sr.server(connection, b"REQUEST"))

    for i in range(len(info_list)):
        if info_list[i]['num_classes'] != info_list[0]['num_classes']:
            print("The number of classes is different.")
            raise ValueError
        if info_list[i]['num_iterations'] != info_list[0]['num_iterations']:
            print("The number of iterations is different.")
            raise ValueError
        if info_list[i]['test_num_iterations'] != info_list[0]['test_num_iterations']:
            print("The number of test iterations is different.")
            raise ValueError
    
    return info_list[0]['num_classes'], info_list[0]['num_iterations'], info_list[0]['test_num_iterations']

def set_model(cfg, num_classes):
    model = models.mobilenet_v2(num_classes=num_classes)
    model.classifier[1] = nn.Linear(1280, cfg['output_size'])
    model.classifier.append(nn.ReLU6(inplace=True))
    model.classifier.append(nn.Linear(cfg['output_size'], num_classes))
    client_model = model.features[:6]
    return model, client_model

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

def set_server_model(model, cfg):
    top_model = Server_Model(model=model)
    optimizer = torch.optim.SGD(params = top_model.parameters(),
                                lr = cfg['lr'],
                                momentum = cfg['momentum'],
                                weight_decay = cfg['weight_decay']
                                )
    criterion = nn.CrossEntropyLoss()
    return top_model, optimizer, criterion

def set_result_file(loss_path, accuracy_path):
    with open(loss_path, 'w') as f:
        f.write("")
    with open(accuracy_path, 'w') as f:
        f.write("")
    return

if __name__ == '__main__':
    
    # SETTING ###########################################################

    config_path = './../../config.json'
    loss_path = './../../sl_result/iid/loss_64.txt'
    accuracy_path = './../../sl_result/iid/accuracy_64.txt'
    
    config = set_config(config_path)
    set_seed(42)
    server_socket, connection_list, client_address_list = set_connection(config)
    num_classes, num_iterations, test_num_iterations = get_info(connection_list, config)
    model, client_model = set_model(config, num_classes)

    for connection in connection_list:
        m_sr.server(connection, b"SEND", client_model)

    top_model, optimizer, criterion = set_server_model(model, config)
    top_model = top_model.to(device)

    set_result_file(loss_path, accuracy_path)

    # TRAINING ###########################################################

    for round in range(config['rounds']):
        
        top_model.train()
        print("--- Round {}/{} ---".format(round+1, config['rounds']))
        
        for epoch in range(config['epochs']):
            
            print("--- Epoch {}/{} ---".format(epoch+1, config['epochs']))

            for i in tqdm(range(num_iterations)):

                loss_list = []
                for connection in connection_list:

                    client_data = m_sr.server(connection, b"REQUEST")
                    smashed_data = client_data['smashed_data'].to(device)
                    labels = client_data['labels'].to(device)

                    optimizer.zero_grad()
                    outputs = top_model(smashed_data)

                    loss = criterion(outputs, labels)

                    loss_list.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    m_sr.server(connection, b"SEND", smashed_data.grad)
                
                if i % 100 == 0:
                    average_loss = sum(loss_list) / len(loss_list)
                    print("Round: {} | Epoch: {} | Iteration: {} | Loss: {:.4f}".format(round+1, epoch+1, i+1, average_loss))
                    cur_iter = i + num_iterations * epoch + round * num_iterations * config['epochs']
                    with open(loss_path, 'a') as f:
                        f.write("Iteration: {}, Loss: {:.4f}\n".format(cur_iter, average_loss))
                
        # TEST ############################################################

        top_model.eval()
        with torch.no_grad():
            accuracy_list = []
            for j, connection in enumerate(connection_list):
                correct = 0
                total = 0
                for i in tqdm(range(test_num_iterations)):
                    client_data = m_sr.server(connection, b"REQUEST")
                    smashed_data = client_data['smashed_data'].to(device)
                    labels = client_data['labels'].to(device)
                    output = top_model(smashed_data)
                    _, predicted = torch.max(output, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                accuracy = correct / total * 100
                accuracy_list.append(accuracy)
                print("Client{}: {}%".format(j+1, accuracy))
                m_sr.server(connection, b"SEND", accuracy)
        
        with open(accuracy_path, 'a') as f:
            for i in range(len(accuracy_list)):
                f.write("Client{}: {:.2f}% | ".format(i+1, accuracy_list[i]))
            f.write("\n")
    
    # CLOSE CONNECTION #####################################################

    for connection in connection_list:
        connection.close()
    server_socket.close()