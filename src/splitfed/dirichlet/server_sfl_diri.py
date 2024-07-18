import sys
sys.path.append('./../../../')

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

# Test

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
    classes_list = []
    for connection in connection_list:
        classes_list.append(m_sr.server(connection, b"REQUEST"))
    for i in range(len(classes_list)):
        if classes_list[i] != classes_list[0]:
            print("Number of classes is different.")
            raise ValueError
    
    client1_info = [103, 4048, 3279, 4766, 1458, 2003, 914, 1204, 2566, 4659]
    client2_info = [4897, 952, 1721, 234, 3542, 2997, 4086, 3796, 2434, 341]

    for i, connection in enumerate(connection_list):
        if i == 0:
            m_sr.server(connection, b"SEND", client1_info)
        else:
            m_sr.server(connection, b"SEND", client2_info) 

    info_list = []
    for connection in connection_list:
        client_info = m_sr.server(connection, b"REQUEST")
        info_list.append(client_info)
    
    for i in range(len(info_list)):
        if info_list[i]['num_iterations'] != info_list[0]['num_iterations']:
            print("Number of iterations is different.")
            raise ValueError
    
    return classes_list[0], info_list[0]['num_iterations']

def set_model(cfg, num_classes):
    model = models.mobilenet_v2(num_classes=num_classes)
    model.classifier[1] = nn.Linear(1280, cfg['output_size'])
    model.classifier.append(nn.ReLU6(inplace=True))
    model.classifier.append(nn.Linear(cfg['output_size'], num_classes))
    client_model = model.features[:6]
    return model, client_model.to(device)

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

    proto_sl_path = './../../../'
    config_path = proto_sl_path + 'config.json'
    dataset_path = proto_sl_path + 'dataset/'
    loss_path = proto_sl_path + 'results/sfl_result/dirichlet/loss_64.txt'
    accuracy_path = proto_sl_path + 'results/sfl_result/dirichlet/accuracy_64.txt'
    
    config = set_config(config_path)
    set_seed(42)
    server_socket, connection_list, client_address_list = set_connection(config)
    num_classes, num_iterations = get_info(connection_list, config)
    model, client_model = set_model(config, num_classes)

    for connection in connection_list:
        m_sr.server(connection, b"SEND", client_model)

    top_model, optimizer, criterion = set_server_model(model, config)
    top_model = top_model.to(device)

    test_dataset = dsets.CIFAR10(root=dataset_path, train=False, transform=transforms.ToTensor(), download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=False)

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
        
        # AVERAGING ########################################################

        client_model_list = []
        for connection in connection_list:
            client_model_list.append(m_sr.server(connection, b"REQUEST"))

        for idx in range(len(client_model_list)):
            for param1, param2 in zip(client_model.parameters(), client_model_list[idx].parameters()):
                if idx == 0:
                    param1.data.copy_(param2.data)
                elif 0 < idx < len(client_model_list)-1:
                    param1.data.add_(param2.data)
                else:
                    param1.data.add_(param2.data)
                    param1.data.div_(len(client_model_list))

        # TEST ############################################################

        top_model.eval()
        correct = 0
        total = 0
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = top_model(client_model(images))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print("Round: {}, Accuracy: {:.2f}%".format(round+1, accuracy))

        with open(accuracy_path, 'a') as f:
            f.write("Round: {} : Accuracy: {:.2f}\n".format(round+1, accuracy))
        
        # SEND MODEL ######################################################

        for connection in connection_list:
            m_sr.server(connection, b"SEND", client_model)
    
    # CLOSE CONNECTION #####################################################

    for connection in connection_list:
        connection.close()
    server_socket.close()