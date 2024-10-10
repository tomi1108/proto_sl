import argparse
import copy
import random
import torch
import torch.nn.functional as F
import numpy as np
import utils.u_send_receive as u_sr

class Mixup_client:
    def __init__(self,
                 args: argparse.ArgumentParser, 
                 device: torch.device,
                 alpha: float = 0.2,
                 shuffle: bool = True       
                 ):
        
        self.args = args
        self.device = device
        self.alpha = alpha
        self.shuffle = shuffle
        self.global_model = None


    def mix_smashed_data(self,
                         images: torch.Tensor,
                         labels: torch.Tensor,
                         smashed_data: torch.Tensor
                         ):
        
        labels = F.one_hot(labels, num_classes=10) # ひとまず手動で設定する
        global_smashed_data = self.global_model(images)
        global_labels = labels.clone().detach()
        if self.shuffle:
            self.shuffle_indexes = torch.randperm(self.args.batch_size)
            global_smashed_data = global_smashed_data[self.shuffle_indexes]
            global_labels = global_labels[self.shuffle_indexes]
        # global_smashed_data = torch.nn.functional.normalize(global_smashed_data, dim=1)
        
        l = torch.distributions.Beta(self.alpha, self.alpha).sample([self.args.batch_size])
        smashed_l = l.reshape(self.args.batch_size, 1, 1, 1).to(self.device)
        label_l = l.reshape(self.args.batch_size, 1).to(self.device)

        # smashed_data = torch.nn.functional.normalize(smashed_data, dim=1)
        mixed_smashed_data = smashed_data * smashed_l + global_smashed_data * (1 - smashed_l)
        mixed_labels = labels * label_l + global_labels * (1 - label_l)

        return mixed_smashed_data, mixed_labels
    
    def requires_grad_false(self):
        for param in self.global_model.parameters():
            param.requires_grad = False

class Mixup_server:
    def __init__(self,
                 args: argparse.ArgumentParser,
                 device: torch.device,
                 alpha: float = 0.2
                 ):
        
        self.args = args
        self.device = device
        self.alpha = alpha
    
        # self.num_classes = 10 # CIFAR10用
        # self.queue_size = args.queue_size
        # self.queue_features = torch.zeros(self.queue_size, 512)
        # self.queue_labels = torch.zeros(self.queue_size, dtype=torch.long)
        # self.prototypes = torch.zeros(self.num_classes, 512).to(self.device)
        # self.proto_pointer = torch.zeros(1, dtype=torch.long)
        # self.use_proto = False
    
    def compute_prototypes(self,
                           output: torch.Tensor,
                           labels: torch.Tensor,
                           round: int
                           ):
        
        with torch.no_grad():
            ptr = int(self.proto_pointer)
            self.queue_features[ptr:ptr + self.args.batch_size] = output.reshape(self.args.batch_size, -1)
            self.queue_labels[ptr:ptr + self.args.batch_size] = labels
            ptr = (ptr + self.args.batch_size) % self.args.queue_size
            self.proto_pointer[0] = ptr

            if ptr == 0:
                self.use_proto = True
                for idx in range(self.num_classes):
                    self.prototypes[idx] = self.queue_features[self.queue_labels == idx].mean(0)


        with torch.no_grad():
            ptr = int(self.proto_pointer)
            
            if round == 0: 
                _to_ptr = ptr + self.args.batch_size
            elif round > 0:
                _to_ptr = ptr + int(self.args.batch_size / 2)
            
            if _to_ptr > self.args.queue_size:
                _to_ptr = self.args.queue_size

            if round == 0:
                self.queue_features[ptr:_to_ptr] = output.reshape(self.args.batch_size, -1)
                self.queue_labels[ptr:_to_ptr] = labels
            elif round > 0:
                self.queue_features[ptr:_to_ptr] = output.reshape(self.args.batch_size, -1)[64:self.args.batch_size]
                self.queue_labels[ptr:_to_ptr] = torch.argmax(labels[64:self.args.batch_size])
            ptr = (_to_ptr) % self.args.queue_size
            self.proto_pointer[0] = ptr

            if ptr == 0:
                self.use_proto = True
                for idx in range(self.num_classes):
                    self.prototypes[idx] = self.queue_features[self.queue_labels == idx].mean(0)

    
    def mix_smashed_and_proto(self,
                              smashed_data: torch.Tensor,
                              labels: torch.Tensor,
                              round: int
                              ):

        # if round == 0:
        #     batch_size = self.args.batch_size
        # elif round > 0:
        #     batch_size = self.args.batch_size / 2

        # l = torch.distributions.Beta(self.alpha, self.alpha).sample([batch_size])
        # smashed_l = l.reshape(batch_size, 1, 1, 1).to(self.device)
        # label_l = l.reshape(batch_size, 1).to(self.device)

        # one_hot_label = torch.eye(self.num_classes)[torch.arange(self.num_classes)].to(self.device)
        # num_loop = batch_size // self.num_classes
        # mix_index = list(range(self.num_classes)) * num_loop + random.sample(range(self.num_classes), batch_size - num_loop * self.num_classes) # バッチサイズが128の場合
        # random.shuffle(mix_index)

        # smashed_data_size = smashed_data.size()
        # mix_smashed_data = smashed_data * smashed_l + self.prototypes[mix_index].reshape(smashed_data_size) * (1 - smashed_l)
        # mix_labels = F.one_hot(labels, self.num_classes) * label_l + one_hot_label[mix_index] * (1 - label_l)

        # mix_smashed_data = torch.cat((smashed_data[0:64], mix_smashed_data[64:self.args.batch_size]), dim=0)
        # mix_labels = torch.cat((labels[0:64], mix_labels[64:self.args.batch_size]), dim=0)

        # return mix_smashed_data, mix_labels

        l = torch.distributions.Beta(self.alpha, self.alpha).sample([self.args.batch_size])
        smashed_l = l.reshape(self.args.batch_size, 1, 1, 1).to(self.device)
        label_l = l.reshape(self.args.batch_size, 1).to(self.device)

        one_hot_label = torch.eye(self.num_classes)[torch.arange(self.num_classes)].to(self.device)
        mix_index = list(range(self.num_classes)) * 12 + random.sample(range(self.num_classes), 8) # バッチサイズが128の場合
        random.shuffle(mix_index)

        smashed_data_size = smashed_data.size()
        mix_smashed_data = smashed_data * smashed_l + self.prototypes[mix_index].reshape(smashed_data_size) * (1 - smashed_l)
        if round == 0:
            mix_labels = F.one_hot(labels, self.num_classes) * label_l + one_hot_label[mix_index] * (1 - label_l)

        if round > 0:
            mix_labels = labels * label_l + one_hot_label[mix_index] * (1 - label_l)

            mix_smashed_data = torch.cat((smashed_data[0:64], mix_smashed_data[64:self.args.batch_size]), dim=0)
            mix_labels = torch.cat((labels[0:64], mix_labels[64:self.args.batch_size]), dim=0)

        return mix_smashed_data, mix_labels

    def mix_smashed_data(self,
                         connections: dict
                         ):

        num_clinets = self.args.num_clients        
        client_data_dict = {}
        for client_id, connection in connections.items():
            client_data_dict[client_id] = u_sr.server(connection, b"REQUEST")
            if self.args.Mix_flag == False:
                client_data_dict[client_id]['labels'] = F.one_hot(client_data_dict[client_id]['labels'], num_classes=10) # TODO: クラス数が10以上の場合
        
        l = torch.distributions.Beta(self.alpha, self.alpha).sample([self.args.batch_size])
        smashed_l = l.reshape(self.args.batch_size, 1, 1, 1)
        label_l = l.reshape(self.args.batch_size, 1)

        for i in range(num_clinets):
            client_id = i + 1
            smashed_data = client_data_dict['Client {}'.format(client_id)]['smashed_data']
            labels = client_data_dict['Client {}'.format(client_id)]['labels']

            _smashed_data = client_data_dict['Client {}'.format((client_id % num_clinets)+1)]['smashed_data']
            _labels = client_data_dict['Client {}'.format((client_id % num_clinets)+1)]['labels']

            mix_smashed_data = smashed_data * smashed_l + _smashed_data * (1 - smashed_l)
            mix_labels = labels * label_l + _labels * (1 - label_l)

            client_data_dict['Client {}'.format(client_id)]['smashed_data'] = mix_smashed_data
            client_data_dict['Client {}'.format(client_id)]['labels'] = mix_labels
        
        return client_data_dict

    def train(self,
              client_data_dict: dict,
              client_id: str,
              server_model: torch.nn.Module,
              optimizer: torch.optim,
              criterion: torch.nn
              ):
        
        smashed_data = client_data_dict[client_id]['smashed_data'].to(self.device)
        smashed_data.retain_grad()
        labels = client_data_dict[client_id]['labels'].to(self.device)

        optimizer.zero_grad()
        _, output = server_model(smashed_data)
        loss = criterion(output, labels)
        loss.backward(retain_graph=True)
        optimizer.step()
        
        return loss, smashed_data.grad