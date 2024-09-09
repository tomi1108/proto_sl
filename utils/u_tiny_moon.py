import torch
import argparse
import torch.utils
import torch.utils.data
from tqdm import tqdm

class Tim:
    def __init__(self, 
                 args: argparse.ArgumentParser, 
                 device: torch.device,
                 projection_head: torch.nn.Sequential,
                 train_loader: torch.utils.data.DataLoader 
                 ):

        self.args = args
        self.device = device
        self.projection_head = projection_head
        self.train_loader = train_loader

        self.criterion = torch.nn.CrossEntropyLoss()
        self.cos = torch.nn.CosineSimilarity(dim=1)
        self.num_classes = 10 # いずれは学習データごとに変わるようにする
        self.data_size = len(self.train_loader) * self.args.batch_size

        self.previous_average = None
        self.global_average = None

    def calculate_average(self,
                          model: torch.nn.Sequential,
                          prev_flag: bool = False,
                          glob_flag: bool = False
                          ):

        queue_feature = torch.zeros(self.data_size, self.args.output_size)
        queue_label = torch.zeros(self.data_size, dtype=torch.long)
        average_feature = torch.zeros(self.num_classes, self.args.output_size)
        ptr = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                feature_vector = self.projection_head(model(images))
                queue_feature[ptr: ptr + self.args.batch_size] = feature_vector
                queue_label[ptr: ptr + self.args.batch_size] = labels
                ptr = (ptr + self.args.batch_size) % self.data_size

            for cls_idx in range(self.num_classes):
                class_features = queue_feature[queue_label == cls_idx]
                if class_features.size(0) > 0:
                    average_feature[cls_idx] = class_features.mean(0)
                else:
                    average_feature[cls_idx] = torch.zeros(self.args.output_size)
        
        if prev_flag or glob_flag:
            if prev_flag:
                self.previous_average = average_feature
            elif glob_flag:
                self.global_average = average_feature
    
    def train(self,
              smashed_data: torch.Tensor,
              labels: torch.Tensor,
              client_model: torch.nn.Sequential,
              optimizer: torch.optim
              ):
        
        optimizer.zero_grad()
        smashed_output = self.projection_head(smashed_data)
        pos_output = torch.zeros(len(smashed_output), self.args.output_size)
        neg_output = torch.zeros(len(smashed_output), self.args.output_size)
        
        for data_idx, label in enumerate(labels):
            pos_output[data_idx] = self.global_average[label]
            neg_output[data_idx] = self.previous_average[label]

        pos_cos = self.cos(smashed_output, pos_output.to(self.device))
        logits = pos_cos.reshape(-1, 1)
        neg_cos = self.cos(smashed_output, neg_output.to(self.device))
        logits = torch.cat((logits, neg_cos.reshape(-1, 1)), dim=1)

        logits /= 0.5
        logits_labels = torch.zeros(smashed_data.size(0)).to(self.device).long()

        loss = self.criterion(logits, logits_labels)
        loss.backward(retain_graph=True)
        
        grads = [ param.grad.clone() for param in client_model.parameters() ]
        
        return grads

# def calculate_average(args: argparse.ArgumentParser, model, projection_head, data_loader):
    
#     self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     num_classes = 10 # ここは仮設定で、いずれはnum_classを引数とする
#     self.data_size = len(data_loader) * args.batch_size
#     queue_feature = torch.zeros(data_size, args.output_size)
#     queue_label = torch.zeros(data_size, dtype=torch.long)
#     average_feature = torch.zeros(num_classes, args.output_size)
#     pointer = torch.zeros(1, dtype=torch.long)

#     with torch.no_grad():
#         for images, labels in tqdm(data_loader):
#             images, labels = images.to(device), labels.to(device)
#             feature_vector = projection_head(model(images))
#             ptr = int(pointer)
#             queue_feature[ptr:ptr+args.batch_size] = feature_vector
#             queue_label[ptr:ptr+args.batch_size] = labels
        
#         for cls_idx in range(num_classes):
#             class_features = queue_feature[queue_label == cls_idx]
#             if class_features.size(0) > 0:
#                 average_feature[cls_idx] = queue_feature[queue_label == cls_idx].mean(0)
#             else:
#                 average_feature[cls_idx] = torch.zeros(args.output_size)
#         # average_feature = torch.nn.functional.normalize(average_feature, dim=1, eps=1e-8)
    
#     return average_feature

def calculate_logits(args: argparse.ArgumentParser, smashed_data, labels, global_average, previous_average, projection_head, cos, device):

    # smashed_output = torch.nn.functional.normalize(projection_head(smashed_data), dim=1, eps=1e-8)
    smashed_output = projection_head(smashed_data)
    pos_output = torch.zeros(len(smashed_output), args.output_size)
    neg_output = torch.zeros(len(smashed_output), args.output_size)
    
    for data_idx, label in enumerate(labels):
        pos_output[data_idx] = global_average[label]
        neg_output[data_idx] = previous_average[label]
    
    pos_cos = cos(smashed_output, pos_output.to(device))
    logits = pos_cos.reshape(-1, 1)
    neg_cos = cos(smashed_output, neg_output.to(device))
    logits = torch.cat((logits, neg_cos.reshape(-1, 1)), dim=1)

    logits /= 0.5
    logits_labels = torch.zeros(smashed_data.size(0)).to(device).long()

    return logits, logits_labels
