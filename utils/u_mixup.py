import argparse
import torch
import torch.nn.functional as F
import numpy as np

class Mixup:
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
    
    def train(self,
              images: torch.Tensor,
              labels: torch.Tensor,
              smashed_data: torch.Tensor
              ):
        
        labels = F.one_hot(labels, num_classes=10) # ひとまず手動で設定する
        global_smashed_data = self.global_model(images)
        global_labels = labels.clone().detach()
        if self.shuffle:
            shuffle_indexes = torch.randperm(self.args.batch_size)
            global_smashed_data = global_smashed_data[shuffle_indexes]
            global_labels = global_labels[shuffle_indexes]
        
        l = torch.distributions.Beta(self.alpha, self.alpha).sample([self.args.batch_size])
        # l = np.random.beta(self.alpha, self.alpha, self.args.batch_size)
        smashed_l = l.reshape(self.args.batch_size, 1, 1, 1).to(smashed_data.device)
        label_l = l.reshape(self.args.batch_size, 1).to(labels.device)

        mixed_smashed_data = smashed_data * smashed_l + global_smashed_data * (1 - smashed_l)
        mixed_labels = labels * label_l + global_labels * (1 - label_l)

        return mixed_smashed_data, mixed_labels


    def requires_grad_false(self):
        for param in self.global_model.parameters():
            param.requires_grad = False