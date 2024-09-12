import argparse
import torch
import torch.nn.functional as F
import numpy as np

class Mixup:
    def __init__(self,
                 args: argparse.ArgumentParser, 
                 device: torch.device,
                 projection_head: torch.nn.Sequential,
                 alpha: float = 0.2,
                 shuffle: bool = True       
                 ):
        
        self.args = args
        self.device = device
        self.projection_head = projection_head
        self.alpha = alpha
        self.shuffle = shuffle
        self.global_model = None
        self.previous_model = None
        self.total_epochs = (args.num_rounds - 1) * args.num_epochs
        self.current_epoch = 0
        self.cos = torch.nn.CosineSimilarity(dim=1)
    
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
        
        l = torch.distributions.Beta(self.alpha, self.alpha).sample([self.args.batch_size])
        self.smashed_l = l.reshape(self.args.batch_size, 1, 1, 1).to(self.device)
        label_l = l.reshape(self.args.batch_size, 1).to(self.device)

        mixed_smashed_data = smashed_data * self.smashed_l + global_smashed_data * (1 - self.smashed_l)
        mixed_labels = labels * label_l + global_labels * (1 - label_l)

        return mixed_smashed_data, mixed_labels
    

    def train(self,
              images: torch.Tensor,
              smashed_data: torch.Tensor,
              client_model: torch.nn.Sequential,
              optimizer: torch.optim
              ):
        
        optimizer.zero_grad()

        smashed_output = self.projection_head(smashed_data) # CMとGMの出力のMixup
        neg = self.previous_model(images)
        shuffled_neg = neg[self.shuffle_indexes]

        mixed_neg = neg * self.smashed_l + shuffled_neg * (1 - self.smashed_l)
        mixed_neg_output = self.projection_head(mixed_neg)

        # print(smashed_output.size())
        # print(mixed_neg_output.size())

        similarity = self.cos(smashed_output, mixed_neg_output)
        loss = torch.mean(similarity)
        # print(loss.item())
        loss.backward()
        optimizer.step()

        grads = [ param.grad.clone() for param in client_model.parameters() ]

        return grads


    def requires_grad_false(self):
        for param1, param2 in zip(self.global_model.parameters(), self.previous_model.parameters()):
            param1.requires_grad = False
            param2.requires_grad = False

    

    def update_alpha(self):

        self.current_epoch += 1
        self.alpha = max(0, self.alpha * (1 - self.current_epoch / self.total_epochs))
        print(self.alpha)
