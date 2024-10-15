import argparse
import torch
import torch.nn.functional as F

class prototype:
    def __init__(
        self,
        args: argparse.ArgumentParser,
        num_classes: int,
        feature_size: int,
        device: str
    ):
        
        self.num_classes = num_classes
        self.feature_size = feature_size
        self.queue_size = args.queue_size
        self.batch_size = args.batch_size
        self.temperature = args.temperature

        self.criterion = torch.nn.CrossEntropyLoss()
        self.queue_features = torch.zeros(self.queue_size, self.feature_size).to(device)
        self.queue_labels = torch.zeros(self.queue_size, dtype=torch.long).to(device)
        self.prototypes = torch.zeros(num_classes, self.feature_size).to(device)
        self.pointer = torch.zeros(1, dtype=torch.long).to(device)
        self.use_proto = False

        self.proto_loss = 0.0
        self.total_loss = 0.0
        self.loss_count = 0
    
    def compute_prototypes(
        self,
        projected_feature: torch.Tensor,
        labels: torch.Tensor,
        main_loss,
        mu: float
    ):
        
        with torch.no_grad():

            ptr = int(self.pointer)
            self.queue_features[ptr:ptr+self.batch_size] = projected_feature
            self.queue_labels[ptr:ptr+self.batch_size] = labels
            ptr = (ptr + self.batch_size) % self.queue_size
            self.pointer[0] = ptr

            if ptr == 0:

                self.use_proto = True

                # プロトタイプの更新
                for cls in range(self.num_classes):
                    self.prototypes[cls] = self.queue_features[self.queue_labels == cls].mean(0)
                self.prototypes = F.normalize(self.prototypes, dim=1)

        if self.use_proto:

            projected_feature = F.normalize(projected_feature, dim=1)
            proto_output = torch.mm(projected_feature, self.prototypes.t())
            proto_output /= self.temperature

            p_loss = self.criterion(proto_output, labels)
            t_loss = mu * main_loss + (1 - mu) * p_loss

            self.proto_loss += p_loss.item()
            self.total_loss += t_loss.item()
            self.loss_count += 1

            return t_loss

        else:

            return main_loss
    

    def get_loss(self):
        
        if self.use_proto:

            self.proto_loss /= self.loss_count
            self.total_loss /= self.loss_count

        return self.proto_loss, self.total_loss
                    

    def reset_loss(self):

        self.proto_loss = 0.0
        self.total_loss = 0.0
        self.loss_count = 0
    