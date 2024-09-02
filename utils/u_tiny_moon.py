import torch
import argparse
from tqdm import tqdm

def calculate_average(args: argparse.ArgumentParser, model, projection_head, data_loader):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 10 # ここは仮設定で、いずれはnum_classを引数とする
    data_size = len(data_loader) * args.batch_size
    queue_feature = torch.zeros(data_size, args.output_size)
    queue_label = torch.zeros(data_size, dtype=torch.long)
    average_feature = torch.zeros(num_classes, args.output_size)
    pointer = torch.zeros(1, dtype=torch.long)

    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images, labels = images.to(device), labels.to(device)
            feature_vector = projection_head(model(images))
            ptr = int(pointer)
            queue_feature[ptr:ptr+args.batch_size] = feature_vector
            queue_label[ptr:ptr+args.batch_size] = labels
        
        for cls_idx in range(num_classes):
            class_features = queue_feature[queue_label == cls_idx]
            if class_features.size(0) > 0:
                average_feature[cls_idx] = queue_feature[queue_label == cls_idx].mean(0)
            else:
                average_feature[cls_idx] = torch.zeros(args.output_size)
        # average_feature = torch.nn.functional.normalize(average_feature, dim=1, eps=1e-8)
    
    return average_feature

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
