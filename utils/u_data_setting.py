import argparse
import torchvision.datasets as dsets
import torchvision.transforms as transforms
# import u_send_receive as u_sr

from torch.utils.data import DataLoader, Subset

def client_setting(args: argparse.ArgumentParser):
    
    if args.dataset_type == 'cifar10':
        train_dataset = dsets.CIFAR10(root=args.dataset_path, train=True, transform=transforms.ToTensor(), download=True)
        if args.fed_flag == False:
            test_dataset = dsets.CIFAR10(root=args.dataset_path, train=False, transform=transforms.ToTensor(), download=True)
            test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=False)
    
    num_classes = len(train_dataset.classes)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if args.fed_flag == True:
        return train_loader, None
    else:
        return train_loader, test_loader


def server_setting(args: argparse.ArgumentParser):
    
    if args.fed_flag == True:
        if args.dataset_type == 'cifar10':
            test_dataset = dsets.CIFAR10(root=args.dataset_path, train=False, transform=transforms.ToTensor(), download=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=False)
        return test_loader
    else:
        return None