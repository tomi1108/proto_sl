import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple

def test_with_global_model(
    server_model: torch.nn.Module,
    client_model: torch.nn.Sequential,
    test_loader: DataLoader,
    device: str
) -> Tuple[float, float]:
    
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        
        for images, labels in tqdm(test_loader):
        
            images, labels = images.to(device), labels.to(device)
            _, outputs = server_model(client_model(images))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            running_loss += loss.item()
        
        accuracy = 100 * correct / total
        running_loss /= len(test_loader)
    
    return accuracy, running_loss