import torch
from kat_rational import KAT_Group
from torch import nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import numpy as np
import random

def set_random_seed(seed_value=42):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)
    random.seed(seed_value)

class NeuralNet(nn.Module):
    def __init__(self, activation_func):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.activation = activation_func
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

def train_and_benchmark(activation_func, label, epochs=10, seed=42):
    set_random_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNet(activation_func).to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model.train()
    start_time = time.time()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'{label} - Epoch {epoch+1}: Loss {total_loss / len(data_loader)}')
    duration = time.time() - start_time
    print(f'{label} Training completed in {duration:.2f} seconds.')
    
    # Testing phase
    model.eval()
    total_correct = 0
    total_images = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    accuracy = total_correct / total_images * 100
    duration = time.time() - start_time
    print(f'{label} Testing Accuracy: {accuracy:.2f}%, Training completed in {duration:.2f} seconds.')


if __name__ == "__main__":
    gelu = nn.GELU()
    train_and_benchmark(gelu, 'GELU')
    
    kat_activation = KAT_Group() # Placeholder for KAT_1DGroup if not accessible
    train_and_benchmark(kat_activation, 'KAT 1DGroup')
    print(kat_activation.weight_numerator, kat_activation.weight_denominator)

