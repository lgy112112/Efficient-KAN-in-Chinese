import torch
from torch import nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import numpy as np
import random
from kat_rational import KAT_Group2D
from tqdm import tqdm

def set_random_seed(seed_value=42):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)
    random.seed(seed_value)

class CIFARNet(nn.Module):
    def __init__(self, activation_func):
        super(CIFARNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.act1 = activation_func()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.act2 = activation_func()
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.act3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(self.act1(self.conv1(x)))
        x = self.pool(self.act2(self.conv2(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = self.act3(self.fc1(x))
        x = self.fc2(x)
        return x

def train_one_epoch(model, data_loader, criterion, optimizer, device, epoch):
    """
    Train the model for a single epoch on the given data_loader.
    Returns the total loss for that epoch.
    """
    model.train()
    total_loss = 0.0
    
    # Create a progress bar
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}", unit="batch")

    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()

        # Update postfix on the progress bar at a set interval (e.g., every 10 batches)
        if batch_idx % 10 == 0:
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss


def evaluate(model, data_loader, device):
    """
    Evaluate the model on the given data_loader.
    Returns the number of correctly predicted samples and total samples.
    """
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    return total_correct, total_samples

def train_and_benchmark(activation_func, label, epochs=10, seed=42):
    set_random_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CIFARNet(activation_func).to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Training
    start_time = time.time()
    for epoch in range(epochs):
        total_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch
        )

    training_duration = time.time() - start_time
    print(f"{label} Training completed in {training_duration:.2f} seconds.")

    # Evaluation
    total_correct, total_samples = evaluate(model, test_loader, device)
    accuracy = 100.0 * total_correct / total_samples
    total_time = time.time() - start_time

    print(f"{label} Testing Accuracy: {accuracy:.2f}%, Total time: {total_time:.2f} seconds.")

if __name__ == "__main__":
    kat_activation = KAT_Group2D  # Replace with your actual KAT_1DGroup class if available
    train_and_benchmark(kat_activation, 'KAT 2DGroup')

    
    kat_activation = nn.ReLU  # Replace with your actual KAT_1DGroup class if available
    train_and_benchmark(kat_activation, 'ReLU')