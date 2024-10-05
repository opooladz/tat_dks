import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Enable TF32 tensor cores for faster float32 matrix multiplications
torch.set_float32_matmul_precision('high')

from dks.examples.pytorch.modified_resnet_torch import create_resnet18
from dks.examples.pytorch.psgd_kron import Kron

import torch

torch._dynamo.config.cache_size_limit = 100_000_000
torch._dynamo.config.capture_scalar_outputs = False


# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
num_epochs = 10
batch_size = 128
learning_rate = 0.001

# Data augmentation and normalization for training
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    ),
])

# Only normalization for test data
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    ),
])

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform_train
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform_test
)

# Data loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)

# TAT parameters -- no idea what should be a good value for these
tat_params = {
    "c_val_0_target": 0.3,  # Must be between 0.0 and 1.0
    "c_curve_target": 0.3   # Must be greater than 0.0
}

# Create the ResNet18 model with TAT
model = create_resnet18(
    num_classes=10,  # CIFAR-10 has 10 classes
    use_batch_norm=False,  # TAT is typically used without batch normalization
    shortcut_weight=0.8,
    activation_name="leaky_relu",  # TAT with Leaky ReLU is recommended
    dropout_rate=0.2,
    transformation_method="TAT",
    tat_params=tat_params
)

# Move model to device
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Kron(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Training function
def train(epoch):
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 100 == 99:  # Print every 100 batches
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
            running_loss = 0.0
    scheduler.step()

# Testing function
def test(epoch):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}] Test Accuracy: {accuracy:.2f}%')

# Training loop
for epoch in range(num_epochs):
    train(epoch)
    test(epoch)

print('Finished Training')

