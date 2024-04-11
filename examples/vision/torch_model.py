import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# # Set seed for reproducibility
# torch.manual_seed(24)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# Define a simple neural network using torch.nn.Sequential
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 30)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(30, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define transformation for loading MNIST data
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Define data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Initialize the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


if __name__ == '__main__':
    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')



    # Test the model on the test dataset
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

    # Save the trained PyTorch model to disk
    torch.save(model.state_dict(), 'pytorch_model.pth')


def mnist_6_200():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,10),
    )
    return model


class ModifiedModel(nn.ModuleList):
    def __init__(self, base_model, num_layers_to_remove=1):
        super(ModifiedModel, self).__init__()

        num_layers = len(list(base_model.children()))
        layers_to_keep = num_layers - num_layers_to_remove

        # Initialize the base model with all layers except the last 'num_layers_to_remove' layers
        for idx, (name, layer) in enumerate(base_model.named_children()):
            if idx < layers_to_keep:
                self.add_module(name, layer)

    def forward(self, x):
        # Forward pass through the modified model
        for layer in self:
            x = layer(x)
        return x



class Simple3NN(nn.Module):
    # def __init__(self):
    #     super(Simple3NN, self).__init__()
    #     self.fc1 = nn.Linear(3, 2)  # Input layer to first hidden layer
    #     self.r1 = nn.ReLU()
    #     self.fc2 = nn.Linear(2, 2)  # First hidden layer to second hidden layer
    #     self.r2 = nn.ReLU()
    #     self.fc3 = nn.Linear(2, 2)  # Second hidden layer to output layer
    #
    #     # Initialize weights and biases with fixed values
    #     self.fc1.weight.data = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    #     self.fc1.bias.data = torch.tensor([0.1, 0.2])
    #     self.fc2.weight.data = torch.tensor([[0.2, 0.3], [0.4, 0.5]])
    #     self.fc2.bias.data = torch.tensor([0.1, 0.2])
    #     self.fc3.weight.data = torch.tensor([[0.3, 0.4], [0.5, 0.6]])
    #     self.fc3.bias.data = torch.tensor([0.1, 0.2])

    def __init__(self):
        super(Simple3NN, self).__init__()
        self.fc1 = nn.Linear(3, 2)  # Input layer to first hidden layer
        self.r1 = nn.ReLU()
        self.fc2 = nn.Linear(2, 2)  # First hidden layer to second hidden layer
        self.r2 = nn.ReLU()
        self.fc3 = nn.Linear(2, 2)  # Second hidden layer to output layer

        # Set random seed for reproducibility
        torch.manual_seed(44)

        # Initialize weights randomly within the range [-1, 1]
        self.fc1.weight.data = torch.rand((2, 3)) * 12 - 6
        self.fc1.bias.data = torch.rand((2,)) * 12 - 6
        self.fc2.weight.data = torch.rand((2, 2)) * 12 - 6
        self.fc2.bias.data = torch.rand((2,)) * 12 - 6
        # self.fc3.weight.data = torch.rand((2, 2)) * 12 - 6
        # self.fc3.bias.data = torch.rand((2,)) * 12 - 6

        self.fc3.weight.data = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
        self.fc3.bias.data = torch.tensor([0.0, 0.0])

        # self.fc3.weight.data = torch.tensor([[-1.00, +1.0]])
        # self.fc3.bias.data = torch.tensor([0.0])

    def forward(self, x):
        x = self.r1(self.fc1(x))  # First hidden layer with ReLU activation
        x = self.r2(self.fc2(x))  # Second hidden layer with ReLU activation
        x = self.fc3(x)  # Output layer
        return x

