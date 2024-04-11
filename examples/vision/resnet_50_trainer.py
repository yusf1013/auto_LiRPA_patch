import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models

# Define the path to the ImageNet dataset folder
data_dir = 'data/ILSVRC/Data/CLS-LOC'

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Function to load and preprocess ImageNet images
def load_imagenet_images(data_dir, transform):
    image_paths = []
    labels = []
    for root, dirs, files in os.walk(data_dir):
        if root.endswith('test'):
            continue
        for file in files[:2]:
            if file.endswith('.JPEG'):  # Check for JPEG files
                image_path = os.path.join(root, file)
                image_paths.append(image_path)
                labels.append(os.path.basename(root))

    print("Reading finished")
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        images.append(image)

    return images, labels


# Load and preprocess ImageNet images
images, labels = load_imagenet_images(data_dir, transform)

# Convert images and labels to PyTorch tensors
images_tensor = torch.stack(images)
labels_tensor = torch.tensor(
    [int(label.split('.')[0]) for label in labels])  # Assuming labels are in the format 'nxxxxxxx.JPEG'

# Print the number of loaded images and labels
print(f'Number of images: {len(images)}')
print(f'Number of labels: {len(labels)}')
print('Shape of images tensor:', images_tensor.shape)
print('Shape of labels tensor:', labels_tensor.shape)

# Instantiate the ResNet-50 model
model = models.resnet50(pretrained=True)

# Modify the last fully connected layer for 1000 classes in ImageNet
num_classes = 1000
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model (assuming train_loader and val_loader are defined)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Epoch {epoch + 1}/{num_epochs}, Accuracy: {accuracy:.2f}%')

# Save the trained model
torch.save(model.state_dict(), 'resnet50_imagenet.pth')

# Load the saved model for inference
loaded_model = models.resnet50(pretrained=False)
loaded_model.fc = torch.nn.Linear(loaded_model.fc.in_features, num_classes)
loaded_model.load_state_dict(torch.load('resnet50_imagenet.pth'))
loaded_model.eval()
