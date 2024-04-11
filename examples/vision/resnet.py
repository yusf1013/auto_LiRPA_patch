import requests
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# Load the pretrained model
model = models.resnet50(pretrained=True)
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the labels used by the pretrained model
LABELS_URL = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
labels = requests.get(LABELS_URL).json()

# Get a list of all files in the directory
image_files = os.listdir('data/ILSVRC/Data/CLS-LOC/test')

# Sort the list to ensure consistency
image_files.sort()

# Loop over the first 10 images
for i in range(10):
    # Select the image file
    image_file = image_files[i]

    # Construct the full image path
    image_path = os.path.join('data/ILSVRC/Data/CLS-LOC/test/', image_file)

    # Load the image
    image = Image.open(image_path).convert('RGB')

    # Apply the transformation to the image and add an extra batch dimension
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)

    # Make sure the tensor is on the same device as the model
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    # Get the model's prediction
    with torch.no_grad():
        output = model(input_batch)

    # Get the top prediction
    _, predicted_idx = torch.max(output, 1)
    predicted_label = labels[predicted_idx.item()]

    # Set the top prediction as the title
    plt.title(f'Image: {image_file}, Predicted: {predicted_label}')

    # Display the image
    plt.imshow(image)
    plt.show()