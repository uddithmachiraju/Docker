import os
import torch
from PIL import Image 
from model.net import Network 
import torch.nn.functional as F
from model.config import network_config 
from torchvision import transforms 

model_path = os.path.join(os.path.dirname(__file__), "saved-models/mnist_cnn.pth")

device = network_config['device'] 

model = Network()
model.load_state_dict(torch.load(model_path, map_location = device)) 
model.eval() 

# Processing the image
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = Image.open(image_bytes).convert("L")  # Convert image to grayscale
    return transform(image).unsqueeze(0)  # Add batch dimension

# Function to make prediction
def predict(image_bytes):
    image_tensor = transform_image(image_bytes)
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(F.softmax(output, dim=1), dim=1).item()
    return prediction