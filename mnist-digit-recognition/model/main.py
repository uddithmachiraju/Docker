import torch
from net import Network
from torch import optim  
from config import network_config  
from torchvision import datasets, transforms  
from torch.optim.lr_scheduler import StepLR 
from test import test_model
from train import train_model

# Data Loaders with Augmentation
transform = transforms.Compose([
    transforms.RandomRotation(10), 
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train = True, download = True, transform = transform)
test_dataset = datasets.MNIST('./data', train = False, transform = transform)

# Determine the number of classes from the dataset
n_classes = len(train_dataset.classes) 

train_dataloder = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = network_config['batch-size'], shuffle = True) 
test_dataloder = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = network_config['batch-size'], shuffle = False)  

# Initialize the model and move it to the specified device
device = network_config['device'] 
model = Network().to(device)

# Define the optimizer (Adadelta) with the specified learning rate
optimizer = optim.Adam(model.parameters(), lr=network_config['learning-rate'], weight_decay=1e-4)

# Define the learning rate scheduler (step every epoch)
scheduler = StepLR(optimizer=optimizer, step_size=1)

# Training loop placeholder (currently only updates scheduler per epoch)
for epoch in range(1, network_config['epochs'] + 1):
    train_model(model = model, device = device, train_dataloader = train_dataloder,
                optimizer = optimizer, epoch = epoch, log_interval = network_config['log-interval'])
    test_model(model = model, device = network_config['device'], test_dataloader = test_dataloder) 
    scheduler.step() 

# Save model state if enabled in config
if network_config['save-model']:
    torch.save(model.state_dict(), network_config['path']) 
    print('Model saved') 
