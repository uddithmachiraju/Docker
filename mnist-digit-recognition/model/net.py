import torch 
from torch import nn 

# Define CNN Model using Sequential Blocks
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),  # Conv Layer 1
            nn.ReLU(),
            nn.BatchNorm2d(32),  
            nn.Conv2d(32, 64, 3, 1),  # Conv Layer 2
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),  # Pooling Layer
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(9216, 256),  # Fully Connected Layer 1
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),  # Fully Connected Layer 2 (Output)
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x):
        return self.model(x)
    
if __name__ == '__main__':
    net = Network(10)
    noise = torch.randn((1, 1, 28, 28)) 
    print(net(noise).shape) 