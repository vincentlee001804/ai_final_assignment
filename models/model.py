import torch.nn as nn

class MyCNNModel(nn.Module):
    def __init__(self, num_classes):
        super(MyCNNModel, self).__init__()

        # Input is 224*224*3, going into this layer:
        self.conv1 = nn.Conv2d(
            in_channels=3, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        # Output features maps produced: 224*224*64

        self.conv2 = nn.Conv2d(
            in_channels=64, 
            out_channels=128, 
            kernel_size=3, 
            stride=2, 
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        # Output features maps produced: 112*112*128

        self.conv3 = nn.Conv2d(
            in_channels=128, 
            out_channels=256, 
            kernel_size=3, 
            stride=2, 
            padding=1
        )
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        # Output features maps produced: 56*56*256

        self.fc = nn.Linear(56*56*256, num_classes)
        # Output features maps produced: num_classes

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        
        x = x.view(x.size(0), -1) # flatten the output of the last conv layer
        
        x = self.fc(x) # apply the fully connected layer
        
        return x

