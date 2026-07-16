from a_imports import *

'''
This follows a Sequential/FeedForwarding flow

'''
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, padding = 1, padding_mode = "circular")
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding = 1, padding_mode = "circular")
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding = 1, padding_mode = "circular")
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2) # Pooling matrix size is 2x2
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3) # 30% samples dropped from a flat vector 
        self.lin1 = nn.Linear(64, 64)
        self.lin2 = nn.Linear(64,2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.GAP(x)
        x = torch.flatten(x,1)
        x = self.dropout(x)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        
        return x


