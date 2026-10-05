import torch
import torch.nn as nn



class TinyCNN(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_shape,hidden_units, kernel_size = 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units),
            nn.MaxPool2d(kernel_size=2 , stride=2),
            nn.Conv2d(hidden_units,hidden_units*2, kernel_size = 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units*2),
            nn.MaxPool2d(kernel_size=2 , stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units*2*32*32, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(64, output_shape)
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
