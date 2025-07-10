import torch
import torch.nn as nn

class DADiscriminator(nn.Module):
    def __init__(self):
        super(DADiscriminator, self).__init__()
        self.model = nn.Sequential(
            # nn.Flatten(),
            # nn.Linear(256 * 3 * 3, 512),
            # nn.LeakyReLU(0.2, inplace=True),

            # nn.Linear(512, 128),
            # nn.LeakyReLU(0.2, inplace=True),

            # nn.Linear(128, 1),
            # nn.Sigmoid()
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 1),
            # nn.Linear(9216, 1024),
            # nn.ReLU(inplace=True),
            # nn.Linear(1024, 2048),
            # nn.ReLU(inplace=True),
            # nn.Linear(2048, 3072),
            # nn.ReLU(inplace=True),
            # nn.Linear(3072, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
