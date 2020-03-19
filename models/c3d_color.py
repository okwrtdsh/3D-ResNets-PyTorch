import torch.nn as nn


class C3D(nn.Module):
    """
    The C3D network as described in [1].
    """

    def __init__(self,
                 sample_size,
                 sample_duration,
                 num_classes=400):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(256)
        self.bn4 = nn.BatchNorm3d(512)
        self.bn5 = nn.BatchNorm3d(512)
        self.bn6 = nn.BatchNorm2d(4096)

    def forward(self, x):

        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.pool1(h)

        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu(h)
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.conv3b(h)
        h = self.bn3(h)
        h = self.relu(h)
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.conv4b(h)
        h = self.bn4(h)
        h = self.relu(h)
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.conv5b(h)
        h = self.bn5(h)
        h = self.relu(h)
        h = self.pool5(h)

        h = h.view(-1, 8192)
        h = self.fc6(h)
        h = self.bn6(h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)

        logits = self.fc8(h)
        probs = self.softmax(logits)

        return probs

