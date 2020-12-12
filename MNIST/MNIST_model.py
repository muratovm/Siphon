import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class Model():
    
    def __init__(self,device):
        outputs = 10
        image_shape = (1,28,28)
        self.device = device
        self.network = ConvNet(outputs, image_shape).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), amsgrad=True, weight_decay=0.01)
        self.criterion = nn.CrossEntropyLoss()
        self.start_epoch = 0
        
    def load_weights(self, load_path):
        checkpoint = torch.load(load_path)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']
        
    def save_weights(self,epoch, save_path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()}, 
            save_path)
       
    def current_snapshot_name(self):
        from time import gmtime, strftime
        import socket

        hostname = socket.gethostname()

        date = strftime("%b%d_", gmtime())
        clock = strftime("%X", gmtime())
        now = clock.split(":")
        now = date+'-'.join(now)

        name = now+"_"+hostname
        return name

class ConvNet(nn.Module):
    def __init__(self,outputs,image_shape):
        super(ConvNet, self).__init__()
        img_size = list(image_shape)
        img_size = torch.Size([1] + img_size)
        empty = torch.zeros(img_size)
        
        channels = 3
        kernel = 3
        padding = 1
        self.conv1 = nn.Sequential(nn.Conv2d(image_shape[0],
                                             out_channels = channels,
                                             kernel_size = kernel,
                                             padding = padding),
                                  nn.BatchNorm2d(channels),
                                  nn.MaxPool2d(2),
                                  nn.ReLU())
        units = self.conv1(empty).numel()
        print("units after conv", units)
        self.fc = nn.Sequential(nn.Linear(units, outputs))
        print("fc parameters: ",sum(p.numel() for p in self.fc.parameters()))
    
    def forward(self, x):
        #x: batch, channel, height, width
        batch_size = len(x)
        out = self.conv1(x)
        out = out.reshape((batch_size,-1))
        out = self.fc(out)
        return out