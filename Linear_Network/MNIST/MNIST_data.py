from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler,SequentialSampler
import gzip
import numpy as np
import time
import torch

class MNIST_IO(Dataset):
    def __init__(self, data_path, label_path):
        
        self.image_data = self.read_images(data_path)
        self.label_data = self.read_labels(label_path)
    
    
    def read_images(self,file):
        file = gzip.open(file,'rb')
        magic_number = int.from_bytes(file.read(4),"big")
        num_images = int.from_bytes(file.read(4), "big")
        rows = int.from_bytes(file.read(4), "big")
        cols = int.from_bytes(file.read(4), "big")
        
        buf = file.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images,1, rows, cols)
        return torch.tensor(data).cuda()
        
    def read_labels(self,file):
        file = gzip.open(file,'rb')
        magic_number = int.from_bytes(file.read(4),"big")
        num_labels = int.from_bytes(file.read(4), "big")
        start = time.time()
        buf = file.read(num_labels)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_labels, 1)
        return torch.tensor(data).cuda()
    
    
    def __getitem__(self, idx):
        image = self.image_data[idx]
        label = self.label_data[idx]
        sample = {'image': image, 
                  'label': label}
        return sample
    
    def __len__(self):
        return len(self.image_data)
    
class Reservoir(Dataset):
    def __init__(self,
                 training_data_path, 
                 training_labels_path, 
                 testing_data_path, 
                 testing_labels_path,
                ):
        
        self.training_pool = Training_Pool(training_data_path,
                                           training_labels_path,
                                           batch_size = 10,
                                           train_portion = 0.9)
        self.testing_pool = Testing_Pool(testing_data_path,testing_labels_path,batch_size = 1)
    

class Training_Pool(MNIST_IO):
    def __init__(self,
                 training_data_path, 
                 training_labels_path,
                 batch_size,
                 train_portion):
        """
        training_data_path specifies the path of the training data, file or folder location
        training_labels_path specifies the path of the training labels, file or folder location
        """
        super().__init__(training_data_path, training_labels_path)
        self.make_dataloaders(batch_size, len(self), train_portion)
        
        
    def make_dataloaders(self,batch_size, size, train_portion):
        
        # spliting the dataset
        indices = list(range(size))
        split = int(np.floor(train_portion * size))
        end = int(np.floor(size))
        
        train_indices, val_indices = indices[:split], indices[split:end]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        
        
        self.dataloader = DataLoader(self, batch_size = batch_size, num_workers = 0, sampler=train_sampler)
        self.validloader = DataLoader(self, batch_size = batch_size, num_workers = 0, sampler=valid_sampler)
        
        print("Total training stacks", len(self.dataloader))
        print("Total validation stacks",len(self.validloader))
        
class Testing_Pool(MNIST_IO):
    def __init__(self,
                 testing_data_path, 
                 testing_labels_path,
                 batch_size):
        """
        testing_data_path specifies the path of the testing data, file or folder location
        testing_labels_path specifies the path of the testing labels, file or folder location
        """
        
        super().__init__(testing_data_path, testing_labels_path)
        self.make_dataloaders(batch_size = batch_size, size = len(self))
        
    def make_dataloaders(self,batch_size, size, train=0.9, test=0.1):
        self.testloader = DataLoader(self, batch_size = batch_size, num_workers = 0)
        print("Total testing stacks", len(self.testloader))