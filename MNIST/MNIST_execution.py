import os
import sys
import math
import gzip
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm_notebook

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler,SequentialSampler
from skimage import io, transform
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter

# Ignore warnings
import warnings
import time
warnings.filterwarnings("ignore")

from MNIST_data import *
from MNIST_model import Model

class Trainer():
    def __init__(self, model, dataset):
        self.model = model
        self.network = model.network
        self.device = model.device
        self.optimizer = model.optimizer
        self.criterion = model.criterion
        self.dataset = dataset
        self.save_path = "../snapshots/{}_{:.5f}_weights.pt"
        
        
    def initializeEpoch(self):
        self.summation = 0
        self.val_summation = 0
        self.validation_training = enumerate(self.dataset.training_pool.validloader)
    
    def fit(self, epochs, report_period, load_path):
        
        if load_path:
            self.model.load_weights(load_path)
            
        
        iters_trained = []
        training_losses = []
        validation_losses = []
        count = 0
        
        for epoch in range(self.model.start_epoch, self.model.start_epoch+epochs):
            self.initializeEpoch()
            dataloader = self.dataset.training_pool.dataloader
            for i_batch, sampled_batch in tqdm_notebook(enumerate(dataloader),
                                                       total=len(dataloader)):
                
                torch.cuda.empty_cache()
                self.model.network.train()
                images = sampled_batch['image'].to(self.device).float()
                labels = sampled_batch['label'].to(self.device).long()
                
                #backwards pass
                self.optimizer.zero_grad()
                prediction = self.model.network(images)
                
                #calculate loss
                labels = labels.reshape((10,))
                loss = self.criterion(prediction, labels)
                loss.backward()
                self.optimizer.step()
                
                #get batch losses
                val_i,batch = self.validationBatch()
                val_loss = self.score(batch)
                self.summation += float(loss)
                self.val_summation += float(val_loss)
                
                if i_batch % report_period == 0:
                    iters_trained.append(count)
                    average_loss = round(self.summation/float(i_batch+1),5)
                    average_val_loss = round(self.val_summation/float(i_batch+1),5)
                    training_losses.append(average_loss)
                    validation_losses.append(average_val_loss)
                count += 1
                
                    
            print("Epoch: "+str(epoch))
            print("Training Loss: "+str(average_loss))
            print("Validation Loss: "+str(average_val_loss))
            
            self.model.save_weights(epoch, self.save_path.format(epoch,average_loss))
            
            plt.plot(iters_trained,training_losses, label="training")
            plt.plot(iters_trained,validation_losses, label="validation")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.show()
        return self.model
            
        
        
    def validationBatch(self):
        try:
            val_i,batch = next(self.validation_training)
        except StopIteration:
            self.validation_training = enumerate(self.dataset.training_pool.validloader)
            val_i,batch = next(self.validation_training)
        return val_i,batch
        
                
    def score(self, sampled_batch):
        self.model.network.eval()
        #inputs and forward pass
        images = sampled_batch['image']
        images = images.to(self.device).float()
        labels = sampled_batch['label'].to(self.device).long()
        labels = labels.reshape((10,))

        #forward pass
        prediction = self.model.network(images)

        #calculate loss
        loss = self.criterion(prediction, labels)
        torch.cuda.empty_cache()
        return loss
        
        
class Tester():
    def __init__(self):
        self.dataset = Reservoir("../data/training/images/train-images-idx3-ubyte.gz",
                        "../data/training/labels/train-labels-idx1-ubyte.gz",
                        "../data/testing/images/t10k-images-idx3-ubyte.gz",
                        "../data/testing/labels/t10k-labels-idx1-ubyte.gz")
                        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('using device', self.device)
        image_shape = self.dataset.training_pool[0]['image'].shape
        print("Data size", len(self.dataset.training_pool))
        self.model = Model(self.device)
        
        print("number of parameters: ", sum(p.numel() for p in self.model.network.parameters()))
        
        
    
    def run_training(self,epochs, snapshot=""):
        report_period = 100
        
        trainer = Trainer(self.model, self.dataset)
        self.model = trainer.fit(epochs, report_period, snapshot)
        
    def test_accuracy(self):
        total = 0
        correct = 0

        self.model.network.eval()

        for i, batch in enumerate(self.dataset.testing_pool.testloader):
            image = batch['image'][0].float()
            image = image.reshape(1,1,28,28).to(self.device).float()
            label = batch['label'][0]
            output = torch.argmax(self.model.network(image))

            if output.item() == label.item():
                correct += 1
            total += 1

        print("Total: "+str(total))
        print("Correct: "+str(correct))

        print("Percent Correct: {}%".format((correct/total)*100))
        
        
tester = Tester()
tester.run_training(1)
tester.test_accuracy()
 

