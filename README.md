# Siphon
Machine Learning Intelligence
Not meant for general use

Built with PyTorch

## Data

Contains modules for MNIST data parsing

Data structure in the repository can be manually generated with data_mapper.py and stored in data_structure.txt to document the file structure of a dataset. The data folder will always be contained at the root of the repository.
The Reservoir class stores the training and testing data and dataloaders. The training dataset is automatically split into training and validation datasets based on the given train/validation split. Training and Testing classes inherit Dataclass.

## Model
Current convolutional model is a small template for more complicated types of models.

## Training
Training class allows for the saving of model weights and provides a simple fit function based on the supplied data and model. The progress of the training can be monitored with training and validation loss monitoring. 

## Testing
Testing class tied everything together by running the training fit method for a specified number of epochs and computes the accuracy of the model on the test set. 


## Installing requirements

pip install -r requirements.txt