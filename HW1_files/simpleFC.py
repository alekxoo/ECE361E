import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse
import random
import numpy as np

# Argument parser
parser = argparse.ArgumentParser(description='ECE361E HW1 - SimpleFC')
# Define the mini-batch size, here the size is 128 images per batch
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
# Define the number of epochs for training
parser.add_argument('--epochs', type=int, default=25, help='Number of epoch to train')
# Define the learning rate of your optimizer
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
args = parser.parse_args()

# The size of input features
input_size = 28 * 28
# The number of target classes, you have 10 digits to classify
num_classes = 10

# NOTE: addition to recognize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Always make assignments to local variables from your args at the beginning of your code for better
# control and adaptability
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr

# Each experiment you will do will have slightly different results due to the randomness
# of 1. the initialization value for the weights of the model, 2. sampling batches of training data
# 3. numerical algorithms for computation (in CUDA.) In order to have reproducible results,
# we have fixed a random seed to a specific value such that we "control" the randomness.
random_seed = 1
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
g = torch.Generator()
g.manual_seed(random_seed) # for data loader shuffling

# MNIST Dataset (Images and Labels)
train_dataset = dsets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='data', train=False, transform=transforms.ToTensor())

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, generator=g)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Define your model
class SimpleFC(nn.Module):
    def __init__(self, input_size, num_classes, dropout):
        super(SimpleFC, self).__init__()
        self.linear1 = nn.Linear(input_size, 512)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(512, 256)
        self.dropout2= nn.Dropout(dropout)
        self.linear3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(dropout)
        self.linear4 = nn.Linear(128, num_classes)

    # Your model only contains a single linear layer
    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = self.dropout1(out)
        out = F.relu(self.linear2(out))
        out = self.dropout2(out)
        out = F.relu(self.linear3(out))
        out = self.dropout3(out)
        out = self.linear4(out)
        return out

dropout_rate = [0.0, 0.2, 0.5, 0.8]

for dropout in dropout_rate:
    print("\ndropout rate: ", dropout)
    model = SimpleFC(input_size, num_classes, dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define your loss and optimizer
    criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        train_correct = 0
        train_total = 0
        train_loss = 0
        # Sets the model in training mode.
        model = model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            # NOTE
            images, labels = images.to(device), labels.to(device) 
            # Here we vectorize the 28*28 images as several 784-dimensional inputs
            images = images.view(-1, input_size)
            # Sets the gradients to zero
            optimizer.zero_grad()
            # The actual inference
            outputs = model(images)
            # Compute the loss between the predictions (outputs) and the ground-truth labels
            loss = criterion(outputs, labels)
            # Do backpropagation to update the parameters of your model
            loss.backward()
            # Performs a single optimization step (parameter update)
            optimizer.step()
            train_loss += loss.item()
            # The outputs are one-hot labels, we need to find the actual predicted
            # labels which have the highest output confidence
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            # Print every 100 steps the following information
            # if (batch_idx + 1) % 100 == 0:
            #     print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f Acc: %.2f%%' % (epoch + 1, num_epochs, batch_idx + 1,
            #                                                                     len(train_dataset) // batch_size,
            #                                                                     train_loss / (batch_idx + 1),
            #                                                                     100. * train_correct / train_total))       
            if (batch_idx + 1) == 468:
                print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f Acc: %.2f%%' % (epoch + 1, num_epochs, batch_idx + 1,
                                                                                len(train_dataset) // batch_size,
                                                                                train_loss / (batch_idx + 1),
                                                                                100. * train_correct / train_total))       

        # Testing phase
        test_correct = 0
        test_total = 0
        test_loss = 0
        # Sets the model in evaluation mode
        model = model.eval()
        # Disabling gradient calculation is useful for inference.
        # It will reduce memory consumption for computations.
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                # NOTE
                images, labels = images.to(device), labels.to(device) 
                # Here we vectorize the 28*28 images as several 784-dimensional inputs
                images = images.view(-1, input_size)
                # Perform the actual inference
                outputs = model(images)
                # Compute the loss
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                # The outputs are one-hot labels, we need to find the actual predicted
                # labels which have the highest output confidence
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        print('Test accuracy: %.2f %% Test loss: %.4f' % (100. * test_correct / test_total, test_loss / (batch_idx + 1)))