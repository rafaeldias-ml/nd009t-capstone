# torch imports
import torch.nn.functional as F
import torch.nn as nn


class BinaryClassifier(nn.Module):
    """
    Define a neural network that performs binary classification.
    The network should accept your number of features as input, and produce 
    a single sigmoid value, that can be rounded to a label: 0 or 1, as output.
    
    Notes on training:
    To train a binary classifier in PyTorch, use BCELoss.
    BCELoss is binary cross entropy loss, documentation: https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss
    """

    def __init__(self, input_features, hidden_dim1, hidden_dim2, hidden_dim3, output_dim, dropout_rate):
        """
        Initialize the model by setting up linear layers.
        Use the input parameters to help define the layers of your model.
        :param input_features: the number of input features in your training/test data
        :param hidden_dim: helps define the number of nodes in the hidden layer(s)
        :param output_dim: the number of outputs you want to produce
        :param dropout_rate: dropout rate
        """
        super(BinaryClassifier, self).__init__()

        # define any initial layers, here
        self.fc1 = nn.Linear(input_features, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.drop = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(hidden_dim3, output_dim)
        # sigmoid layer
        self.sig = nn.Sigmoid()        

    
    ## Define the feedforward behavior of the network
    def forward(self, x):
        """
        Perform a forward pass of our model on input features, x.
        :param x: A batch of input features of size (batch_size, input_features)
        :return: A single, sigmoid-activated value as output
        """
        
        # define the feedforward behavior
        out = F.relu(self.fc1(x)) # activation on hidden layer 1
        out = F.relu(self.fc2(out)) # activation on hidden layer 2
        out = F.relu(self.fc3(out)) # activation on hidden layer 3
        out = self.drop(out)
        out = self.fc4(out)
        return self.sig(out) # returning class score
    