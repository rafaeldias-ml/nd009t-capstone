import argparse
import json
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from sklearn.metrics import roc_auc_score


# imports the model in model.py by name
from model import BinaryClassifier

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BinaryClassifier(model_info['input_features'], 
                             model_info['hidden_dim1'],
                             model_info['hidden_dim2'],
                             model_info['hidden_dim3'],
                             model_info['output_dim'],
                             model_info['dropout_rate']
                            )

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # set to eval mode, could use no_grad
    model.to(device).eval()

    print("Done loading model.")
    return model

# Gets training data in batches from the training.csv file
def _get_train_data_loader(batch_size, training_dir):
    print("Get train data loader.")
    train_data = pd.read_csv(os.path.join(training_dir, "training.csv"), header=None, names=None)

    train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()
    train_x = torch.from_numpy(train_data.drop([0], axis=1).values).float()

    train_ds = torch.utils.data.TensorDataset(train_x, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)

# Gets validation data in batches from the validation.csv file
def _get_validation_data_loader(batch_size, training_dir):
    print("Get validation data loader.")
    val_data = pd.read_csv(os.path.join(training_dir, "validation.csv"), header=None, names=None)

    val_y = torch.from_numpy(val_data[[0]].values).float().squeeze()
    val_x = torch.from_numpy(val_data.drop([0], axis=1).values).float()

    val_ds = torch.utils.data.TensorDataset(val_x, val_y)

    return torch.utils.data.DataLoader(val_ds, batch_size=batch_size)


# Provided training function
def train(model, train_loader, test_loader, epochs, patience, criterion, optimizer, device, model_path):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    test_loader  - The PyTorch DataLoader that should be used during training for validation.
    patience     - The number of epochs without improvement to wait before early stopping
    epochs       - The total number of epochs to train for.
    criterion    - The loss function used for training. 
    optimizer    - The optimizer to use during training.
    device       - Where the model and data should be loaded (gpu or cpu).
    model_path   - Model path to save model when it improves
    """

    test_score = 0
    current_patience = patience
    # training loop is provided
    for epoch in range(1, epochs + 1):
        model.train() # Make sure that the model is in training mode.

        total_loss = 0

        for batch in train_loader:
            # get data
            batch_x, batch_y = batch

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            
            # get predictions from model
            y_pred = model(batch_x)
            
            # perform backprop
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data.item()
        print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(train_loader)))

        # validation
        score = test(model, device, test_loader)
        if score > test_score:
            test_score = score
            current_patience = patience
            print('Saving model ...')
            # Save the model parameters
            with open(model_path, 'wb') as f:
                torch.save(model.cpu().state_dict(), f)
        else:
            current_patience -= 1
            
        if current_patience == 0:
            print('Early stopping after %d epochs without improvement' % patience)
            break

def test(model, device, test_loader):
    """
    This is the test method that is called on epoch end, during training phase. The parameters
    passed are as follows:
    model        - The PyTorch model under training.
    test_loader  - The PyTorch DataLoader that should be used for validation.
    device       - Where the model and data should be loaded (gpu or cpu).
    """    
    #model in eval mode skips Dropout etc
    model.eval()
    y_true = []
    y_pred = []
    
    # set the requires_grad flag to false as we are in the test mode
    with torch.no_grad():
        for batch in test_loader:
            # get data
            batch_x, batch_y = batch

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            target = batch_y.numpy()
            pred = model(batch_x).numpy()

            y_true.extend(target.tolist()) 
            y_pred.extend(pred.reshape(-1).tolist())

    score = roc_auc_score(y_true, y_pred)

    print('ROC on test set is %.5f' % score)
    
    return score

if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--patience', type=int, default=5, metavar='N',
                        help='number of epochs without improvement before early stopping (default: 5)')    
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    
    # Model Parameters
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--l2_reg', type=float, default=0.001, metavar='LR',
                        help='l2 regularization (default: 0.001)')
    parser.add_argument('--input_features', type=int, default=2, metavar='IN',
                        help='number of input features to model (default: 2)')
    parser.add_argument('--hidden_dim1', type=int, default=10, metavar='H',
                        help='hidden dim1 of model (default: 10)')
    parser.add_argument('--hidden_dim2', type=int, default=10, metavar='H',
                        help='hidden dim2 of model (default: 10)')
    parser.add_argument('--hidden_dim3', type=int, default=10, metavar='H',
                        help='hidden dim3 of model (default: 10)')    
    parser.add_argument('--output_dim', type=int, default=1, metavar='OUT',
                        help='output dim of model (default: 1)')
    parser.add_argument('--dropout_rate', type=float, default=0.3, metavar='F',
                        help='dropout rate (default: 0.3)')

    # args holds all passed-in arguments
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)
    val_loader = _get_validation_data_loader(args.batch_size, args.data_dir)

    ## Build the model by passing in the input params
    model = BinaryClassifier(args.input_features,
                             args.hidden_dim1,
                             args.hidden_dim2,
                             args.hidden_dim3,
                             args.output_dim,
                             args.dropout_rate).to(device)

    ## Define an optimizer and loss function for training
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
    criterion = nn.BCELoss()

    model_path = os.path.join(args.model_dir, 'model.pth')
    # Trains the model (given line of code, which calls the above training function)
    train(model, train_loader, val_loader, args.epochs, args.patience, criterion, optimizer, device, model_path)

    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'input_features': args.input_features,
            'hidden_dim1': args.hidden_dim1,
            'hidden_dim2': args.hidden_dim2,
            'hidden_dim3': args.hidden_dim3,
            'output_dim': args.output_dim,
            'dropout_rate': args.dropout_rate,
        }
        torch.save(model_info, f)

