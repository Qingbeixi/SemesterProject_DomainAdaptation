from models.repetition import *
import torch
from models.VAE import *
from models.ExtractorRegressor import *
import numpy as np
import torch.optim as optim
from myDataset import *
from torchvision import transforms, datasets
import os
import argparse
import torch.optim
from helps import *
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import json
from models.VAE_rec import *

"""get the path to files"""
current_dir = os.path.dirname(os.path.abspath(__file__))
data_filedir = os.path.join(current_dir, 'N-CMAPSS')
data_filepath = os.path.join(current_dir, 'N-CMAPSS', 'N-CMAPSS_DS03-012.h5')
sample_dir_path = os.path.join(data_filedir, 'Samples_whole')
model_temp_path = os.path.join(current_dir, 'Models', 'oned_cnn_rep.h5')
tf_temp_path = os.path.join(current_dir, 'TF_Model_tf')
pic_dir = os.path.join(current_dir, 'Figures')


def vae_loss(x_hat, x, mu, logvar):
    # Reconstruction loss (RMSE)
    reconstruction_loss = torch.sqrt(torch.mean((x_hat - x)**2))
    
    # KL divergence
    kl_divergence = - 0.0 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    loss = reconstruction_loss + kl_divergence
    
    return loss

def main():
    """add parameter"""
    parser = argparse.ArgumentParser(description='sample creator')
    parser.add_argument('-model', type=str, default='ExtractorRegressor', help='model type to choose')
    parser.add_argument('-task', type=str, default='s', help='condition of domain adaptation')
    parser.add_argument('--sampling', type=int, default=10, help='sub sampling of the given data. If it is 10, then this indicates that we assumes 0.1Hz of data collection')
    parser.add_argument('-w', type=int, default=50, help='sequence length') # required=True
    parser.add_argument('-s', type=int, default=1, help='stride of filter')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-bs', type=int, default=256, help='batch size')
    parser.add_argument('-sub', type=int, default=10, help='subsampling stride')
    parser.add_argument('-ep', type=int, default=20, help='max epoch')
    parser.add_argument('-load', type=int, default=False, help='whether load previous model')
    args = parser.parse_args()

    """define each group type""" # 
    units_small = [1,5,9,12,14] # 
    units_medium = [2,3,4,7,15] # 
    units_long = [6,8,10,11,13] # 
    EOF = [72,73,67,60,93,63,80,71,84,66,59,93,77,76,67]
    domainType = args.task
    win_len = args.w
    win_stride = args.s
    lr = args.lr
    ep = args.ep
    bs = args.bs
    sub = args.sub
    sampling = args.sampling
    load_status = args.load
    

    sample_dict = {}
    label_dict = {}
    units_all = [units_small,units_medium,units_long] 
    for i,units in enumerate(units_all):
        sample_list = []
        sample_label_list = []
        for id,index in enumerate(units):
            sample_array, label_array = load_array (sample_dir_path, index, win_len, win_stride, sampling)
            sample_array = sample_array[::sub]
            label_array = label_array[::sub]
            sample_list.append(sample_array)
            sample_label_list.append(label_array/EOF[index-1]) # normalize to 0-1

        X_sample = np.concatenate(sample_list)
        y_sample_label = np.concatenate(sample_label_list).reshape(-1,1)
        print(X_sample.shape)
        print(y_sample_label.shape)
        sample_dict[i] = X_sample
        label_dict[i] = y_sample_label
        
        """release memory"""
        release_list(sample_list)
        release_list(sample_label_list)
        sample_list = []
        sample_label_list = []

    """use the domainType to construct the train and validation set"""
    # domainType "s to m"
    train_str = domainType[0]
    str_map = {"s":0,"m":1,"l":2}
    X_train = sample_dict[str_map[train_str]]

    train_dataset = TurbineDatasetVAE(X_train)
    model = AE()

    optimizer = optim.Adam(model.parameters(), lr=lr, betas = (0.9,0.999), eps=1e-07, amsgrad=True)
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("We use the device",device)
    # Move the model to the GPU
    model.to(device)
    # Train the model
    num_epochs = ep
    criterion = nn.MSELoss()

    """Train the model"""
    print("We apply the domain adaptation for",domainType)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    train_document = [] # document the loss, and save to file
   
    # start training loop
    for epoch in range(ep):
        model.train()
        train_loss = 0
        
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            x_hat = model(data)
            loss = torch.sqrt(torch.mean((x_hat - data)**2))
            loss.backward()
            train_loss += loss.item()* data.size(0)
            
            optimizer.step()
        
        train_loss /= len(train_loader.dataset)
        print('Epoch: {} Average loss: {:.8f}'.format(epoch+1, train_loss))
    
    with torch.no_grad():
        X_train = torch.tensor(X_train).to(device)
        X_hat = model(X_train)
        print(X_hat[0][0])
        print(X_train[0][0])   
        print(torch.sqrt(torch.mean((X_train - X_hat)**2)))

        feature_rmse = torch.sqrt(torch.mean((X_train - X_hat)**2, dim=[0,1]))
        # print the RMSE for each feature
        for i, rmse in enumerate(feature_rmse):
            print("Feature {}: {:.4f}".format(i+1, rmse))
    
    torch.save(model.state_dict(), "models/ae_model_params.pth")

if __name__ == '__main__':
    main()