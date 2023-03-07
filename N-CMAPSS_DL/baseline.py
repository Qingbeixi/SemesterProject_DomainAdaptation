# units: [2,5,10,11,14,15,16,18,20]
# small [14] 1-3h
# medium [15] 3-5h
# long [2,5,10,11,16,18,20] 5-7h

"We propose to train on one set and test on another"
"Document: sample numbers, losses(rmse) for train and test after each epoch"
from models.repetition import *
import torch
from models.VAE import *
from models.ExtractorRegressor import *
import numpy as np
import torch.optim as optim
from myDataset import TurbineDataset
from torchvision import transforms, datasets
import os
import argparse
import torch.optim
from helps import *
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import json

"""get the path to files"""
current_dir = os.path.dirname(os.path.abspath(__file__))
data_filedir = os.path.join(current_dir, 'N-CMAPSS')
data_filepath = os.path.join(current_dir, 'N-CMAPSS', 'N-CMAPSS_DS03-012.h5')
sample_dir_path = os.path.join(data_filedir, 'Samples_whole')
model_temp_path = os.path.join(current_dir, 'Models', 'oned_cnn_rep.h5')
tf_temp_path = os.path.join(current_dir, 'TF_Model_tf')
pic_dir = os.path.join(current_dir, 'Figures')

def main():
    """add parameter"""
    parser = argparse.ArgumentParser(description='sample creator')
    parser.add_argument('-model', type=str, default='ExtractorRegressor', help='model type to choose')
    parser.add_argument('-task', type=str, default='s to m', help='condition of domain adaptation')
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
    mymodel = args.model
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
    val_str = domainType[5]
    str_map = {"s":0,"m":1,"l":2}
    X_train = sample_dict[str_map[train_str]]
    y_train = label_dict[str_map[train_str]]
    X_test = sample_dict[str_map[val_str]]
    y_test = label_dict[str_map[val_str]]
    train_dataset = TurbineDataset(X_train,y_train)
    validate_dataset = TurbineDataset(X_test,y_test)

    """prepare the model for training"""
    if mymodel == "ExtractorRegressor": 
        model = FullModel() # receive(batch_size,50,20) 
        if load_status:
            model.load_state_dict(torch.load(os.path.join("models", mymodel+".pt")))
    elif mymodel == "DA1DCNN":
        model = DA1DCNN()
    else:
        model = VAERegressor()
    
    print(model)

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
    val_loader =  DataLoader(validate_dataset, batch_size=bs, shuffle=True)
    train_document = [] # document the loss, and save to file
    val_document = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            if mymodel == "VAE":
                y_pred, mu, logvar = model(x_batch)
                kl_divergence = -1e-5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # the factor is very important not to domain the loss
                loss = torch.sqrt(criterion(y_pred, y_batch)) + kl_divergence # cannot be in place
            else:
                y_pred = model(x_batch)
                loss = torch.sqrt(criterion(y_pred, y_batch))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x_batch.size(0)
        
        with torch.no_grad():
            test_loss = 0
            for (x_batch, y_batch) in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                if mymodel == "VAE":
                    y_pred, mu, logvar = model(x_batch)
                    kl_divergence = -1e-5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    # val_loss = torch.sqrt(criterion(y_pred, y_batch))
                    val_loss = torch.sqrt(criterion(y_pred, y_batch)) + kl_divergence # cannot be in place
                else:
                    y_pred = model(x_batch)
                    val_loss = torch.sqrt(criterion(y_pred, y_batch))
                test_loss += val_loss.item() * x_batch.size(0)
        
        epoch_train_loss = running_loss / len(train_dataset)
        epoch_val_loss = test_loss / len(validate_dataset)
        train_document.append(epoch_train_loss)
        val_document.append(epoch_val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train_RMSELoss: {epoch_train_loss:.4f}, Test_RMSELoss:{epoch_val_loss:.4f}")
    
    # test results
    print("RMSE for test")
    save_dict_loss = {}
    output_lst = []
    truth_lst = []
    with torch.no_grad():
        for index in units_all[str_map[val_str]]: # units for test
                print ("test idx: ", index)
                sample_array, label_array = load_array(sample_dir_path, index, win_len, win_stride, sampling)
                
                sample_array = sample_array[::sub]
                label_array = label_array[::sub]
                
                # estimator = load_model(model_temp_path)
                sample_tensor = torch.tensor(sample_array, dtype=torch.float32)
                sample_tensor = sample_tensor.to(device)
                if mymodel == "VAE": 
                    y_pred_test, _, _ = model(sample_tensor)
                else:
                    y_pred_test = model(sample_tensor)
                y_pred_test = y_pred_test * EOF[index-1]
                rms_temp = np.sqrt(mean_squared_error(y_pred_test.cpu(), label_array))
                print("the rms for test index {} is {}".format(index,rms_temp))
                
                # document testing loss results
                name = "Test Loss for unit " + str(index) 
                save_dict_loss[name] = float(rms_temp)
                output_lst.append(y_pred_test.cpu())
                truth_lst.append(label_array)
        output_array = np.concatenate(output_lst)[:, 0]
        trytg_array = np.concatenate(truth_lst)
        rms = np.sqrt(mean_squared_error(output_array, trytg_array))
        rms = round(rms, 2)
        print("rms for test",rms)    
                # print(output_lst[0].shape)
        
    with open('models' + '/%s_%s.json'%(mymodel,domainType), 'w') as f:
            print(model,file=f)
            print("We apply the domain adaptation for",domainType,file=f)
            print("Train shape",X_train.shape,file=f)
            print("Test shape",X_test.shape,file=f)
            for epoch in range(num_epochs):
                print(f"Epoch {epoch+1}/{num_epochs}, Train_RMSELoss: {train_document[epoch]:.4f}, Test_RMSELoss:{val_document[epoch]:.4f}",file=f)
            json.dump(save_dict_loss, f)
            print("rms for test",rms,f)  
    
    fig_verify = plt.figure(figsize=(24, 10))
    plt.plot(train_document, color="blue")
    plt.plot(val_document, color="green", linewidth=2.0)
    plt.title('Domain Adaptation Type %s training process' %str(domainType), fontsize=30)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.ylabel('RMSE', fontdict={'fontsize': 24})
    plt.xlabel('Epochs', fontdict={'fontsize': 24})
    plt.legend(['Train Loss', 'Val Loss'], loc='upper right', fontsize=28)
    fig_verify.savefig('Figures' + '/%s_%s.png'%(mymodel,domainType))

    torch.save(model.state_dict(), "models/ExtractorRegressor.pt")





    print("RMSE for train")
    with torch.no_grad():
        for index in units_all[str_map[train_str]]: # units for test
                print ("test idx: ", index)
                sample_array, label_array = load_array(sample_dir_path, index, win_len, win_stride, sampling)
                
                sample_array = sample_array[::sub]
                label_array = label_array[::sub]
                
                # estimator = load_model(model_temp_path)
                sample_tensor = torch.tensor(sample_array, dtype=torch.float32)
                sample_tensor = sample_tensor.to(device)

                if mymodel == "VAE": 
                    y_pred_test, _, _ = model(sample_tensor)
                else:
                    y_pred_test = model(sample_tensor)
                y_pred_test = y_pred_test * EOF[index-1]
                rms_temp = np.sqrt(mean_squared_error(y_pred_test.cpu(), label_array))
                print("the rms for train index {} is {}".format(index,rms_temp))

if __name__ == '__main__':
    main()