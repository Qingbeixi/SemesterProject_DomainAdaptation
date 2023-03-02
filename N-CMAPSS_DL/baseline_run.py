import torch
from models.VAE import *
from models.ExtractorRegressor import *
import numpy as np
import torch.optim as optim
from myDataset import TurbineDataset
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
import os
import argparse
import torch.optim
from helps import *
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

"get the path to files"
current_dir = os.path.dirname(os.path.abspath(__file__))
data_filedir = os.path.join(current_dir, 'N-CMAPSS')
data_filepath = os.path.join(current_dir, 'N-CMAPSS', 'N-CMAPSS_DS02-006.h5')
sample_dir_path = os.path.join(data_filedir, 'Samples_whole')
model_temp_path = os.path.join(current_dir, 'Models', 'oned_cnn_rep.h5')
tf_temp_path = os.path.join(current_dir, 'TF_Model_tf')
pic_dir = os.path.join(current_dir, 'Figures')


def main():
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='sample creator')
    parser.add_argument('-w', type=int, default=50, help='sequence length') # required=True
    parser.add_argument('-s', type=int, default=1, help='stride of filter')
    parser.add_argument('-f', type=int, default=10, help='number of filter')
    parser.add_argument('-k', type=int, default=10, help='size of kernel')
    parser.add_argument('-bs', type=int, default=256, help='batch size')
    parser.add_argument('-ep', type=int, default=30, help='max epoch')
    parser.add_argument('-pt', type=int, default=20, help='patience')
    parser.add_argument('-vs', type=float, default=0.1, help='validation split')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-sub', type=int, default=10, help='subsampling stride')
    parser.add_argument('--sampling', type=int, default=10, help='sub sampling of the given data. If it is 10, then this indicates that we assumes 0.1Hz of data collection')



    args = parser.parse_args()

    win_len = args.w
    win_stride = args.s
    partition = 3
    n_filters = args.f
    kernel_size = args.k
    lr = args.lr
    bs = args.bs
    ep = args.ep
    pt = args.pt
    vs = args.vs
    sub = args.sub
    sampling = args.sampling

    train_units_samples_lst =[]
    train_units_labels_lst = []
    output_lst = []
    truth_lst = []

    units_index_train = [2.0, 5.0, 10.0, 16.0, 18.0, 20.0]
    units_index_test = [11.0, 14.0, 15.0]

    for index in units_index_train:
        # print("Load data index: ", index)
        sample_array, label_array = load_array (sample_dir_path, index, win_len, win_stride, sampling)
        #sample_array, label_array = shuffle_array(sample_array, label_array)
        #print("sample_array.shape", sample_array.shape)
        #print("label_array.shape", label_array.shape)
        sample_array = sample_array[::sub]
        label_array = label_array[::sub]
        print("sub sample_array.shape", sample_array.shape)
        print("sub label_array.shape", label_array.shape)
        train_units_samples_lst.append(sample_array)
        train_units_labels_lst.append(label_array)
    

    X_train = np.concatenate(train_units_samples_lst)
    y_train = np.concatenate(train_units_labels_lst).reshape(-1,1)
    print ("samples are aggregated")

    release_list(train_units_samples_lst)
    release_list(train_units_labels_lst)
    train_units_samples_lst =[]
    train_units_labels_lst = []
    print("Memory released")

    #sample_array, label_array = shuffle_array(sample_array, label_array)
    # print("samples are shuffled")
    print("X_train.shape", X_train.shape)
    print("y_train.shape", y_train.shape)

    # # create dataset for loading
    # # transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = TurbineDataset(X_train,y_train)

    # # Initialize model and optimizer
    # model = VAERegressor(input_dim=1000, latent_dim=50, hidden_dim=200)
    model = FullModel() # receive(batch_size,50,20) 
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas = (0.9,0.999), eps=1e-07, amsgrad=True)
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("We use the device",device)
    # Move the model to the GPU
    model.to(device)
    # Train the model
    num_epochs = 30

    # torch.autograd.set_detect_anomaly(True)
    criterion = nn.MSELoss()
    for epoch in range(num_epochs):
        running_loss = 0.0
        dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        for i, (x_batch, y_batch) in enumerate(dataloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            # y_pred, mu, logvar = model(x_batch)
            y_pred = model(x_batch)
            # loss = F.mse_loss(y_pred, y_batch)
            loss = torch.sqrt(criterion(y_pred, y_batch))
            # kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            # loss = loss + kl_divergence # cannot be in place
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x_batch.size(0)
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, RMSELoss: {epoch_loss:.4f}")
    
    output_lst = []
    truth_lst = []

    torch.save(model.state_dict(), "model.pt")
    # model.load_state_dict(torch.load("model.pt"))
    
    with torch.no_grad():
        for index in units_index_test:
            print ("test idx: ", index)
            sample_array, label_array = load_array(sample_dir_path, index, win_len, win_stride, sampling)
            # estimator = load_model(tf_temp_path, custom_objects={'rmse':rmse})
            print("sample_array.shape", sample_array.shape)
            print("label_array.shape", label_array.shape)
            sample_array = sample_array[::sub]
            label_array = label_array[::sub]
            print("sub sample_array.shape", sample_array.shape)
            print("sub label_array.shape", label_array.shape)
            
            # estimator = load_model(model_temp_path)
            sample_tensor = torch.tensor(sample_array, dtype=torch.float32)
            sample_tensor = sample_tensor.to(device)
            # y_pred_test, _, _ = model(sample_tensor)
            y_pred_test = model(sample_tensor)

            output_lst.append(y_pred_test.cpu())
            truth_lst.append(label_array)
        
            print(output_lst[0].shape)

    # print(truth_lst[0].shape)
    # print(np.concatenate(output_lst).shape)
    # print(np.concatenate(truth_lst).shape)

    output_array = np.concatenate(output_lst)[:, 0]
    trytg_array = np.concatenate(truth_lst)
    print(output_array.shape)
    print(trytg_array.shape)
    rms = np.sqrt(mean_squared_error(output_array, trytg_array))
    print(rms)
    rms = round(rms, 2)
    print("rms for test",rms)

    for idx in range(len(units_index_test)):
        fig_verify = plt.figure(figsize=(24, 10))
        plt.plot(output_lst[idx], color="green")
        plt.plot(truth_lst[idx], color="red", linewidth=2.0)
        plt.title('Unit%s inference' %str(int(units_index_test[idx])), fontsize=30)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        plt.ylabel('RUL', fontdict={'fontsize': 24})
        plt.xlabel('Timestamps', fontdict={'fontsize': 24})
        plt.legend(['Predicted', 'Truth'], loc='upper right', fontsize=28)
        plt.show()
        fig_verify.savefig(pic_dir + "/unit%s_test_w%s_s%s_bs%s_lr%s_sub%s_rmse-%s.png" %(str(int(units_index_test[idx])),
                                                                                int(win_len), int(win_stride), int(bs),
                                                                                    str(lr), int(sub), str(rms)))


if __name__ == '__main__':
    main()