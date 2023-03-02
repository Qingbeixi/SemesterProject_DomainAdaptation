
import os
import numpy as np
import torch
from random import shuffle
from matplotlib import pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import mean_squared_error
'''
load array from npz files
'''
current_dir = os.path.dirname(os.path.abspath(__file__))
pic_dir = os.path.join(current_dir, 'Figures')

def load_part_array (sample_dir_path, unit_num, win_len, stride, part_num):
    filename =  'Unit%s_win%s_str%s_part%s.npz' %(str(int(unit_num)), win_len, stride, part_num)
    filepath =  os.path.join(sample_dir_path, filename)
    loaded = np.load(filepath)
    return loaded['sample'], loaded['label']

def load_part_array_merge (sample_dir_path, unit_num, win_len, win_stride, partition):
    sample_array_lst = []
    label_array_lst = []
    print ("Unit: ", unit_num)
    for part in range(partition):
      print ("Part.", part+1)
      sample_array, label_array = load_part_array (sample_dir_path, unit_num, win_len, win_stride, part+1)
      sample_array_lst.append(sample_array)
      label_array_lst.append(label_array)
    sample_array = np.dstack(sample_array_lst)
    label_array = np.concatenate(label_array_lst)
    sample_array = sample_array.transpose(2, 0, 1)
    print ("sample_array.shape", sample_array.shape)
    print ("label_array.shape", label_array.shape)
    return sample_array, label_array


def load_array (sample_dir_path, unit_num, win_len, stride, sampling):
    filename =  'Unit%s_win%s_str%s_smp%s.npz' %(str(int(unit_num)), win_len, stride, sampling)
    filepath =  os.path.join(sample_dir_path, filename)
    loaded = np.load(filepath)

    return loaded['sample'].transpose(2, 0, 1), loaded['label']

# def rmse(y_true, y_pred):
#     return torch.sqrt(torch.mean(torch.square(y_pred - y_true), axis=-1))

class rmse_loss(nn.Module):
    def __init__(self):
        super(rmse_loss, self).__init__()

    def forward(self, y_pred, y_true):
        mse_loss = nn.functional.mse_loss(y_pred, y_true)
        rmse_loss = torch.sqrt(mse_loss)
        return rmse_loss

def shuffle_array(sample_array, label_array):
    ind_list = list(range(len(sample_array)))
    print("ind_list befor: ", ind_list[:10])
    print("ind_list befor: ", ind_list[-10:])
    ind_list = shuffle(ind_list)
    print("ind_list after: ", ind_list[:10])
    print("ind_list after: ", ind_list[-10:])
    print("Shuffeling in progress")
    shuffle_sample = sample_array[ind_list, :, :]
    shuffle_label = label_array[ind_list,]
    return shuffle_sample, shuffle_label

def figsave(history, win_len, win_stride, bs, lr, sub):
    fig_acc = plt.figure(figsize=(15, 8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Training', fontsize=24)
    plt.ylabel('loss', fontdict={'fontsize': 18})
    plt.xlabel('epoch', fontdict={'fontsize': 18})
    plt.legend(['Training loss', 'Validation loss'], loc='upper left', fontsize=18)
    plt.show()
    print ("saving file:training loss figure")
    fig_acc.savefig(pic_dir + "/training_w%s_s%s_bs%s_sub%s_lr%s.png" %(int(win_len), int(win_stride), int(bs), int(sub), str(lr)))
    return



# def get_flops(model):
#     concrete = tf.function(lambda inputs: model(inputs))
#     concrete_func = concrete.get_concrete_function(
#         [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
#     frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
#     with tf.Graph().as_default() as graph:
#         tf.graph_util.import_graph_def(graph_def, name='')
#         run_meta = tf.compat.v1.RunMetadata()
#         opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
#         flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
#         return flops.total_float_ops




def scheduler(epoch, lr):
    if epoch == 30:
        print("lr decay by 10")
        return lr * 0.1
    elif epoch == 70:
        print("lr decay by 10")
        return lr * 0.1
    else:
        return lr



def release_list(a):
   del a[:]
   del a