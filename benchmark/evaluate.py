import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import os

def load_data_100k(path='./', delimiter='\t'):
    """
        Function for loadign a data from the dataset, its formating, 
        and constracting a representation on which the model will train
    """

    # Loading the data
    train_data = np.loadtxt(os.path.join(path, 'u1.base'), skiprows=0, delimiter=delimiter).astype('int32')
    test_data = np.loadtxt(os.path.join(path, 'u1.test'), skiprows=0, delimiter=delimiter).astype('int32')
    total_data = np.concatenate((train_data, test_data), axis=0)

    n_users = np.unique(total_data[:, 0]).size      # num of users
    n_movies = np.unique(total_data[:, 1]).size     # num of movies
    n_train_ratinigs = train_data.shape[0]          # num of training ratings
    n_test_ratings = test_data.shape[0]             # num of test ratings

    # Preparing a matrix where the data about the users preferences will be stored 
    train_ratings = np.zeros((n_movies, n_users), dtype='float32')
    test_ratings = np.zeros((n_movies, n_users), dtype='float32')

    # Filling the matrix
    for i in range(n_train_ratinigs):
        train_ratings[train_data[i, 1] - 1, train_data[i, 0] - 1] = train_data[i, 2]

    for i in range(n_test_ratings):
        test_ratings[test_data[i, 1] - 1, test_data[i, 0] - 1] = test_data[i, 2]

    # Creating masks for loss calculations 
    train_mask = np.greater(train_ratings, 1e-12).astype('float32')  # masks indicating non-zero entries
    test_mask = np.greater(test_ratings, 1e-12).astype('float32')

    return n_movies, n_users, train_ratings, train_mask, test_ratings, test_mask

def local_kernel(u, v):
    """
        Function for computing a local kernel of local kernelised weight matrix
    """
    dist = torch.norm(u - v, p=2, dim=2)
    hat = torch.clamp(1. - dist**2, min=0.)
    return hat

# Class of a layer of the final model that will make one pass trough the one FC with some regularizations(sparsity and l2 ones)
class KernelLayer(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_dim, lambda_s, lambda_2, activation=nn.Sigmoid()):
      super().__init__()
      self.W = nn.Parameter(torch.randn(n_inputs, n_hidden))
      self.u = nn.Parameter(torch.randn(n_inputs, 1, n_dim))
      self.v = nn.Parameter(torch.randn(1, n_hidden, n_dim))
      self.b = nn.Parameter(torch.randn(n_hidden))

      self.lambda_s = lambda_s
      self.lambda_2 = lambda_2

      nn.init.xavier_uniform_(self.W, gain=torch.nn.init.calculate_gain("relu"))
      nn.init.xavier_uniform_(self.u, gain=torch.nn.init.calculate_gain("relu"))
      nn.init.xavier_uniform_(self.v, gain=torch.nn.init.calculate_gain("relu"))
      nn.init.zeros_(self.b)
      self.activation = activation

    def forward(self, x):
      w_hat = local_kernel(self.u, self.v)
    
      sparse_reg = torch.nn.functional.mse_loss(w_hat, torch.zeros_like(w_hat))
      sparse_reg_term = self.lambda_s * sparse_reg
      
      l2_reg = torch.nn.functional.mse_loss(self.W, torch.zeros_like(self.W))
      l2_reg_term = self.lambda_2 * l2_reg

      W_eff = self.W * w_hat  # Applying local kernelised weight matrix
      y = torch.matmul(x, W_eff) + self.b
      y = self.activation(y)

      return y, sparse_reg_term + l2_reg_term

# Class of a submodule of the final model that makes a pass through several kernel layers with diffrent arguments given(+ dropout = 0.33)
class KernelNet(nn.Module):
    def __init__(self, n_users, n_hidden, n_dim, n_layers, lambda_s, lambda_2):
      super().__init__()
      layers = []
      for i in range(n_layers):
        if i == 0:
          layers.append(KernelLayer(n_users, n_hidden, n_dim, lambda_s, lambda_2))
        else:
          layers.append(KernelLayer(n_hidden, n_hidden, n_dim, lambda_s, lambda_2))
      layers.append(KernelLayer(n_hidden, n_users, n_dim, lambda_s, lambda_2, activation=nn.Identity()))
      self.layers = nn.ModuleList(layers)
      self.dropout = nn.Dropout(0.33)

    def forward(self, x):
      total_reg = 0.0
      for i, layer in enumerate(self.layers):
        x, reg = layer(x)
        
        if (i < len(self.layers) - 1):
          x = self.dropout(x)

        total_reg += reg

      return x, total_reg
    
class CompleteNet(nn.Module):
    def __init__(self, kernel_net, n_m, gk_size, dot_scale):
      super().__init__()
      self.gk_size = gk_size                # Global kernel size
      self.dot_scale = dot_scale            # Scaling factitor for a convolution
      self.local_kernel_net = kernel_net    # Pretrained "local" kernel net
      self.conv_kernel = torch.nn.Parameter(torch.randn(n_m, gk_size**2) * 0.1)
      nn.init.xavier_uniform_(self.conv_kernel, gain=torch.nn.init.calculate_gain("relu"))
      

    # Applying the global kernel, performing a convolution, and applying the local kernel
    def forward(self, x, x_local):
      gk = self.global_kernel(x_local, self.gk_size, self.dot_scale)
      x = self.global_conv(x, gk)
      x, global_reg_loss = self.local_kernel_net(x)
      return x, global_reg_loss

    # Process of average pooling with applying the global kernel(+ scaled factor if != 1.0)
    def global_kernel(self, input, gk_size, dot_scale):
      avg_pooling = torch.mean(input, dim=1)
      avg_pooling = avg_pooling.view(1, -1)

      gk = torch.matmul(avg_pooling, self.conv_kernel) * dot_scale
      gk = gk.view(1, 1, gk_size, gk_size)

      return gk

    # Performing a convolution...
    def global_conv(self, input, W):
      input = input.unsqueeze(0).unsqueeze(0)
      conv2d = nn.LeakyReLU()(F.conv2d(input, W, stride=1, padding=1))
      return conv2d.squeeze(0).squeeze(0)
    
def dcg_k(score_label, k):
    """
        Function that calculates DCG@K score
    """
    dcg, i = 0., 0
    for s in score_label:
        if i < k:
            dcg += (2**s[1] - 1) / np.log2(2 + i)
            i += 1
    return dcg

def ndcg_k(y_hat, y, k):
    """
        Function that calculates NDCG@K score
    """
    score_label = torch.stack([y_hat, y], axis=1).tolist()
    score_label = sorted(score_label, key=lambda d:d[0], reverse=True)
    score_label_ = sorted(score_label, key=lambda d:d[1], reverse=True)
    norm, i = 0., 0
    for s in score_label_:
        if i < k:
            norm += (2**s[1] - 1) / np.log2(2 + i)
            i += 1
    dcg = dcg_k(score_label, k)
    return dcg / norm

def call_ndcg(y_hat, y):
    """
        Function for calculating NDCG@K score for all the users
    """
    ndcg_sum, num = 0, 0
    y_hat, y = y_hat.T, y.T
    n_users = y.shape[0]

    for i in range(n_users):
        y_hat_i = y_hat[i][torch.where(y[i])]
        y_i = y[i][torch.where(y[i])]

        if y_i.shape[0] < 2:
            continue

        ndcg_sum += ndcg_k(y_hat_i, y_i, y_i.shape[0])  # user-wise calculation
        num += 1

    return ndcg_sum / num


if __name__ == "__main__":
    # Autochoice of a device where the model will be trained on 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Current device is:", device)

    relpath = os.path.dirname(__file__)
    data_path = os.path.join(relpath, "../data/raw/")

    # Data Load
    try:
        path = os.path.join(data_path, 'ml-100k/')
        n_movies, n_users, train_ratings, train_mask, test_ratings, test_mask = load_data_100k(path=path, delimiter='\t')
        train_r, test_r = torch.tensor(train_ratings, dtype=torch.double, device=device), torch.tensor(test_ratings, dtype=torch.float, device=device)
        train_m, test_m = torch.tensor(train_mask, dtype=torch.float, device=device), torch.tensor(test_mask, dtype=torch.float, device=device)
    except FileNotFoundError:
        print('Error: Unable to load data')

    """ Common hyperparameter settings: """

    n_hidden = 500  # size of hidden layers
    n_dim = 5       # inner AutoEncoder embedding size
    n_layers = 2    # number of hidden layers
    gk_size = 3     # size of square kernel for convolution

    """ Hyperparameters to tune for specific case: """

    max_epoch_p = 30    # max number of epochs for pretraining
    max_epoch_f = 1000  # max number of epochs for finetuning
    patience_p = 5      # number of consecutive rounds of early stopping condition before actual stop for pretraining
    patience_f = 10     # and finetuning
    tol_p = 1e-4        # minimum threshold for the difference between consecutive values of train rmse, used for early stopping, for pretraining
    tol_f = 1e-5        # and finetuning
    lambda_2 = 20.      # regularisation of number or parameters
    lambda_s = 0.006    # regularisation of sparsity of the final matrix
    dot_scale = 1       # dot product weight for global kernel

    # Load models from the "/models" directory 
    test_local_kernel = KernelNet(n_users, n_hidden, n_dim, n_layers, lambda_s, lambda_2).double().to(device)
    test_complete_model = CompleteNet(test_local_kernel, n_movies, gk_size, dot_scale).double().to(device)

    test_local_kernel.load_state_dict(torch.load(os.path.join(relpath, "../models/best_local_kernel.pt"), map_location=device))
    test_complete_model.load_state_dict(torch.load(os.path.join(relpath, "../models/best.pt"), map_location=device))

    # Switch to an "eval" mode of the model and making a prediction on a loaded data
    test_complete_model.eval()
    train_r_local, _ = test_local_kernel(train_r)
    pre, _ = test_complete_model(train_r, train_r_local)
    pre = pre.float()

    # Calulating RMSEs and NDCGs
    error = (test_m * (torch.clip(pre, 1., 5.) - test_r) ** 2).sum() / test_m.sum()           # test error
    test_rmse = np.sqrt(error.item())

    error_train = (train_m * (torch.clip(pre, 1., 5.) - train_r) ** 2).sum() / train_m.sum()  # train error
    train_rmse = np.sqrt(error_train.item())

    test_ndcg = call_ndcg(torch.clip(pre, 1., 5.), test_r)
    train_ndcg = call_ndcg(torch.clip(pre, 1., 5.), train_r)
    
    print(f'RMSE: {test_rmse}\nNDCG: {test_ndcg}')