import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from nav_gym.learning.modules.submodules.mlp_modules import MLP
from nav_gym.learning.utils.utils import get_graph_feature
from torch.autograd import Variable
from nav_gym.learning.modules.submodules.cnn_modules import Pool1DConv


class SimplePointNet(nn.Module):

    # input_shape = [seq_len, dim] --> has to be swapped to [dim, seq_len]
    def __init__(self, input_shape, channels, fc_shape, fc_output, activation_fn=nn.LeakyReLU):
        super().__init__()
        self.model = Pool1DConv(input_shape, channels, fc_shape, fc_output, 1, activation_fn)

    def forward(self, x):
        # TODO: different pooling methods configurable
        return self.model(x)
    

class MultiHeadAttention(nn.Module):
    def __init__(self, input_shape, encoder_dim, encoder_hidden, num_heads):
        super().__init__()

        self.input_shape = input_shape
        self.global_dim = encoder_dim
        self.num_heads = num_heads
        self.d_k = encoder_dim // self.num_heads

        self.query_encoder = nn.Sequential(
            nn.Linear(self.input_shape[1], encoder_hidden),
            # nn.ELU(),
            # nn.Linear(encoder_hidden, encoder_hidden),
            nn.ELU(),
            nn.Linear(encoder_hidden, encoder_dim),
        )
        self.value_encoder = nn.Sequential(
            nn.Linear(self.input_shape[1], encoder_hidden),
            # nn.ELU(),
            # nn.Linear(encoder_hidden, encoder_hidden),
            nn.ELU(),
            nn.Linear(encoder_hidden, encoder_dim),
        )
        self.key_encoder = nn.Sequential(
            nn.Linear(self.input_shape[1], encoder_hidden),
            # nn.ELU(),
            # nn.Linear(encoder_hidden, encoder_hidden),
            nn.ELU(),
            nn.Linear(encoder_hidden, encoder_dim),
        )

        self.layer_norm = nn.LayerNorm(encoder_dim)


    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        query = self.query_encoder(x)
        key = self.key_encoder(x)
        value = self.value_encoder(x)

        # Linear operation and split into N heads
        query = query.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # weight = torch.matmul(query, key.permute(0, 2, 1)) / key.shape[2]**0.5 # [num_robots/batch_size, max_num_neighbor, max_num_neighbors]
        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, value)

        # soft_weight = torch.softmax(weight, dim=-1)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads*self.d_k)
        # # global_encoding = torch.matmul(soft_weight, value).mean(dim=1) # [num_robots/batch_size, feat_dim]
        # output = torch.matmul(soft_weight, value) # [num_robots/batch_size, max_num_neighbor, feat_dim]
        output = torch.max(output, dim = 1)[0] # TODO: try others
        # output = torch.mean(output, dim = 1)
        return output
    

class DGCNN(nn.Module):
    """
    From https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py
    """
     
    def __init__(self, input_dim, emb_dim, k, output):
        super(DGCNN, self).__init__()
        self.k = k

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(64*2, emb_dim, kernel_size=1, bias=False),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.conv3 = nn.Sequential(nn.Conv1d(input_dim - 3, 32, kernel_size=1, bias=False),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=1, bias=False),
                                   nn.LeakyReLU(negative_slope=0.2))
    
        self.linear1 = nn.Linear((emb_dim+64)*2, 128, bias=False)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, output)

    def forward(self, x):
        batch_size = x.size(0)
        x_pos = x[:,:,:3]
        x_feature = x[:,:,3:]
        x_pos = x_pos.permute(0, 2, 1)
        x = get_graph_feature(x_pos, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        features = self.conv4(self.conv3(x_feature.permute(0,2,1)))
        feature1 = F.adaptive_max_pool1d(features, 1).view(batch_size, -1)
        feature2 = F.adaptive_avg_pool1d(features, 1).view(batch_size, -1)
        features = torch.cat((feature1, feature2), 1)
        x = torch.cat((x, features), dim=1)

        x = F.leaky_relu(self.linear1(x), negative_slope=0.2)
        x = F.leaky_relu(self.linear2(x), negative_slope=0.2)
        x = self.linear3(x)
        return x
    

class Conv3DNet(nn.Module):
    def __init__(self, d1, d2, d3, channels, kernel_sizes, output):
        super(Conv3DNet, self).__init__()

        self.d1 = d1
        self.d2 = d2
        self.d3 = d3

        # Define the 3D convolutional layers
        # For simplicity, I'll just use 2 convolutional layers, but you can adjust this as needed
        self.conv1 = nn.Conv3d(1, channels[0], kernel_size=kernel_sizes[0], stride=1, padding=1)  # Assuming input channel is 1
        self.conv2 = nn.Conv3d(channels[0], channels[1], kernel_size=kernel_sizes[1], stride=1, padding=1)

        # Determine the size of the output from the convolutional layers
        # This will be used as input size for the fully connected layer
        self.features_size = channels[1] * d1//4 * d2//4 * d3//4  # due to maxpool

        self.fc1 = nn.Linear(self.features_size, 256)
        self.fc2 = nn.Linear(256, output)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.shape[0], self.d1, self.d2, self.d3)
        x = x.unsqueeze(1)
        # 3D Convolutional layers
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # Flatten
        x = x.view(-1, self.features_size)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x