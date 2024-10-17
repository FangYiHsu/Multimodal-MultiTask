import torch,gc
import torch.nn as nn
# from models.Inception_ConvLSTMCell import Inception_ConvLSTMCell,ConvLSTM
# from models.Inception_ConvLSTMCell import Inception_ConvLSTMCell
from models.LeakyReLU.ConvLSTMCell import ConvLSTMCell

class RefinementCell(nn.Module):
    def __init__(self,input_dim,hidden_dim,kernel_size,bias):
        super(RefinementCell, self).__init__()
        self.conv = nn.Conv2d(
                    in_channels=input_dim, 
                    out_channels=hidden_dim, 
                    kernel_size=kernel_size,
                    padding=1
        )
        self.residual_block = ResidualBlock(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim
        )
        self.convlstm = ConvLSTMCell(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    # num_layers=1,
                    bias=bias
        )

    def forward(self,x,state):
        # print("x=", x.size())
        out = self.conv(x)
        out = self.residual_block(out)
        out = self.residual_block(out)
        h_out, c_out = self.convlstm(out,state)
        out = self.residual_block(h_out)
        out = self.residual_block(out)
        out = self.conv(x)
        return out, h_out, c_out

class Channel_Att(nn.Module):
    def __init__(self, channels, t=16):
        super(Channel_Att, self).__init__()
        self.channels = channels
      
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)


    def forward(self, x):
        residual = x

        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        x = torch.sigmoid(x) * residual #
        
        return x


class Att(nn.Module):
    def __init__(self, channels, out_channels=None, no_spatial=True):
        super(Att, self).__init__()
        self.Channel_Att = Channel_Att(channels)
        self.Spatial_Att = Channel_Att(channels)
  
    def forward(self, x):
        x_out1=self.Channel_Att(x)
        x_out1=self.Spatial_Att(x_out1)
 
        return x_out1

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.LeakyReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ResidualBlock(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_dim, 
                    out_channels=hidden_dim, 
                    kernel_size=3,
                    padding=1)
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, 
                    out_channels=hidden_dim, 
                    kernel_size=3,
                    padding=1)
        self.nam = Att(hidden_dim)

        # self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        # self.relu = nn.ReLU(inplace=True)
        # self.ca = ChannelAttention(input_dim)
        # self.sa = SpatialAttention()

    def forward(self,x):
        residual = x
        out = self.conv1(x)
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv2(x)
        out = self.bn(x)
        out = self.nam(out)
        # out = self.ca(out) * out
        # out = self.sa(out) * out
        # print("conv=", out.size())
        out += residual
        out = self.relu(out)
        return out