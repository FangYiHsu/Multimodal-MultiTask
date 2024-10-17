import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedBackbone(nn.Module):
    def __init__(self):
        super(SharedBackbone, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        return x

class Head_0(nn.Module):
    def __init__(self):
        super(Head_0, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return x

class Head_5(nn.Module):
    def __init__(self):
        super(Head_5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return x

class Head_11(nn.Module):
    def __init__(self):
        super(Head_11, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return x

class MultiTaskModel(nn.Module):
    def __init__(self,obs_model):
        super(MultiTaskModel, self).__init__()
        self.feature1 = obs_model
        self.shared_backbone = SharedBackbone()
        self.head_0 = Head_0()
        self.head_5 = Head_5()
        self.head_11 = Head_11()

    def forward(self, x,y):
        x = self.feature1(x)
        fusion = torch.cat([x.squeeze(0), y.squeeze(0)], dim=1) # 12,2,512,512
        shared_features = self.shared_backbone(fusion) #12,128,512,512
        head_0_output = self.head_0(shared_features[0].unsqueeze(0))
        head_5_output = self.head_0(shared_features[5].unsqueeze(0))
        head_11_output = self.head_0(shared_features[11].unsqueeze(0))
        
        
        return torch.cat([head_0_output, head_5_output, head_11_output],dim=0)


class Net(nn.Module):
    def __init__(self, obs_model, op_model, error_model):
        super(Net, self).__init__()
    
        self.feature1 = obs_model
        self.feature2 = op_model
        self.feature3 = error_model

        self.conv1 = nn.Conv3d(in_channels=3, out_channels=24, kernel_size=3, stride=1, padding=1)#1,8
        self.bn1 = nn.BatchNorm3d(24)
        self.relu = nn.LeakyReLU()

        self.conv2 = nn.Conv3d(in_channels=24, out_channels=18, kernel_size=3, stride=1, padding=1)#1,8
        self.bn2 = nn.BatchNorm3d(18)

        self.conv3 = nn.Conv3d(in_channels=18, out_channels=1, kernel_size=3, stride=1, padding=1)#1,8
        self.bn3 = nn.BatchNorm3d(1)

    def forward(self, x, y, z):

        # in :1,6,1,512,512 
        # out:1,12,1,512,512
        x1 = self.feature1(x) # OBS
        x2 = self.feature2(y) # OP
        x3 = self.feature3(z) # Error

        cat = torch.cat([x1,x2,x3],dim=2).permute(0,2,1,3,4) #1,3,12,512,512
        x4 = self.relu(self.bn1(self.conv1(cat)))
        x4 = self.relu(self.bn2(self.conv2(x4)))
        x4 = self.relu(self.bn3(self.conv3(x4)))
        print("x4",x4.size())
        x4 = x4.permute(0,2,1,3,4) # 1,12,1,512,512
        # x4 = F.adaptive_avg_pool3d(x4, (1, 512, 512)) # 1,12,1,512,512
        
        return x1,x2,x3,x4
    

class Net_2(nn.Module):
    def __init__(self, obs_model, op_model, error_model,decoder):
        super(Net_2, self).__init__()
    
        self.feature1 = obs_model
        self.feature2 = op_model
        self.feature3 = error_model
        self.feature4 = decoder

        self.conv1 = nn.Conv3d(in_channels=36, out_channels=24, kernel_size=3, stride=1, padding=1)#1,8
        self.bn1 = nn.BatchNorm3d(24)

        self.conv2 = nn.Conv3d(in_channels=24, out_channels=18, kernel_size=3, stride=1, padding=1)#1,8
        self.bn2 = nn.BatchNorm3d(18)

        self.conv3 = nn.Conv3d(in_channels=18, out_channels=6, kernel_size=3, stride=1, padding=1)#1,8
        self.bn3 = nn.BatchNorm3d(6)

        # self.conv4 = nn.Conv2d(12, 6, kernel_size=1)
        self.relu = nn.LeakyReLU()

    def forward(self, x, y, z):

        # in :1,6,1,512,512 
        # out:1,12,1,512,512
        x1 = self.feature1(x) # OBS
        x2 = self.feature2(y) # OP
        x3 = self.feature3(z) # Error

        cat = torch.cat([x1,x2,x3],dim=1) # 1,36,1,512,512
        x4 = self.relu(self.bn1(self.conv1(cat)))
        x4 = self.relu(self.bn2(self.conv2(x4)))
        x4 = self.relu(self.bn3(self.conv3(x4))) # 1,6,1,512,512

        # x4 = self.conv4(x4.squeeze(2))
        x4 = self.feature4(x4)
        
        return x1,x2,x3,x4

class FusionModule(nn.Module):
    def __init__(self):
        super(FusionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x

class OutputModule(nn.Module):
    def __init__(self):
        super(OutputModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x

class Net_3(nn.Module):
    def __init__(self, obs_model):
        super(Net_3, self).__init__()
        self.feature1 = obs_model
        self.fusion_module = FusionModule()
        self.output_module_list = nn.ModuleList([OutputModule() for _ in range(12)])

    def forward(self, x, y):
        all_output = []
        for batch in range(x.shape[0]):
            x1 = self.feature1(x[batch].unsqueeze(0)) 
            x3 = torch.cat([x1.squeeze(0), y[batch].squeeze(0)], dim=1)  # 12, 2, 512, 512
            x3 = F.leaky_relu(self.fusion_module(x3))
            
            output = []
            for i, module in enumerate(self.output_module_list):
                output.append(module(x3[i].unsqueeze(0)))
                
            all_output.append(torch.cat(output, dim=0).unsqueeze(0))
        return torch.cat(all_output, dim=0)  # 1, 12, 1, 512, 512
    
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, query, key, value):
        attn_output, _ = self.multihead_attn(query, key, value)
        return attn_output

class Net_4(nn.Module):
    def __init__(self, model, num_heads):
        super(Net_4, self).__init__()
        self.attn = CrossAttention(512, num_heads)
        self.fc = nn.Linear(512, 512)
        self.hprnn = model  # 假設已有HPRNN模型

    def forward(self, obs, op, error):
        obs = torch.squeeze(obs)
        op = torch.squeeze(op)
        error = torch.squeeze(error)
        
        obs = self.attn(obs, op, op)
        op = self.attn(op, obs, obs)
        error = self.attn(error, obs, obs)
        print(obs.shape, op.shape, error.shape)
        fused_feat = obs + op + error
        fused_feat = self.fc(fused_feat)
        fused_feat = torch.unsqueeze(fused_feat,0)
        fused_feat = torch.unsqueeze(fused_feat,2)
        
        output = self.hprnn(fused_feat)
        return output