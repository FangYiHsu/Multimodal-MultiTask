import os
import torch
import torch.nn as nn
import torch.optim as optim

from time import time
from utils import AverageMeter
from HPRNN_NAM_LeakyReLU import HPRNN
from configs import configs
from MS_SSIM_L1_LOSS_multi import ssim as ssim_criterion

from MultiNet import *
from radar_echo_CREF import load_data_CREF

import warnings
warnings.filterwarnings("ignore")

device = configs.device

def train(model, train_loader, optimizer):
    model.train()
    model.to(device)
    train_loss_sum = 0
    num_of_batch_size = train_loader.step_per_epoch
    l1_criterion = nn.L1Loss()
    rmse_criterion = nn.MSELoss(reduction="mean")
    start = time()
    
    for index in range(num_of_batch_size):

        data_OBS, target = train_loader.generator_getClassifiedItems_3(index, "Sun_Moon_Lake")
        # data_OP, target_OP = train_loader.generator_getClassifiedItems_OP(index)
        data_error, target_error, real_op, real_obs = train_loader.generator_getClassifiedItems_Error(index)
        
        data_OBS = torch.FloatTensor(data_OBS).to(device) 
        real_op = torch.FloatTensor(real_op).to(device) 
        target = torch.FloatTensor(target).to(device)
               
        optimizer.zero_grad()
        Multi_output = torch.squeeze(model(data_OBS, real_op),0)
        target = torch.squeeze(target,2).permute(1,0,2,3)
        
        net_loss = 0
        for i,t in enumerate([0,5,11]):
            ssim_loss = torch.clamp(1-ssim_criterion(Multi_output[i], target[t], 255.0), min=0, max=1)
            l1_loss = l1_criterion(Multi_output[i], target[t])
            rmse_loss = rmse_criterion(Multi_output[i], target[t])
            loss = 1.0*ssim_loss + 0.1*torch.mean(l1_loss) + rmse_loss
            net_loss += loss
        net_loss = net_loss/3
        
        loss_meter = AverageMeter()
        loss_meter.update(net_loss.data.item(), data_OBS.size(0))
        net_loss.backward()
        optimizer.step()
        
        train_loss_sum += loss_meter.val

    train_loss = train_loss_sum / num_of_batch_size

    print("Epoch Time Cost: ", time()-start)
    print("train_num_of_batch_size", num_of_batch_size)
    print("train_loss", train_loss)

    return train_loss

def test(model, test_loader):
    model.eval()
    model.to(device)

    test_loss = 0
    l1_criterion = nn.L1Loss()
    rmse_criterion = nn.MSELoss(reduction="mean")
    num_of_batch_size = test_loader.step_per_epoch
    
    with torch.no_grad():
        for index in range(num_of_batch_size):
            data_OBS, target = test_loader.generator_getClassifiedItems_3(index, "Sun_Moon_Lake")
            # data_OP, target_OP = test_loader.generator_getClassifiedItems_OP(index)
            data_error, target_error, real_op, real_obs = test_loader.generator_getClassifiedItems_Error(index)
            
            data_OBS = torch.FloatTensor(data_OBS).to(device) 
            real_op = torch.FloatTensor(real_op).to(device) 
            target = torch.FloatTensor(target).to(device)

            Multi_output = torch.squeeze(model(data_OBS, real_op),0)
            target = torch.squeeze(target,dim=2).permute(1,0,2,3)

            net_loss = 0
            for i,t in enumerate([0,5,11]):
                ssim_loss = torch.clamp(1-ssim_criterion(Multi_output[i], target[t], 255.0), min=0, max=1)
                l1_loss = l1_criterion(Multi_output[i], target[t])
                rmse_loss = rmse_criterion(Multi_output[i], target[t])
                loss = 1.0*ssim_loss + 0.1*torch.mean(l1_loss) + rmse_loss
                net_loss += loss
            net_loss = net_loss/3

            test_loss_meter = AverageMeter()
            test_loss_meter.update(net_loss.data.item(), data_OBS.size(0))
            test_loss += net_loss.data.item()

    test_loss /= num_of_batch_size
    print("num_of_batch_size=",num_of_batch_size)
    print('\nTest set: Average loss: {:.4f}'.format(test_loss))
    return test_loss

use_cuda = torch.cuda.is_available()
torch.manual_seed(configs.seed)

save_path = configs.save_path
train_kwargs = {'batch_size': configs.batch_size}
test_kwargs = {'batch_size': configs.batch_size_test}
if use_cuda:
    cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
    
# OBS
obs_model_path = "model50_loss0.36020187107124574.pkl"
model_HPRNN_1 = HPRNN(64).to(device)
model_HPRNN_1.load_state_dict(torch.load(obs_model_path))
obs_hprnn = model_HPRNN_1.to(device)
for param in obs_hprnn.parameters():
    param.requires_grad = False
    
radar_echo_storage_path_OBS = "E:/Radar/OBS/2020/"
radar_echo_storage_path_OP = "E:/Radar/OP/2020/"

def main():
    
    load_radar_echo_df_path = "C:/Users/Weather/Desktop/CREF_2018_2021_120min.pkl"
     
    # 用全部資料測試    
    radar = load_data_CREF(radar_echo_storage_path_OBS=radar_echo_storage_path_OBS, 
                        radar_echo_storage_path_OP=radar_echo_storage_path_OP,       
                        load_radar_echo_df_path=load_radar_echo_df_path,
                        input_shape=[640, 640],
                        output_shape=[512, 512],
                        period=configs.input_length,
                        predict_period=configs.output_length,
                        places=["Sun_Moon_Lake"],
                        random=False,
                        date_range= configs.train_date,
                        test_date= configs.test_date)
    train_loader = radar.generator(
        'train', batch_size=configs.batch_size)
    test_loader = radar.generator(
        'val', batch_size=configs.batch_size)

    # Multi
    model = MultiTaskModel(obs_hprnn).to(device)
    optimizer = optim.Adam(model.parameters(),lr=configs.lr)

    for epoch in range(1, configs.epochs + 1):
        print("========================== Epoch now: ",epoch)
        
        train_loss = train(model, train_loader, optimizer)
        test_loss = test(model, test_loader)

        loss_file = save_path + 'loss.txt'
        with open(loss_file,'a') as file_obj:
            file_obj.write("-----itr ="+str(epoch)+"----- \n")
            file_obj.write("model train loss " + str(train_loss)  + '\n' )
            file_obj.write("model test loss " + str(test_loss)  + '\n')

        if (epoch >=1):
            print("saving model...")
            path = os.path.join(save_path,'model{}_loss{}.pkl'.format(str(epoch), str(train_loss)))
            torch.save(model.state_dict(),path)

# 用一筆資料測試    
def main_one():

    load_radar_echo_df_path = "CREF_2020_test.pkl" #None #"E:/multi_165/CREF_2020_test_0527.pkl"
     
    radar = load_data_CREF(radar_echo_storage_path_OBS=radar_echo_storage_path_OBS, 
                        radar_echo_storage_path_OP=radar_echo_storage_path_OP,       
                        load_radar_echo_df_path=load_radar_echo_df_path,
                        input_shape=[640, 640],
                        output_shape=[512, 512],
                        period=configs.input_length,
                        val_split=0, #測試一筆時才加入
                        predict_period=configs.output_length,
                        places=["Sun_Moon_Lake"],
                        random=False,
                        date_range= configs.one_date,
                        test_date= None) #configs.one_test_date)
    train_loader = radar.generator(
        'train', batch_size=configs.batch_size)
    # test_loader = radar.generator(
    #     'test', batch_size=configs.batch_size)
    
    # Multi
    # model = Net_3(obs_hprnn).to(device)
    model = MultiTaskModel(obs_hprnn).to(device)
    optimizer = optim.Adam(model.parameters(),lr=configs.lr)

    for epoch in range(1, configs.epochs + 1):
        print("========================== Epoch now: ",epoch)
        
        train_loss = train(model, train_loader, optimizer)
        # test_loss = test(model, test_loader)

        loss_file = save_path + 'loss.txt'
        with open(loss_file,'a') as file_obj:
            file_obj.write("-----itr ="+str(epoch)+"----- \n")
            file_obj.write("model train loss " + str(train_loss)  + '\n' )
            # file_obj.write("model test loss " + str(test_loss)  + '\n')

        if epoch >=10:
            print("saving model...")
            path = os.path.join(save_path,'model{}_loss{}.pkl'.format(str(epoch), str(train_loss)))
            torch.save(model.state_dict(),path)

if __name__ == '__main__':
    # main_one()
    main()
