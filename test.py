import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from configs import configs
from skimage.metrics import structural_similarity

from area_20 import area_20
from visualize.Verification import Verification 
from visualize.visualized_pred import visualized_area_with_map, visualized_area_with_map_mae

from MultiNet import *
from metrics import FAR_POD
from HPRNN_NAM_LeakyReLU import HPRNN
from radar_echo_CREF import load_data_CREF
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")
obs_model_path = "model50_loss0.36020187107124574.pkl"

save_path = configs.save_path
device = configs.device

pkl_name = 'model40_loss0.3657704293727875.pkl' #改成所要的model
model_name = save_path + pkl_name
test_date = configs.pred_date

data_name_list = []
for each in test_date:
    data_name_list.append(datetime.strptime(each[0], '%Y-%m-%d %H:%M').strftime('%Y%m%d_%H%M'))
# data_name = datetime.strptime(configs.pred_date[0][0], '%Y-%m-%d %H:%M').strftime('%Y%m%d_%H%M')
print(data_name_list)
# 回波資料路徑
radar_echo_storage_path_OBS = "E:/FY/Radar/OBS/2020/"
radar_echo_storage_path_OP = "E:/FY/Radar/OP/2020/"
load_radar_echo_df_path = None #"CREF_2020_test.pkl" #"E:/save_pkl/2020_pred.pkl"

# 模型儲存路徑

def test(model, test_loader):
    model.load_state_dict(torch.load(model_name))
    model.to(device)
    model.eval()

    test_loss = 0
    s_loss_sum = 0
        
    with torch.no_grad():
        num_of_batch_size = test_loader.step_per_epoch
        print(num_of_batch_size)
        for index in range(num_of_batch_size):
            print("=====",data_name_list[index])
            save_test_path = save_path + '/{}/{}/'.format(pkl_name[:-4],data_name_list[index])
            if not os.path.isdir(save_test_path):
                os.makedirs(save_test_path) 
                
            data_OBS, target = test_loader.generator_getClassifiedItems_3(index, "Sun_Moon_Lake")
            data_error, target_error, real_op, real_obs = test_loader.generator_getClassifiedItems_Error(index)
            
            data_OBS = torch.FloatTensor(data_OBS).to(device) 
            real_op = torch.FloatTensor(real_op).to(device) 
            target = torch.FloatTensor(target).to(device)
            
            Multi_output = model(data_OBS, real_op)

            for i,t in enumerate([0,5,11]): #range(configs.output_length-1,-1,-1):
                print("draw t = ",t)
                vis_gx = np.array(torch.squeeze(Multi_output[i]).cpu())
                vis_gx[vis_gx <= 5] = 0 # clear dbz < 1
                vis_x = np.array(torch.squeeze(target[:, t, :, :, :]).cpu())
                
                print(vis_gx.shape, vis_x.shape)
                visualized_area_with_map(vis_gx, 'Sun_Moon_Lake', shape_size=[512,512], title='Multi_pred_{}'.format(t), savepath=save_test_path)
                # visualized_area_with_map(vis_x, 'Sun_Moon_Lake', shape_size=[512,512], title='vis_gt_{}'.format(t), savepath=save_test_path)
            
                # visualized_area_with_map_mae(vis_gx-vis_x, 'Sun_Moon_Lake', shape_size=[512,512], title='Multi_pred_ME{}'.format(t), savepath=save_test_path)

                # 評估指標
                for thrshold in [0.5,30,40]:
                    score, _ = structural_similarity(vis_x, vis_gx, full=True, channel_axis=True, data_range=vis_x.max()-vis_x.min())
                    csi, far, pod, bias, sr = FAR_POD(vis_x, vis_gx, thrshold)
                    
                    Evaluation_Index = save_test_path + '{}_Evaluation_Index_th={}.txt'.format(data_name_list[index],thrshold)
                    print(csi, far, pod, bias, sr)
                    with open(Evaluation_Index,'a') as file_obj:
                        file_obj.write('====== {} =====\n'.format(str(i)))
                        file_obj.write('rmse = {}\n'.format(np.sqrt(((vis_x-vis_gx)**2).mean())))
                        file_obj.write('ssim = {}\n'.format(score))
                        file_obj.write('csi ={}\n'.format(csi))
                        file_obj.write('far ={}\n'.format(far))
                        file_obj.write('pod ={}\n'.format(pod))
                        file_obj.write('bias ={}\n'.format(bias))
                        file_obj.write('sr ={}\n\n'.format(sr))
                    
                    
    test_loss /= num_of_batch_size
    print('\nTest set: Average loss: {:.4f}'.format(s_loss_sum/6))
    # csi_picture(img_out = Multi_output,test_ims= target,save_path = save_test_path+'csi_{}/'.format(data_name),data_name=data_name)

def draw_CSI(csi, data_name, save_path):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
        ax.set_facecolor((229.0/255.0, 229.0/255.0, 229.0/255.0))
        plt.xlim(0, 60)
        plt.ylim(-0.05, 1.0)
        plt.xlabel('Threshold')
        plt.ylabel('CSI')
        plt.title('{}\nThresholds CSI'.format(data_name))
        plt.grid(True)
        for period in range(configs.input_length):
            plt.plot(np.arange(csi.shape[1]), [np.nan] + csi[period, 1:].tolist(), linewidth=2.0, label='{} min'.format((period+1)*10))

        plt.legend(loc='upper right')
        fig.savefig(fname=save_path+'Thresholds_CSI_ALL.png', format='png')
        plt.clf()

        csi_CREF = np.loadtxt(open('F:/PJ/code/predrnn/Result/CREF/20220323_T6tT6/20220323000to6_512x512mse/20220323000to6csi.csv',"rb"), delimiter=",", skiprows=0)
        Integrated = np.loadtxt(open('D:/PJ/HPRNN/save_model/2014_2021_high_SSIM/test_itr_23_csi_202203230100to12/csi_202203230100to12/202203230100to12_01to06.csv',"rb"), delimiter=",", skiprows=0)
        Integrated_ms_ssim = np.loadtxt(open('D:/PJ/HPRNN/save_model/ms-ssim/2014_2021_high/test_itr_22_csi_202203230000to12/csi_202203230000to12/202203230000to12_01to06.csv',"rb"), delimiter=",", skiprows=0)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
        ax.set_facecolor((229.0/255.0, 229.0/255.0, 229.0/255.0))
        plt.xlim(0, 60)
        plt.ylim(-0.05, 1.0)
        plt.xlabel('Threshold')
        plt.ylabel('CSI')
        plt.title('{}\nThresholds CSI'.format(data_name))
        plt.grid(True)

        plt.plot(np.arange(csi_CREF.shape[1]), [np.nan] + np.mean(csi_CREF[:, 1:], 0).tolist(), '-', label='OP',color='red')
        plt.plot(np.arange(Integrated.shape[1]), [np.nan] + np.mean(Integrated[:, 1:], 0).tolist(), '-', label='HPRNN_SSIM',color='black') #SSIM_B #2014_2021_high_2020_2021_little
        plt.plot(np.arange(Integrated_ms_ssim.shape[1]), [np.nan] + np.mean(Integrated_ms_ssim[:, 1:], 0).tolist(), '-', label='HPRNN_MS-SSIM',color='blue') #MS-SSIM_B
        plt.plot(np.arange(csi.shape[1]), [np.nan] + np.mean(csi[:, 1:], 0).tolist(), '-', label='HPRNN_RMSE_l1',color='green')  
        plt.legend(loc='upper right')

        fig.savefig(fname=save_path+'Thresholds_AVG_CSI.png', format='png')
        plt.clf()
 
def csi_picture(img_out, test_ims, save_path,data_name='csi'):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)       
        csi = []
        for period in range(configs.output_length):
            print("period=",period)
            csi_eva = Verification(pred=img_out[:, period].reshape(-1, 1), target=test_ims[:, period].reshape(-1, 1), threshold=60, datetime='')
            csi.append(np.nanmean(csi_eva.csi, axis=1))
        
        csi = np.array(csi)
        np.savetxt(save_path+'{}.csv'.format(data_name), csi, delimiter = ',')
        np.savetxt(save_path+'{}_01to06.csv'.format(data_name), csi[:6,], delimiter = ',')
        np.savetxt(save_path+'{}_07to12.csv'.format(data_name), csi[6:,], delimiter = ',')
        draw_CSI(csi[:6,], data_name, save_path)

def get_xy_hiroi(place=None):
    
    lat = area_20[place].lat
    lon = area_20[place].lon
    x = int(np.ceil((lon - 115.0)/0.0125))
    y = int(881 - np.ceil((29.0125 - lat)/0.0125))

    return x, y

def main():
    torch.manual_seed(configs.seed)
        
    radar = load_data_CREF(radar_echo_storage_path_OBS=radar_echo_storage_path_OBS, 
                        radar_echo_storage_path_OP=radar_echo_storage_path_OP,       
                        load_radar_echo_df_path=load_radar_echo_df_path,
                        input_shape=[640, 640],
                        output_shape=[512, 512],
                        period=configs.input_length,
                        predict_period=configs.output_length,
                        places=["Sun_Moon_Lake"],
                        random=False,
                        date_range= test_date,
                        test_date= test_date)
    test_loader = radar.generator('test', batch_size=1, save_path=save_path)
    
    # OBS
    model_HPRNN_1 = HPRNN(64).to(device)
    model_HPRNN_1.load_state_dict(torch.load(obs_model_path))
    obs_hprnn = model_HPRNN_1.to(device)

    # Multi
    # model = MultiTaskModel(obs_hprnn)
    model = MultiTaskModel(obs_hprnn).to(device)
    test(model, test_loader)

if __name__ == '__main__':
    main()