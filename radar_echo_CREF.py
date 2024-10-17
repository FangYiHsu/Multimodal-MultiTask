## Other libs
import os
import gzip, struct
import pandas as pd
import numpy as np
import pickle as pkl
from time import time 
from datetime import datetime, timedelta
from enum import Enum
import pickle as pkl

## ML libs
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

## Custom libs
from area_20 import area_20
from scipy import interpolate

"""
    Classes:
     
        fileformat   輸入資料格式列舉
        load_data    載入雷達回波資料，並處理成序列並存於DataFrame中
        generator    資料生成器，用於將資料載入模型
    
    load_data Parameter:

        radar_echo_storage_path    雷達回波原始資料儲存路徑
        input_shape                輸入矩陣大小
        output_shape               輸出矩陣大小
        period                     輸入時間序列長度
        predict_period             輸出時間序列長度
        place                      訓練資料地點
        date_range                 訓練資料時間範圍  ex.[['2017-01-01 00:00', '2017-11-30 23:59'],
                                                      ['2018-05-01 00:00', '2018-11-30 23:59']]
        test_date                  測試日期 ex. [['2018-08-23 00:00', '2018-08-30 23:59']]
        val_split                  驗證集比例
        random                     資料順序是否打亂
        random_seed                隨機種子
        radar_echo_name_format     雷達回波原始資料檔案名稱格式
        radar_echo_file_format     雷達回波原始資料檔案格式
        load_radar_echo_df_path    雷達回波處理後DataFrame儲存路徑
        
"""

class fileformat(Enum):
    GZ = '.gz'
    NPY = '.npy'
    NONE = ''

class load_data_CREF(object):
    def __init__(self, radar_echo_storage_path_OP,
                       radar_echo_storage_path_OBS,
                       input_shape=[105, 105],
                       output_shape=[1, 1],
                       period=6, 
                       predict_period=12,
                       places=['Sun_Moon_Lake'],
                       date_range=[['2017-01-01 00:00', '2017-11-30 23:59']],
                       test_date=[['2018-08-23 00:00', '2018-08-30 23:59']],
                       val_split=0.2,
                       random=True,
                       random_seed=45,
                       radar_echo_name_format=[['CREF_010min.%Y%m%d.%H%M%S','CREF_020min.%Y%m%d.%H%M%S',
                                                'CREF_030min.%Y%m%d.%H%M%S','CREF_040min.%Y%m%d.%H%M%S',
                                                'CREF_050min.%Y%m%d.%H%M%S','CREF_060min.%Y%m%d.%H%M%S',
                                                'CREF_070min.%Y%m%d.%H%M%S','CREF_080min.%Y%m%d.%H%M%S',
                                                'CREF_090min.%Y%m%d.%H%M%S','CREF_100min.%Y%m%d.%H%M%S',
                                                'CREF_110min.%Y%m%d.%H%M%S','CREF_120min.%Y%m%d.%H%M%S'],
                                                'COMPREF.%Y%m%d.%H%M'],
                       radar_echo_file_format=fileformat.GZ,
                       load_radar_echo_df_path=None):
        
        self._radar_echo_storage_path_OP = radar_echo_storage_path_OP
        self._radar_echo_storage_path_OBS = radar_echo_storage_path_OBS
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._period = period
        self._predict_period = predict_period
        self._places = places
        self._date_range = date_range
        self._test_date = test_date
        self._val_split = val_split
        self._random = random
        self._random_seed = random_seed
        self._radar_echo_name_format = radar_echo_name_format
        self._radar_echo_file_format = radar_echo_file_format
        self._load_radar_echo_df_path = load_radar_echo_df_path

        self._date_ranged = self._buildSourceDateRange() # 把字串轉換成 datetime 格式
        self._places_dict = {} # 經緯度轉換後的 xy 值
        self._initPlaceLatLontoXY(places)
        self._createRadarEchoDict()
        self._createDataSetSequence()
        self._testValDatetimeExtract()

    def _buildSourceDateRange(self):
        """Turn string format datetime to datetime format list for source data"""
        if self._test_date == None:
            date_range = self._date_range
        else:
            date_range = np.vstack((self._date_range, self._test_date))
        date_range_temp = []

        for date in date_range:
            date_start = datetime.strptime(date[0], "%Y-%m-%d %H:%M") - timedelta(minutes=10*(self._period+1))
            date_end = datetime.strptime(date[-1], "%Y-%m-%d %H:%M") + timedelta(minutes=10*(self._predict_period-1))
            date_range_temp += pd.date_range(date_start, date_end, freq='10T').tolist()#!
        
        date_range_temp = list(dict.fromkeys(date_range_temp))
        return date_range_temp

    def _buildDateRange(self, date_range):
        """Turn string format datetime to datetime format list"""
        date_range_temp = []
        for date in date_range:
            date_range_temp += pd.date_range(date[0], date[1], freq='10T').tolist()
        return date_range_temp
    
    def _initPlaceLatLontoXY(self,places):
        """Transfor place (longitude, latitude) to matrix (X, Y)"""
        for j in range(len(places)):
            self._place = places[j]
            lat = area_20[self._place].lat
            lon = area_20[self._place].lon
            # OP
            self._op_y = int(1501 - np.ceil((31 - lat)/0.01))
            self._op_x = int(np.ceil((lon - 113.5)/0.01))
            # OBS
            self._obs_y = int(881 - np.ceil((29.0125 - lat)/0.0125))
            self._obs_x = int(np.ceil((lon - 115.0)/0.0125))
            self._places_dict[places[j]]={'op_y':self._op_y,'op_x':self._op_x, 'obs_y':self._obs_y,'obs_x':self._obs_x}
        for  place,xy in self._places_dict.items():
            print("place in {}, op_x is {} op_y is {} obs_x is {} obs_y is {}".format(place,xy['op_x'],xy['op_y'],xy['obs_x'],xy['obs_y']))
    
        return self._places_dict

    def _createRadarEchoDict(self):
        """Create Radar Echo file list dict index by date"""

        # 已有 PKL
        if self._load_radar_echo_df_path:
            print("已經有 pkl")
            self._radar_echo_df = pd.read_pickle(self._load_radar_echo_df_path)
            print("OP_self._radar_echo_df", self._radar_echo_df.shape)
            return 0
        
        if not os.path.isdir(self._radar_echo_storage_path_OP) or not os.path.isdir(self._radar_echo_storage_path_OBS):
            print("Radar Echo Storage Path Error!")
            exit()

        print("\nCreating Radar Echo dataframe...")
        # OP
        OP_list = []
        print("==== OP ====")
        op_format = ['CREF_010min','CREF_020min',
                    'CREF_030min','CREF_040min',
                    'CREF_050min','CREF_060min',
                    'CREF_070min','CREF_080min',
                    'CREF_090min','CREF_100min',
                    'CREF_110min','CREF_120min']

        for root, dirs, files in os.walk(self._radar_echo_storage_path_OP):
            for f in files:
                fullpath = os.path.join(root, f).replace("\\", "/")
                minute = fullpath.split('/')[-1] # CREF_010min.20180414.000000.gz
                try:
                    if minute[:11] in op_format:
                        date = datetime.strptime(minute[-18:-5], '%Y%m%d.%H%M')
                        if date in self._date_ranged:
                            OP_list.append([date.strftime("%Y%m%d.%H%M"),fullpath])
                except:
                    pass

        op = pd.DataFrame(OP_list, columns=['datetime', 'path'])
        group = op.groupby('datetime')['path'].agg(list).reset_index()
        group = group[group['path'].apply(len) >= self._predict_period]
        print(group.shape)
       
        # OBS
        print("==== OBS ====")
        OBS_list = []
        for root, dirs, files in os.walk(self._radar_echo_storage_path_OBS):
            for f in files:
                fullpath = os.path.join(root, f).replace("\\", "/")
                minute = fullpath.split('/')[-1]
                try:
                    date = datetime.strptime(minute, self._radar_echo_name_format[1]+self._radar_echo_file_format.value)
                    if date in self._date_ranged:
                        OBS_list.append([date.strftime("%Y%m%d.%H%M"),fullpath])
                except Exception as e:
                    pass
        obs = pd.DataFrame(OBS_list, columns=['datetime', 'path'])
        print(obs.shape)

        print("==== merge ====")
        self._radar_echo_df = pd.merge(group, obs, on='datetime')
        self._radar_echo_df['datetime'] = pd.to_datetime(self._radar_echo_df['datetime'], format='%Y%m%d.%H%M')
        self._radar_echo_df = self._radar_echo_df.drop_duplicates(subset='datetime', keep='first').sort_values('datetime').set_index('datetime')
        self._radar_echo_df.columns = ['OP_path', 'OBS_path']
        print("self._radar_echo_df", self._radar_echo_df.shape)

        print("Radar Echo loading...")
        for place,xy in self._places_dict.items():

            for i in range(self._predict_period):
                radar_echo_OP = []  
                for op_path in self._radar_echo_df['OP_path']:
                    try:
                        radar_echo_OP.append(self._radarEchoUnpack(op_path[i],xy['op_x'],xy['op_y']))
                    except Exception as e:
                        radar_echo_OP.append([]) 
                        print("OP_radar_echo error", e, op_path)

                print(i," OP ",np.array(radar_echo_OP).shape)
                self._radar_echo_df.insert(i+2, "OP_{}".format(i), radar_echo_OP)

            radar_echo_OBS = []
            for obs_path in self._radar_echo_df['OBS_path']:
                try:
                    radar_echo_OBS.append(self._radarEchoUnpack_Obs(obs_path,xy['obs_x'],xy['obs_y']))
                except Exception as e:
                    radar_echo_OBS.append([])  
                    print("OBS_radar_echo error", e)

            print("OBS ",np.array(radar_echo_OBS).shape)

            self._radar_echo_df.insert(14, "OBS_data", radar_echo_OBS)

        # datetime(index), OP_path, OBS_path, OP_0~OP_12, OBS_data
        print("self._radar_echo_df",self._radar_echo_df)
        self._radar_echo_df.to_pickle('{}.pkl'.format("CREF_2020_test"))

        print("ALL Loading finished!")

    def _createDataSetSequence(self):
        """Processing data sequence for each datetime"""
        radar_date = []

        for dt in self._date_ranged:
            checked, idx_list = self._datetimeSequenceCheck(dt)

            if checked:
                radar_date.append([dt, idx_list])
        
        print("Checking finished!")

        # 預測時間點與他的序列
        self._radar_echo_sequence_df = pd.DataFrame({'datetime': [val[0] for val in radar_date],
                                                     'RadarEchoSequence': [val[1] for val in radar_date]}).set_index('datetime').sort_index()
        # print("self._radar_echo_sequence_df",self._radar_echo_sequence_df.info())
    def _datetimeSequenceCheck(self, dt):
        """Check data sequnce completeness"""
        date_start = dt - timedelta(minutes=10*(self._period+1))
        date_end = dt + timedelta(minutes=10*(self._predict_period-1))
        daterange = pd.date_range(date_start, end=date_end, freq='10T')
        
        # 檢查整個時段是否都有讀取到gz檔案
        for date in daterange:
            if not date in self._radar_echo_df.index:
                return False, []

        return True, daterange.tolist()
 
    def _testValDatetimeExtract(self):
        "Extract test datetime and val datetime from Radar Echo Sequence dataframe"
        if self._test_date != None:
            test_date_range = self._buildDateRange(self._test_date)
            test_date_range_temp_seq = []
            test_date_range_temp = []
            for date in test_date_range:
                if date in self._radar_echo_sequence_df.index:
                    test_date_range_temp.append(date)
                    test_date_range_temp_seq.append(date)

            self._test_radar_echo_seq_df = pd.DataFrame({'datetime': test_date_range_temp,
                                                        'RadarEchoSequence': self._radar_echo_sequence_df['RadarEchoSequence'][test_date_range_temp_seq]})
            # print("self._test_radar_echo_seq_df",self._test_radar_echo_seq_df)
        else:
            test_date_range_temp = []
            

        train_date_range = self._buildDateRange(self._date_range)
        train_date_range_temp_seq = []
        train_date_range_temp = []
        for date in train_date_range:
            if date in self._radar_echo_sequence_df.index and date not in test_date_range_temp:
                train_date_range_temp.append(date)
                train_date_range_temp_seq.append(date)

        self._train_val_radar_echo_seq_df = pd.DataFrame({'datetime': train_date_range_temp,
                                                     'RadarEchoSequence': self._radar_echo_sequence_df['RadarEchoSequence'][train_date_range_temp_seq]})

        if self._val_split == 0:
            self._train_radar_echo_seq_df = shuffle(self._train_val_radar_echo_seq_df)
        else:
            self._train_val_radar_echo_seq_df = shuffle(self._train_val_radar_echo_seq_df)
            self._train_radar_echo_seq_df = self._train_val_radar_echo_seq_df.iloc[:int(-(self._val_split)*len(self._train_val_radar_echo_seq_df))].set_index('datetime')#!
            self._val_radar_echo_seq_df = self._train_val_radar_echo_seq_df.iloc[int(-(self._val_split)*len(self._train_val_radar_echo_seq_df)):].set_index('datetime')


    def _radarEchoUnpack(self, file_path, x, y):
        """Load Radar Echo from file"""
        if self._radar_echo_file_format == fileformat.GZ:
            data = gzip.open(file_path).read()
            radar = struct.unpack(1501*1501*'h', data[-1501*1501*2:])

        elif self._radar_echo_file_format == fileformat.NONE:
            with open(file_path, 'rb') as d:
                data = d.read()
                radar = struct.unpack(1501*1501*'h', data[-1501*1501*2:])

        radar = np.array(radar).reshape(1501, 1501).astype(np.float32)/10
        radar_data = radar[y-int(self._input_shape[1]/2):y+int(self._input_shape[1]/2), 
                           x-int(self._input_shape[0]/2):x+int(self._input_shape[0]/2)]
        radar_data = radar_data.flatten()
        radar_data = [val if val > 0.0 else 0.0 for val in radar_data]
        radar_data = np.array(radar_data).reshape(self._input_shape)

        x0 =np.linspace(0,1,640)
        y0 =np.linspace(0,1,640)
        f0 = interpolate.interp2d(x0, y0, radar_data, kind='linear') #轉換成0.0125度
        x1 =np.linspace(0,1,512)
        y2 =np.linspace(0,1,512)
        radar_change = f0(x1, y2)
        radar_data = np.array(radar_change).reshape(self._output_shape)
        return radar_data

    def _radarEchoUnpack_Obs(self, file_path, x, y):
        """Load Radar Echo from file"""
        radar=[]
        if self._radar_echo_file_format == fileformat.GZ:
            data = gzip.open(file_path).read()
            radar = struct.unpack(881*921*'h', data[-881*921*2:])

        elif self._radar_echo_file_format == fileformat.NONE:
            with open(file_path, 'rb') as d:
                data = d.read()
                radar = struct.unpack(881*921*'h', data[-881*921*2:])
        radar_data=[]
        radar = np.array(radar).reshape(881, 921).astype(np.float32)/10
        radar = radar.astype(np.float32)
        radar_data = radar[y-int(self._output_shape[1]/2):y+int(self._output_shape[1]/2), x-int(self._output_shape[0]/2):x+int(self._output_shape[0]/2)] 
        radar_data = radar_data.astype(np.float32)
        radar_data = radar_data.flatten()
        radar_data = [val if val > 0.0 else 0.0 for val in radar_data]
        radar_data = np.array(radar_data).reshape(self._output_shape)

        return radar_data

    def saveConfig(self, save_path='data/'):
        """Save Config"""
        config_dict = {'radar_echo_storage_path': self._radar_echo_storage_path_OP,
                       'input_shape': self._input_shape,
                       'output_shape': self._output_shape,
                       'period': self._period,
                       'predict_period': self._predict_period,
                       'place': self._place,
                       'date_range': self._date_range,
                       'test_date': self._test_date,
                       'test_split': self._val_split,
                       'random': self._random,
                       'random_seed': self._random_seed,
                       'load_radar_echo_df_path': self._load_radar_echo_df_path,
                       'radar_echo_name_format': self._radar_echo_name_format,
                       'radar_echo_file_format': self._radar_echo_file_format}

        with open(save_path + 'config.pkl', 'wb') as cfg:
            pkl.dump(config_dict, cfg)
    
    def getRadarEchoDataFrame(self):
        """Return Radar Echo dataframe"""
        return self._radar_echo_df

    def getRadarEchoSequenceDataFrame(self):
        """Return Radar Echo sequence dataframe"""
        return self._radar_echo_sequence_df

    def generator(self, type='train', batch_size=32,  save_path='/day'):
        """Create generator"""
        if type == 'train':
            return generator(radar_echo_dataframe=self._radar_echo_df,
                             radar_echo_seq_dataframe=self._train_radar_echo_seq_df,
                             input_shape=self._input_shape,
                             output_shape=self._output_shape,
                             period=self._period,
                             predict_period=self._predict_period,
                             batch_size=batch_size,
                             random=self._random)

        if type == 'val':
            return generator(radar_echo_dataframe=self._radar_echo_df,
                             radar_echo_seq_dataframe=self._val_radar_echo_seq_df,
                             input_shape=self._input_shape,
                             output_shape=self._output_shape,
                             period=self._period,
                             predict_period=self._predict_period,
                             batch_size=batch_size,
                             random=self._random)

        if type == 'test':
            return generator(radar_echo_dataframe=self._radar_echo_df,
                             radar_echo_seq_dataframe=self._test_radar_echo_seq_df,
                             input_shape=self._input_shape,
                             output_shape=self._output_shape,
                             period=self._period,
                             predict_period=self._predict_period,
                             batch_size=batch_size,
                             random=self._random)

        return None


class generator():
    def __init__(self, radar_echo_dataframe,
                       radar_echo_seq_dataframe,
                       input_shape=[105, 105],
                       output_shape=[1, 1],
                       period=6,
                       predict_period=6,
                       batch_size=32,
                       random=True):

        self._input_shape = input_shape
        self._output_shape = output_shape
        self._period = period
        self._predict_period = predict_period
        self._radar_echo_dataframe = radar_echo_dataframe
        self._radar_echo_seq_dataframe = radar_echo_seq_dataframe
        self._data_shape = np.array(radar_echo_dataframe['OBS_data'].values[0].tolist()).shape
        self._batch_size = batch_size
        self._step_per_epoch = self.__len__()
        self._steps = np.arange(self._step_per_epoch)
        self._random = random

    def generator_getClassifiedItems_3(self, idx, place):
        """Return batch X, y sequence data"""
   
        batch_X = []
        batch_y = []

        for batch in range(idx*self._batch_size, (idx+1)*self._batch_size):
            try:
                OBS_radar_echo = np.array(self._radar_echo_dataframe['OBS_data'][self._radar_echo_seq_dataframe['RadarEchoSequence'][batch]].tolist())
                batch_X.append(self._getCentral(OBS_radar_echo[:self._period], self._output_shape).reshape([self._period]+[1]+self._output_shape))
                batch_y.append(self._getCentral(OBS_radar_echo[-self._predict_period:], self._output_shape).reshape([self._predict_period]+[1]+self._output_shape))

                # radar_echo = np.array(self._radar_echo_dataframe['OBS_data'][self._radar_echo_seq_dataframe['RadarEchoSequence'][batch]].tolist())
                # batch_X.append(self._getCentral(radar_echo[:self._period], self._input_shape).reshape([self._period]+[1]+self._input_shape))  
                # batch_y.append(self._getCentral(radar_echo[-self._predict_period:], self._output_shape).reshape([self._predict_period]+[1]+self._output_shape))
                
            except Exception as e:   
                print("===========",e)
                return np.array(batch_X), np.array(batch_y)  
        return np.array(batch_X), np.array(batch_y)
    
    def generator_getClassifiedItems_OP(self, idx):
        """Return batch X, y sequence data"""

        batch_x_OP = []
        batch_y_OBS = []

        t_1 = 6
        # OP
        for batch in range(idx*self._batch_size, (idx+1)*self._batch_size):
            try:
                OP_radar_echo = []
                op_colums = ['OP_0', 'OP_1','OP_2','OP_3', 'OP_4', 'OP_5',
                            'OP_6', 'OP_7','OP_8','OP_9', 'OP_10', 'OP_11']
                for each in op_colums[:self._period]:
                    OP_radar_echo.append(np.array(self._radar_echo_dataframe[each][self._radar_echo_seq_dataframe['RadarEchoSequence'][batch][t_1]].tolist()))
                batch_x_OP.append(self._getCentral(np.array(OP_radar_echo), self._output_shape).reshape([self._period]+[1]+self._output_shape))
                
                OBS_radar_echo = np.array(self._radar_echo_dataframe['OBS_data'][self._radar_echo_seq_dataframe['RadarEchoSequence'][batch]].tolist())
                batch_y_OBS.append(self._getCentral(OBS_radar_echo[-self._predict_period:], self._output_shape).reshape([self._predict_period]+[1]+self._output_shape))

            except Exception as e:
                print("get_error",e)
                return np.array(batch_x_OP), np.array(batch_y_OBS)
            
        # t-1的未來10min~60min OP, t~t+11的OBS
        return np.array(batch_x_OP), np.array(batch_y_OBS)
    
    def generator_getClassifiedItems_Error(self, idx):
        """Return batch X, y sequence data"""

        batch_y_OP = []
        batch_y_OBS = []
        error_x = []
        error_y = []

        t_7 = 0
        t_1 = 6

        op_colums = ['OP_0', 'OP_1','OP_2','OP_3', 'OP_4', 'OP_5',
                    'OP_6', 'OP_7','OP_8','OP_9', 'OP_10', 'OP_11']
        # OP
        for batch in range(idx*self._batch_size, (idx+1)*self._batch_size):
            try:
                # t-7 10~60min OP
                OP_radar_echo = []
                for each in op_colums[:self._period]:
                    OP_radar_echo.append(np.array(self._radar_echo_dataframe[each][self._radar_echo_seq_dataframe['RadarEchoSequence'][batch][t_7]].tolist()))
                batch_x_OP = self._getCentral(np.array(OP_radar_echo), self._output_shape).reshape([self._period]+[1]+self._output_shape)
                
                # t-6~t-1 OBS
                OBS_radar_echo = np.array(self._radar_echo_dataframe['OBS_data'][self._radar_echo_seq_dataframe['RadarEchoSequence'][batch]].tolist())
                batch_x_OBS = self._getCentral(OBS_radar_echo[1:self._period+1], self._output_shape).reshape([self._period]+[1]+self._output_shape)

                error_x.append(batch_x_OP - batch_x_OBS)

                # t-1 10~120min OP
                OP_radar_echo = []
                for each in op_colums:
                    OP_radar_echo.append(np.array(self._radar_echo_dataframe[each][self._radar_echo_seq_dataframe['RadarEchoSequence'][batch][t_1]].tolist()))
                batch_y_OP.append(self._getCentral(np.array(OP_radar_echo), self._output_shape).reshape([self._predict_period]+[1]+self._output_shape))
                
                # t~t+11 OBS
                OBS_radar_echo = np.array(self._radar_echo_dataframe['OBS_data'][self._radar_echo_seq_dataframe['RadarEchoSequence'][batch]].tolist())
                batch_y_OBS.append(self._getCentral(OBS_radar_echo[-self._predict_period:], self._output_shape).reshape([self._predict_period]+[1]+self._output_shape))

                error_y.append(batch_y_OP[-1] - batch_y_OBS[-1])

            except Exception as e:
                print("get_error",e)
                return np.array(error_x), np.array(error_y), np.array(batch_y_OP), np.array(batch_y_OBS)
        # 誤差、Y_OBS
        return np.array(error_x), np.array(error_y), np.array(batch_y_OP), np.array(batch_y_OBS)

    
    def __getitem1__(self, index):
        """Return batch X, y sequence data"""
        idx = self._steps[index]        
        batch_X = []
        batch_y = []           
        for batch in range(idx*self._batch_size, (idx+1)*self._batch_size):
            try:
                radar_echo = np.array(self._radar_echo_dataframe['Sun_Moon_Lake'][self._radar_echo_seq_dataframe['RadarEchoSequence'][batch]].tolist())
                batch_X.append(self._getCentral(radar_echo[:self._period], self._output_shape).reshape([self._period]+self._output_shape+[1]))
                batch_y.append(self._getCentral(radar_echo[-1:], self._output_shape).reshape([1*self._output_shape[0]*self._output_shape[1]]))
            except:
                return np.array(batch_X), np.array(batch_y)
        return np.array(batch_X), np.array(batch_y)
    
    def __len__(self):
        """Return batch num"""
        __len = int(np.ceil(len(self._radar_echo_seq_dataframe)/self._batch_size))
        print(len)
        if __len == 1:
            return __len
        
        return __len-1
    
    @property
    def step_per_epoch(self):
        """Retuen step per epoch"""
        return self._step_per_epoch

    def on_epoch_end(self):
        """If random index at epoch end"""
        if self._random:
            self._steps = shuffle(self._steps)

    def _getCentral(self, data, shape):
        """get the data Central"""
        if self._data_shape[0] < shape[0] or self._data_shape[1] < shape[1]:
            print("Data shape is not big enough. \nPlease reprocess radar echo datafram using load_data class.")
            exit()
        
        if self._data_shape[0] == shape[0] and self._data_shape[0] == shape[0]:
            return data
            
        return data[:, self._data_shape[0]//2-shape[0]//2:self._data_shape[0]//2+shape[0]//2+1, self._data_shape[1]//2-shape[1]//2:self._data_shape[1]//2+shape[1]//2+1]


def loadConfig(path):
    """Load config from path"""
    with open(path, 'rb') as cfg:
        config_dict =pkl.load(cfg)
    return config_dict

if __name__ == "__main__":
    
    from time import time
    start = time()
    places=['Sun_Moon_Lake']
    date_date = [['2020-01-01 00:00','2020-01-31 23:59']]
    test_date=[['2019-12-30 00:00','2019-12-30 00:09']]

    model_parameter = {"input_shape": [512, 512],
                   "output_shape": [512, 512],
                   "period": 6,
                   "predict_period": 18,
                   "filter": 36,
                   "kernel_size": [1, 1]}

    data = load_data_CREF(radar_echo_storage_path='', 
                        load_radar_echo_df_path=None,
                        input_shape=model_parameter['input_shape'],
                        output_shape=model_parameter['output_shape'],
                        period=model_parameter['period'],
                        predict_period=model_parameter['predict_period'],
                        places=places,
                        date_range=test_date,
                        test_date=test_date,
                        radar_echo_name_format=['CREF_010min.%Y%m%d.%H%M%S', 'CREF_020min.%Y%m%d.%H%M%S', 
                                                'CREF_030min.%Y%m%d.%H%M%S', 'CREF_040min.%Y%m%d.%H%M%S', 
                                                'CREF_050min.%Y%m%d.%H%M%S', 'CREF_060min.%Y%m%d.%H%M%S'])

    end = time()
    data.exportRadarEchoFileList()
    data.saveRadarEchoDataFrame()

    df = data.getRadarEchoDataFrame()
    radarseq = data.getRadarEchoSequenceDataFrame()

    train_generator = data.generator('train')
    val_generator = data.generator('val')
    test_generator = data.generator('test')

    for idx in range(train_generator.step_per_epoch):
        batch_X, batch_y = train_generator.__getitem__(idx)
        print(np.array(batch_X).shape)
        print(np.array(batch_y).shape)
