import csv
import os
import numpy as np
import torch
from torch.utils.data import Dataset

epsilon = np.finfo(float).eps

class myDataset(Dataset):
    def __init__(self, mode, data="./", subjects= [], task = "SpokenEEG"):
        self.sample_rate = 8000
        self.n_classes = 13
        self.mode = mode
        self.iter = iter
        self.savedata = data
        self.subjects = subjects
        self.task = task
        self.max_audio = 32768.0
        self.lenth = 780 #len(os.listdir(self.savedata + '/train/Y/')) #780 # the number data
        self.lenthtest = 260 #len(os.listdir(self.savedata + '/test/Y/')) #260
        self.lenthval = 260 #len(os.listdir(self.savedata + '/val/Y/')) #260
        

    def __len__(self):
        if self.mode == 2:
            return self.lenthval
        elif self.mode == 1:
            return self.lenthtest
        else:
            return self.lenth

    def __getitem__(self, idx):
        '''
        :param idx:
        :return:
        '''
        
        input_, target_cl_, voice_, avg_target_, std_target_, avg_input_, std_input_ = [],[],[],[],[],[],[]
        
        for subi in range(len(self.subjects)):
            sub = self.subjects[subi]
            
            
            if self.mode == 2:
                forder_names = [self.savedata + 'sub%02d/val/'%sub]
            elif self.mode == 1:
                forder_names = [self.savedata + 'sub%02d/test/'%sub]
            else:
                forder_names = [self.savedata + 'sub%02d/train/'%sub]
                
            for forder_name in forder_names:
                # tasks
                allFileList = os.listdir(forder_name + self.task + "/")
                allFileList.sort()
                file_name = forder_name + self.task + '/' + allFileList[idx]
                
                # if self.task.find('vec') != -1: # embedding vector
                #     input, avg_input, std_input = self.read_vector_data(file_name) 
                if self.task.find('mel') != -1:
                    input, avg_input, std_input = self.read_data(file_name)
                elif self.task.find('Voice') != -1: # voice
                    input, avg_input, std_input = self.read_voice_data(file_name)
                else: # EEG
                    input, avg_input, std_input = self.read_data(file_name) 
                    
                input_.append(input)
                avg_input_.append(avg_input)
                std_input_.append(std_input)
                                
                # voice
                allFileList = os.listdir(forder_name + "Voice/")
                allFileList.sort()
                file_name = forder_name + "Voice/"+ allFileList[idx]
                voice, _, _ = self.read_voice_data(file_name)
                voice = np.squeeze(voice)
                voice_.append(voice)
                
                
                # target label
                allFileList = os.listdir(forder_name + "Y/")
                allFileList.sort()
                file_name = forder_name + 'Y/' + allFileList[idx]
                
                target_cl,_,_ = self.read_raw_data(file_name) 
                target_cl = np.squeeze(target_cl)

                target_cl_.append(target_cl)

        # to numpy
        input_ = np.array(input_)
        target_cl_ = np.array(target_cl_)
        voice_ = np.array(voice_)
        avg_target_ = np.array(avg_target_)
        std_target_ = np.array(std_target_)
        avg_input_ = np.array(avg_input_)
        std_input_ = np.array(std_input_)
        
        # to tensor
        input_ = torch.tensor(input_, dtype=torch.float32)
        

        return input_, target_cl_, voice_

   
    def read_vector_data(self, file_name,n_classes):
        with open(file_name, 'r', newline='') as f:
            lines = csv.reader(f)
            data = []
            for line in lines:
                data.append(line)

        data = np.array(data).astype(np.float32)
        (r,c) = data.shape
        data = np.reshape(data,(n_classes,r//n_classes,c))
        
        max_ = np.max(data).astype(np.float32)
        min_ = np.min(data).astype(np.float32)
        avg = (max_ + min_) / 2
        std = (max_ - min_) / 2
        
        data   = np.array((data - avg) / std).astype(np.float32)

        return data, avg, std
    
    
    def read_voice_data(self, file_name):
        with open(file_name, 'r', newline='') as f:
            lines = csv.reader(f)
            data = []
            for line in lines:
                data.append(line)
        data = np.array(data).astype(np.float32)
        
        data = np.array(data / self.max_audio).astype(np.float32)
        avg = np.array([0]).astype(np.float32)

        return data, avg, self.max_audio


    def read_data(self, file_name):
        with open(file_name, 'r', newline='') as f:
            lines = csv.reader(f)
            data = []
            for line in lines:
                data.append(line)

        data = np.array(data).astype(np.float32)
        
        max_ = np.max(data).astype(np.float32)
        min_ = np.min(data).astype(np.float32)
        avg = (max_ + min_) / 2
        std = (max_ - min_) / 2
        
        data   = np.array((data - avg) / std).astype(np.float32)

            
        return data, avg, std


    def read_raw_data(self, file_name):
        with open(file_name, 'r', newline='') as f:
            lines = csv.reader(f)
            data = []
            for line in lines:
                data.append(line)

        data = np.array(data).astype(np.float32)
        avg = np.array([0]).astype(np.float32)
        std = np.array([1]).astype(np.float32)

            
        return data, avg, std


