from pathlib import Path
from tqdm import tqdm
import time
import torchinfo
from cyclopts import App

import h5py
from sklearn.model_selection import train_test_split
import torch
import numpy as np



# Prepare data loader
from torch.utils.data import Dataset, DataLoader
import h5py
from sklearn.metrics import accuracy_score
from torch.optim.optimizer import Optimizer


class radioml_21_dataset(Dataset):
    def __init__(self, dataset_path, matgen_data, frames_per_mod = 4096, snr_lb = -20., snr_ub = 30., fp_data = False):
        super(radioml_21_dataset, self).__init__()
        h5_file = h5py.File(dataset_path,'r')
        if(matgen_data): #If training using the matlab generated data
            snr_key = 'all_SNRs'
            self.mod = h5_file['all_labels'][:,0]
            self.mod_classes = ["BPSK", 
                    "QPSK", 
                    "8PSK",
                    "16QAM",
                    "32QAM", 
                    "64QAM", 
                    "128QAM", 
                    "256QAM",
                    "16APSK", 
                    "32APSK", 
                    "64APSK", 
                    "128APSK",
                    "FM", 
                    "AM-DSB-SC", 
                    "AM-SSB-SC"]
            if(fp_data): #If training using the 32bit floating point data
                data_key = 'all_IQ_float32'              
                # self.data = h5_file['all_IQ_float32']
                print("Extracting Float32 Matlab Generated Data....")
            else:        #By default training using the INT8 data
                data_key = 'all_IQ_int8'
                # self.data = h5_file['all_IQ_int8']
                print("Extracting INT8 Matlab Generated Data....")
            
            # self.data = np.array([np.array(h5_file[obj_ref][:]).T for obj_ref in data]) # Needs to be in shape: (# of Frames, 1024, 2)
            # self.snr = (h5_file['all_SNRs'][0,:]).astype(int) #Needs to be saved like this: array([-20, -20, -20, ...,  30,  30,  30])
            # labels = h5_file['all_labels'][0,:]# Needs to be saved like this: array([ 0,  0,  0, ..., 26, 26, 26])
            # self.len = self.data.shape[0]
            
            
        else:
            data_key = 'X'
            snr_key = 'Z'
            print("Extracting RadioML Generated Data....")
            # self.data = h5_file['X']
            self.mod = np.argmax(h5_file['Y'], axis=1) # comes in one-hot encoding
            self.mod_classes = [
                    "OOK",
                    "4ASK",
                    "8ASK",
                    "BPSK",
                    "QPSK",
                    "8PSK",
                    "16PSK",
                    "32PSK",
                    "16APSK",
                    "32APSK",
                    "64APSK",
                    "128APSK",
                    "16QAM",
                    "32QAM",
                    "64QAM",
                    "128QAM",
                    "256QAM",
                    "AM-SSB-WC",
                    "AM-SSB-SC",
                    "AM-DSB-WC",
                    "AM-DSB-SC",
                    "FM",
                    "GMSK",
                    "OQPSK",
                    "BFSK",
                    "4FSK",
                    "8FSK",
                ]
        
        self.data = h5_file[data_key]
        self.snr = h5_file[snr_key][:,0]
        self.len = self.data.shape[0]
        self.num_classes=len(self.mod_classes)
        self.snr_classes = np.arange(snr_lb, snr_ub+2, 2) # -20dB to 30dB, with step of 2 --> 26 snrs
        self.num_snr = len(self.snr_classes)
        np.random.seed(2021)

        train_indices = []
        test_indices = []
        val_indices = []

        for mod in range(0, len(self.mod_classes)): # all modulations (0 to 26)
            for snr_idx in range(0, len(self.snr_classes)): # all SNRs (0 to 25 = -20dB to +30dB)
                # raw dataset holds frames strictly ordered by modulation and SNR
                # We order the dataset to each mod-snr pair combination for better access of each frame
                # Specifically we divide the dataset into 27 mods group,
                #   and for each group we divide into 26 SNRs,
                # For each modulation-snr pair combination, we have 4096 frames. (27*26*4096 = 2875392)
                # For better analogy, its basically a triple for-loop, with the outer most loop being 27 mods,
                #                                                           then the middle being 26 SNRs,
                #                                                           then inner most being 4096 samples
                # Basically [0[0[0...4095] ...25]...26] with a length of 2875392
                start_idx = self.num_snr*frames_per_mod*mod + frames_per_mod*snr_idx 
                indices_subclass = list(range(start_idx, start_idx+frames_per_mod))
                
                # 90%/10% training/test split, applied evenly for each mod-SNR pair
                # 80 10 10 split 
                split = int(np.ceil(0.8 * frames_per_mod)) 
                split2 = int(np.ceil(0.9 * frames_per_mod)) 

                np.random.shuffle(indices_subclass)
                train_indices_subclass = indices_subclass[:split]
                val_indices_subclass = indices_subclass[split:split2]
                test_indices_subclass = indices_subclass[split2:]
                
                # you could train on a subset of the data, e.g. based on the SNR
                # here we use all available training samples
                if snr_idx >= 0:
                    train_indices.extend(train_indices_subclass)

                test_indices.extend(test_indices_subclass)
                val_indices.extend(val_indices_subclass)
                
        self.train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        self.val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
        self.test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

    def __getitem__(self, idx):
        # transpose frame into Pytorch channels-first format (NCL = -1,2,1024)
        return self.data[idx].transpose(), self.mod[idx], self.snr[idx]

    def __len__(self):
        return self.len

#dataset = radioml_21_dataset(dataset_path) 




class IQDataset(Dataset):
    def __init__(self, data, mods, snrs):
        self.data = data
        self.mods = mods
        self.snrs = snrs


    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]).float(), torch.tensor(self.mods[idx]).float(), torch.tensor(self.snrs[idx]).float()

    def __len__(self):
        return len(self.data)

def get_datasets(dataset_path:Path):
    
    with h5py.File(dataset_path, "r+") as file_handle:
        data = file_handle["X"][:]  # 1024x2 samples
        mods = file_handle["Y"][:]  # mods
        snrs = file_handle["Z"][:]  # snrs
    
    data = data.reshape(data.shape[0], 2, 1024)
    
    # First do a 80-20 split
    # TODO: Temporarily set I tiny train size
    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
        data, mods, snrs,  test_size=0.80, random_state=0, shuffle=True
    )
    
    # Now split the 20 section to 10-10
    (
        X_val,
        X_test,
        Y_val,
        Y_test,
        Z_val,
        Z_test
    ) = train_test_split(X_test, Y_test, Z_test, test_size=0.50, random_state=0, shuffle=True)

    return [
        IQDataset(data=X_train, mods=Y_train, snrs=Z_train),
        IQDataset(data=X_val, mods=Y_val, snrs=Z_val),
        IQDataset(data=X_test, mods=Y_test, snrs=Z_test),
    ]
    


