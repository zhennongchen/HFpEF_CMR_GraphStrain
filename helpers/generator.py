import sys
sys.path.append("/workspace/Documents")
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

class Data_generator(Dataset):
    def __init__(
        self, 
        X_Ecc ,
        X_Err ,
        Y,
        case_index_list,
        shuffle = False,
        EHR = None,
    ):
        super().__init__()
        self.X_Ecc = X_Ecc
        self.X_Err = X_Err
        self.EHR = EHR
        self.Y = Y
        self.shuffle = shuffle
        self.case_index_list = case_index_list
        self.num_cases = len(case_index_list)

        self.index_array = self.generate_index_array()

    def generate_index_array(self):
        np.random.seed()
        index_array = []
        
        if self.shuffle == True:
            f_list = np.random.permutation(self.num_cases)
        else:
            f_list = np.arange(self.num_cases)
        return np.copy(f_list)
        
    def __len__(self):
        return self.num_cases

    def __getitem__(self, index):
        # print('len of index_array:', len(self.index_array))
        # print('in this geiitem, self.index_array is: ', self.index_array, ' we pick index:', index, ' which is:', self.index_array[index])
        f = self.index_array[index]
        # print(' the case index list is :', self.case_index_list, ' therefore the final case index is:', self.case_index_list[f])
        case_index = self.case_index_list[f]
        # load X_Ecc, X_Err
        ecc = self.X_Ecc[case_index,:,:]
        err = self.X_Err[case_index,:,:]
        err_padded = np.vstack([err, np.zeros((4, err.shape[1]))])
        # load EHR
        if self.EHR is not None:
            ehr = self.EHR[case_index,:]
        # load Y
        y = self.Y[case_index]
        
        if self.EHR is not None:
            return torch.from_numpy(ecc).float(), torch.from_numpy(err).float(), torch.from_numpy(err_padded).float(),torch.from_numpy(np.asarray([y])).float(), torch.from_numpy(ehr).float()
        else:
            return torch.from_numpy(ecc).float(), torch.from_numpy(err).float(), torch.from_numpy(err_padded).float(),torch.from_numpy(np.asarray([y])).float(), torch.from_numpy(np.asarray([y])).float()

    def on_epoch_end(self):
        self.index_array = self.generate_index_array()