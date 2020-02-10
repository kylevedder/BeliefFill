import torch.utils.data as data
import torch
import numpy as np
import joblib
import glob

class OccupancyDataset(data.Dataset):
    def __init__(self, input_files, label_files):
        
        input_files_lst = sorted(list(glob.glob(input_files)))
        label_files_lst = sorted(list(glob.glob(label_files)))

        assert(len(input_files_lst) == len(label_files_lst))
        assert(len(input_files_lst) > 0)

        input_data_lst = [joblib.load(f) for f in input_files_lst]
        label_data_lst = [joblib.load(f) for f in label_files_lst]

        assert([e.shape for e in input_data_lst] == [e.shape for e in label_data_lst])

        input_data = np.concatenate(input_data_lst, 0)
        label_data = np.concatenate(label_data_lst, 0)

        self.Xs = np.expand_dims(input_data.astype(np.float32), 1)
        self.ys = np.expand_dims(label_data.astype(np.float32), 1)

        print("Loaded {} Xs and {} ys".format(self.Xs.shape[0], self.ys.shape[0]))
        assert(self.Xs.shape == self.ys.shape)

    def __getitem__(self, index):
        Xs_row = self.Xs[index]
        ys_row = self.ys[index]
        return torch.from_numpy(Xs_row), torch.from_numpy(ys_row)

    def __len__(self):
        return self.Xs.shape[0]