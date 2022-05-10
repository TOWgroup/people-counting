
import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

def load_data(args):
    df = pd.read_csv(args.filepath)

    label_dict = {}
    for i in range(len(df)):
        label_dict[df.iloc[i, 0]] = df.iloc[i, 1:]
    label_df = pd.DataFrame(label_dict).T

    label_df.replace([-1], [0], inplace = True)
    data_df = df.iloc[:, 0]
    return data_df, label_df


class CelebA(Dataset):
    def __init__(self, root_dir, file_name, labels, transform = False):
        self.root_dir = root_dir
        self.file_name = file_name
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir, self.file_name[index])
        img = Image.open(img_name)
        label = torch.tensor(self.labels[index])
        if self.transform:
            img = self.transform(img)
        return img, label



class CelebA_Test(Dataset):
    def __init__(self, pathname, transform = False):
        self.file_name = pathname
        self.transform = transform

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, index):
        img_name = self.file_name[index]
        img = Image.open(img_name)
        if self.transform:
            img = self.transform(img)
        return img

