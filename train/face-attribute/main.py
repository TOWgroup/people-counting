
#   python main.py --filepath Dataset\\list_attr_celeba.csv --data_process --pred result.png

from logging import root
import os
import sys
import argparse
from glob import glob
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn

from model import MultiLableModel
from solver import fit, validate
from celebA import load_data, CelebA, CelebA_Test
from visualize import plot_result, plot_pred


parser = argparse.ArgumentParser(description='Multi-Label Classification!')
if __name__ == "__main__":
    parser.add_argument('--filepath', help='Data path to load data', required=False)
    parser.add_argument('--data_process', help='Split-DataLoader', required=False, action='store_true')
    parser.add_argument('--train', help='Path to save model', type = str, required=False)
    parser.add_argument('--pred', help='Path to saved model', type = str, required=False)

    args = parser.parse_args()

    if args.filepath and args.data_process:
        data_df, label_df = load_data(args)
        #   Split data
        X_train, X_val, Y_train, Y_val = train_test_split(data_df.values, label_df, test_size=0.2, random_state=42)
        X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)
    
    if args.train:
        my_trans = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor()])

        root_dir = 'Dataset\\img_align_celeba'
        train_data = CelebA(root_dir, X_train, Y_train.values, my_trans)
        train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

        val_data = CelebA(root_dir, X_val, Y_val.values, my_trans)
        val_loader = DataLoader(dataset=val_data, batch_size=64, shuffle=True)
        
        #   Define some hyperparameters
        learning_rate = 5e-4
        no_epochs = 7

        model = MultiLableModel()
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.7, mode='max', verbose=True)
        #  Early stopping Pytorch: https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/

        model = MultiLableModel()
        model_his = fit(args, model, train_loader, val_loader, criterion, optimizer, scheduler, no_epochs)

    if args.pred:
        #   Load checkpoint
        load_checkpoint = torch.load('models\\best_model.pth.tar')
        #   Load model
        model = MultiLableModel()
        model.load_state_dict(load_checkpoint['model'])

        #   Load Test Dataloader
        root_dir = 'Dataset\\testset'
        filename = glob(os.path.join(root_dir, '*.png'))
        
        my_trans = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor()])

        test_data = CelebA_Test(filename, my_trans)
        test_loader = DataLoader(dataset=test_data, batch_size=10, shuffle=True)

        # test_his = validate(test_loader, model, criterion)
        plot_pred(args, model, test_loader)
        




