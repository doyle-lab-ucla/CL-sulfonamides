import random
import torch
import numpy as np
from tqdm import tqdm
from typing import Optional

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from .data_processing import scramble_array

class ModelPredict():
    def __init__(self, 
                 df_train,
                 df_val, 
                 MC, 
                 batch_size: int=100, 
                 rand_state: int=42,
                ):

        self.df_train = df_train
        self.df_val = df_val
        self.rand_state = rand_state
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.MC = MC

    def _initialize_metrics(self):
        self.r2_scores = []
        self.mses = []
        self.maes = []

    def _tensor_data_prep(self, splitted_data):
        input_train, labels_train, input_val, labels_val = splitted_data
        
        dataset = TensorDataset(torch.tensor(input_train, dtype=torch.float32), 
                                torch.tensor(labels_train, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        input_val = torch.tensor(input_val, dtype=torch.float32).to(self.device)
        labels_val = torch.tensor(labels_val, dtype=torch.float32).to(self.device)
        
        return dataloader, input_val, labels_val

    def train_model(self, target_cols, output_size):      
        df_train = self.df_train.copy()
        df_val = self.df_val.copy()

        target_col = target_cols[:output_size]
        inpt_cols = [col for col in list(df_train.columns) if col not in target_cols]
        
        input_train = df_train[inpt_cols].values
        labels_train = df_train[target_col].values
        input_val = df_val[inpt_cols].values
        labels_val = df_val[target_col].values

        splitted_data = [input_train, labels_train, input_val, labels_val]
        dataloader, input_val, labels_val = self._tensor_data_prep(splitted_data)
        
        self.model = self.MC.train_model(dataloader, input_val, labels_val)
        self.model.eval()
    
    def predict(self, input_val):
        self._initialize_metrics()
        val_pred = self.model(input_val)
        val_pred = val_pred.cpu()
        val_pred = val_pred.detach().numpy()
        labels_val = labels_val.cpu()

        self.labels_val = labels_val.detach().numpy()[:,0]
        self.val_pred = val_pred[:,0]

        self.r2_scores.append(r2_score(labels_val[:,0], val_pred[:,0]))
        self.mses.append(mean_squared_error(labels_val[:,0], val_pred[:,0], squared = False))
        self.maes.append(mean_absolute_error(labels_val[:,0], val_pred[:,0]))

    def print_performance(self):
        print(f"R2 Score: {self.r2_scores:.5f}")
        print(f"MAE: {self.maes:.3%}") 
        print(f"RMSE: {self.mses:.3%}")
