import numpy as np
from typing import List, Dict, Literal

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler

from .data_visualization import visualize_error, viz_scatter_r2

class ModelClass():
    def __init__(self, 
                 model_tech: Literal['MLP','WeightMLP'], 
                 input_size: int, 
                 hidden_size: int, 
                 output_size: int, 
                 drop_out: float, 
                 step_schedule: int = 80, 
                 num_epochs: int = 100,
                 l1_reg: float = 0,
                 l2_reg: float = 0,
                 rand_state: int = 42, 
                 verbose: bool = False,
                 extra_layer: bool=False,
                 remove_layer: bool=False,
                 weighted_regions: List[Dict] = [{"loss_amplification":1, "target_region":[0,0]}]):
        
        self.model_tech = model_tech  
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.drop_out = drop_out
        self.step_schedule = step_schedule
        self.num_epochs = num_epochs
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.weighted_regions = weighted_regions
        self.extra_layer = extra_layer
        self.remove_layer = remove_layer
        
        
        self.rand_state = rand_state
        self.verbose = verbose
    
    def _init_model(self):
        torch.manual_seed(self.rand_state)
        
        if self.model_tech == "MLP":
            self.model =  MLP(self.input_size, self.hidden_size, self.output_size, self.drop_out, l1_reg=self.l1_reg, l2_reg=self.l2_reg, extra_layer=self.extra_layer, remove_layer=self.remove_layer)
        elif self.model_tech == "WeightMLP":
            self.model =  WeightMLP(self.input_size, self.hidden_size, self.output_size, self.drop_out, l1_reg=self.l1_reg, l2_reg=self.l2_reg, extra_layer=self.extra_layer, weighted_regions=self.weighted_regions)
        else:
            raise ValueError(f"Method {self.model_tech}, is not included.")
       
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_schedule, gamma=0.1)
        # self.loss_fn = nn.MSELoss() #self.model.loss_function(predictions, labels, 0)
        self.loss_fn = self.model.loss_function
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    

        
    def _loss_propagate(self, predictions, labels):
        
        #loss = self.model.loss_function(predictions, labels, 0)
        loss = self.loss_fn(predictions, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss
    
    def _validate_model(self, validation_inputs, validation_labels):

        self.model.eval()
        # validation_predictions = model(validation_inputs) # ERROR
        self.validation_predictions = self.model(validation_inputs)
        validation_loss = self.loss_fn(self.validation_predictions, validation_labels)
        self.model.train()

        return validation_loss.item()

        
    def train_model(self, dataloader, input_val, labels_val):    
        
        self._init_model()

        train_errors_epoch, valid_errors, val_epochs = [], [], []

        for epoch in range(self.num_epochs):
            train_errors = []
            for batch_i, (inputs, labels) in enumerate(dataloader):

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                predictions = self.model(inputs)

                if labels.shape[1] == 1:
                    labels = labels.view(-1,1)

                loss = self._loss_propagate(predictions, labels)
                train_errors.append(loss.item())

            # Call the validation function
            if self.verbose and epoch % 2 == 0:  #or epoch == (num_epochs-1)
                # valid_error = self._validate_model(input_val, labels_val, loss_fn) # ERROR
                valid_error = self._validate_model(input_val, labels_val)
                valid_errors.append(np.mean(valid_error))
                val_epochs.append(epoch)

            train_errors_epoch.append(np.mean(train_errors)) 
            self.scheduler.step()
            
        if self.verbose:
            visualize_error(train_errors_epoch, valid_errors, val_epochs)
            # viz_scatter_r2(labels_val[:,0], val_pred[:,0]) # ERROR
            viz_scatter_r2(labels_val[:,0], self.validation_predictions.detach().numpy()[:,0])

        return self.model
    

class MLP(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 output_size: int, 
                 drop_out_rate: float, 
                 l1_reg: float = 0, 
                 l2_reg: float = 0, 
                 extra_layer: bool = False,
                 remove_layer: bool = False, 
                 verbose: bool = False):
        super(MLP, self).__init__()
        self.extra_layer = extra_layer
        self.remove_layer = remove_layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)          
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.lrelu = nn.LeakyReLU(0.1)
        self.dropout1 = nn.Dropout(p=drop_out_rate)
        self.dropout2 = nn.Dropout(p=drop_out_rate/2)
        if self.extra_layer:
            self.fc25 = nn.Linear(hidden_size, hidden_size)
            self.dropout3 = nn.Dropout(p=drop_out_rate/2)
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        if verbose:
            if self.extra_layer:
                print("Model selected has 4 dense layers.")
            else:
                print("Model selected has 3 dense layers.")
            
    def forward(self, x):
        x = self.fc1(x)
        x = self.lrelu(x)
        x = self.dropout1(x)
        if self.remove_layer == False:
            x = self.fc2(x)
            x = self.lrelu(x)
            x = self.dropout2(x)
        if self.extra_layer:
            x = self.fc25(x)
            x = self.lrelu(x)
            x = self.dropout3(x)
        x = self.fc3(x)
        return x
    
    def l1_loss(self):
        l1_loss = 0
        for param in self.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return self.l1_reg * l1_loss
    
    def l2_loss(self):
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.sum(torch.pow(param, 2))
        return self.l2_reg * l2_loss

    def loss_function(self, output, target):
        mse_loss = nn.MSELoss()(output, target)
        return mse_loss + self.l1_loss() + self.l2_loss()

    ### Old version with noise censored regression when experimental error is large
    # def loss_function(self, output, target, threshold):
    #     mse_loss = nn.MSELoss()(output, target)
    #     if mse_loss < threshold:
    #         return nn.MSELoss()(output, output)
    #     else:
    #         return mse_loss + self.l1_loss() + self.l2_loss()

class WeightMLP(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 output_size: int, 
                 drop_out_rate: float, 
                 l1_reg: int = 0, 
                 l2_reg: int = 0, 
                 extra_layer: bool = False, 
                 weighted_regions: List[Dict] = [{"loss_amplification":1, "target_region":[0,0]}], 
                 verbose: bool = False):
        super(WeightMLP, self).__init__()
        self.extra_layer = extra_layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)          
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.lrelu = nn.LeakyReLU(0.1)
        self.dropout1 = nn.Dropout(p=drop_out_rate)
        self.dropout2 = nn.Dropout(p=drop_out_rate/2)
        if self.extra_layer:
            self.fc25 = nn.Linear(hidden_size, hidden_size)
            self.dropout3 = nn.Dropout(p=drop_out_rate/2)
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.weighted_regions = weighted_regions
        if verbose:
            if self.extra_layer:
                print("Model selected has 4 dense layers.")
            else:
                print("Model selected has 3 dense layers.")
            
    def forward(self, x):
        x = self.fc1(x)
        x = self.lrelu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.lrelu(x)
        x = self.dropout2(x)
        if self.extra_layer:
            x = self.fc25(x)
            x = self.lrelu(x)
            x = self.dropout3(x)
        x = self.fc3(x)
        return x
    
    def l1_loss(self):
        l1_loss = 0
        for param in self.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return self.l1_reg * l1_loss
    
    def l2_loss(self):
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.sum(torch.pow(param, 2))
        return self.l2_reg * l2_loss
    
    def loss_function(self, output, target, threshold=0):
        # Calculate base MSE loss
        mse_loss = nn.MSELoss(reduction='none')(output, target)

        # Create a weight tensor, with double weight for specific target ranges
        weight = torch.ones_like(target)
        for weighted_region in self.weighted_regions:
            weight[(target >= weighted_region["target_region"][0]) & (target <= weighted_region["target_region"][1])] = weighted_region["loss_amplification"]

        # Apply weights
        weighted_mse_loss = mse_loss * weight

        # Average the weighted loss
        average_weighted_mse_loss = torch.mean(weighted_mse_loss)

        # Apply threshold logic
        if average_weighted_mse_loss < threshold:
            return nn.MSELoss()(output, output)
        else:
            return average_weighted_mse_loss + self.l1_loss() + self.l2_loss()      