import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional, Literal

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from .data_processing import scramble_array

class LeaveNoutNx():
    def __init__(self, 
                 df, 
                 leave_out_var,
                 num_elements,
                 MC, 
                 batch_size: int=100, 
                 rand_state: int=42, 
                 n: int=1,
                 ghost_loo_var: Optional[str]=None):
        self.n = n
        self.df = df
        self.leave_out_var = leave_out_var
        self.num_elements = num_elements
        self.rand_state = rand_state
        self.test_list = self._gen_leave_out_list()
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.MC = MC
        self.r2_scores = []
        self.mses = []
        self.maes = []
        self.labels_dict = {} # For predicted labels on ext set
        self.ghost_loo_var = ghost_loo_var
        # self.leave_out_cols = [self.leave_out_var]
        self.predictions_dict = {}
        self.predictions_df = pd.DataFrame(columns=['Index', 'True', 'Predicted', 'Residual', 'SquaredResidual'])

        
    def _gen_leave_out_list(self):
        lo_series = self.df[self.leave_out_var].copy()
        lo_values = list(lo_series.value_counts().keys())

        num_repeats = 5
        all_lo_tests = []

        for i in range(num_repeats):
            current_rand_state = self.rand_state + i
            random.seed(current_rand_state)
            selected_values = random.sample(lo_values, self.num_elements)
            all_lo_tests.append(selected_values)

        return all_lo_tests
    
    def _tensor_data_prep(self, splitted_data):
        input_train, labels_train, input_val, labels_val = splitted_data
        
        dataset = TensorDataset(torch.tensor(input_train, dtype=torch.float32), 
                                torch.tensor(labels_train, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        input_val = torch.tensor(input_val, dtype=torch.float32).to(self.device)
        labels_val = torch.tensor(labels_val, dtype=torch.float32).to(self.device)
        
        return dataloader, input_val, labels_val
  
    def execute_loop(self, target_cols, output_size):
        for oos_test in tqdm(self.test_list):
            if oos_test!=['Train']:
                df_train = self.df[~self.df[self.leave_out_var].isin(oos_test)].copy()
                df_val = self.df[self.df[self.leave_out_var].isin(oos_test)].copy()
                df_train.drop(columns=[self.leave_out_var], inplace = True)
                df_val.drop(columns=[self.leave_out_var], inplace = True)
                target_col = target_cols[:output_size]
                inpt_cols = [col for col in list(df_train.columns) if col not in target_cols]
                
                input_train = df_train[inpt_cols].values
                labels_train = df_train[target_col].values
                input_val = df_val[inpt_cols].values
                labels_val = df_val[target_col].values

                scaler = StandardScaler()
                scaler.fit(input_train)
                input_train = scaler.transform(input_train)
                input_val = scaler.transform(input_val)

                splitted_data = [input_train, labels_train, input_val, labels_val]
                dataloader, input_val, labels_val = self._tensor_data_prep(splitted_data)
                
                model = self.MC.train_model(dataloader, input_val, labels_val)
                model.eval()
                val_pred = model(input_val)
                val_pred = val_pred.cpu()
                val_pred = val_pred.detach().numpy()
                labels_val = labels_val.cpu()

                self.labels_val = labels_val
                self.labels_dict[oos_test[0]] = {'Predicted': val_pred[:,0].tolist(),
                                                 'True': labels_val[:,0].tolist()}

                # Calculate residuals and squared residuals for each data point
                residuals = labels_val[:, 0] - val_pred[:, 0]
                squared_residuals = residuals ** 2

                # Append predictions, true values, residuals, and squared residuals with indices to the DataFrame
                fold_df = pd.DataFrame({
                            'Index': df_val.index,
                            'True': labels_val[:, 0],
                            'Predicted': val_pred[:, 0],
                            'Residual': residuals,
                            'SquaredResidual': squared_residuals
                            })
                self.predictions_df = pd.concat([self.predictions_df, fold_df], ignore_index=True)
                self.predictions_dict[oos_test[0]] = self.predictions_df

                self.r2_scores.append(r2_score(labels_val[:,0], val_pred[:,0]))
                self.mses.append(mean_squared_error(labels_val[:,0], val_pred[:,0], squared = False))
                self.maes.append(mean_absolute_error(labels_val[:,0], val_pred[:,0]))

    def print_performance(self):
    
        print(f"R2 Score: {np.mean(self.r2_scores):.5f} ± {np.std(self.r2_scores):.5f}")
        print(f"Cap Avg R2 Score: {np.mean([v if v >= 0 else 0 for v in self.r2_scores]):.5f} ± {np.std([v if v >= 0 else 0 for v in self.r2_scores]):.5f}")
        print(f"MAE: {np.mean(self.maes):.3%} ± {np.std(self.maes):.3%}") 
        print(f"RMSE: {np.mean(self.mses):.3%} ± {np.std(self.mses):.3%}")

import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional, Literal

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from .data_processing import scramble_array

class LeaveNout():
    def __init__(self, 
                 df, 
                 leave_out_var, 
                 MC, 
                 batch_size: int=100, 
                 rand_state: int=42, 
                 n: int=1,
                 ghost_loo_var: Optional[str]=None):
        self.n = n
        self.df = df
        self.leave_out_var = leave_out_var
        self.rand_state = rand_state
        self.test_list = self._gen_leave_out_list()
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.MC = MC
        self.r2_scores = []
        self.mses = []
        self.maes = []
        self.labels_dict = {} # For predicted labels on ext set
        self.ghost_loo_var = ghost_loo_var
        # self.leave_out_cols = [self.leave_out_var]
        self.predictions_dict = {}
        self.predictions_df = pd.DataFrame(columns=['Index', 'True', 'Predicted', 'Residual', 'SquaredResidual'])

        
    def _gen_leave_out_list(self):
        lo_series = self.df[self.leave_out_var].copy()
        lo_values = list(lo_series.value_counts().keys())
        random.Random(self.rand_state).shuffle(lo_values)
        lo_tests = []
        for i in range(int(len(lo_values)/self.n)):
            lo_tests.append(lo_values[i*self.n:(i+1)*self.n])
        return lo_tests
    
    def _tensor_data_prep(self, splitted_data):
        input_train, labels_train, input_val, labels_val = splitted_data
        
        dataset = TensorDataset(torch.tensor(input_train, dtype=torch.float32), 
                                torch.tensor(labels_train, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        input_val = torch.tensor(input_val, dtype=torch.float32).to(self.device)
        labels_val = torch.tensor(labels_val, dtype=torch.float32).to(self.device)
        
        return dataloader, input_val, labels_val
  
    def execute_loop(self, target_cols, output_size):      
        for oos_test in tqdm(self.test_list):
            if oos_test!=['Train']:
                df_train = self.df[~self.df[self.leave_out_var].isin(oos_test)].copy()
                df_val = self.df[self.df[self.leave_out_var].isin(oos_test)].copy()
                df_train.drop(columns=[self.leave_out_var], inplace = True)
                df_val.drop(columns=[self.leave_out_var], inplace = True)
                target_col = target_cols[:output_size]
                inpt_cols = [col for col in list(df_train.columns) if col not in target_cols]
                
                input_train = df_train[inpt_cols].values
                labels_train = df_train[target_col].values
                input_val = df_val[inpt_cols].values
                labels_val = df_val[target_col].values

                scaler = StandardScaler()
                scaler.fit(input_train)
                input_train = scaler.transform(input_train)
                input_val = scaler.transform(input_val)

                splitted_data = [input_train, labels_train, input_val, labels_val]
                dataloader, input_val, labels_val = self._tensor_data_prep(splitted_data)
                
                model = self.MC.train_model(dataloader, input_val, labels_val)
                model.eval()
                val_pred = model(input_val)
                val_pred = val_pred.cpu()
                val_pred = val_pred.detach().numpy()
                labels_val = labels_val.cpu()

                self.labels_val = labels_val
                self.labels_dict[oos_test[0]] = {'Predicted': val_pred[:,0].tolist(),
                                                 'True': labels_val[:,0].tolist()}

                # Calculate residuals and squared residuals for each data point
                residuals = labels_val[:, 0] - val_pred[:, 0]
                squared_residuals = residuals ** 2

                # Append predictions, true values, residuals, and squared residuals with indices to the DataFrame
                fold_df = pd.DataFrame({
                            'Index': df_val.index,
                            'True': labels_val[:, 0],
                            'Predicted': val_pred[:, 0],
                            'Residual': residuals,
                            'SquaredResidual': squared_residuals
                            })
                self.predictions_df = pd.concat([self.predictions_df, fold_df], ignore_index=True)
                self.predictions_dict[oos_test[0]] = self.predictions_df

                self.r2_scores.append(r2_score(labels_val[:,0], val_pred[:,0]))
                self.mses.append(mean_squared_error(labels_val[:,0], val_pred[:,0], squared = False))
                self.maes.append(mean_absolute_error(labels_val[:,0], val_pred[:,0]))

    def print_performance(self):
    
        print(f"R2 Score: {np.mean(self.r2_scores):.5f} ± {np.std(self.r2_scores):.5f}")
        print(f"Cap Avg R2 Score: {np.mean([v if v >= 0 else 0 for v in self.r2_scores]):.5f} ± {np.std([v if v >= 0 else 0 for v in self.r2_scores]):.5f}")
        print(f"MAE: {np.mean(self.maes):.3%} ± {np.std(self.maes):.3%}") 
        print(f"RMSE: {np.mean(self.mses):.3%} ± {np.std(self.mses):.3%}")

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class CrossVal():
    def __init__(self, 
                 df, 
                 MC, 
                 batch_size: int=100, 
                 rand_state: int=42, 
                 n_cv: int=1, 
                 test_size: float=0.2,
                 scramble_X: bool=False,
                 scramble_y: bool=False):
        self.df = df
        self.rand_state = rand_state
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.MC = MC
        self.n_cv = n_cv
        self.scramble_X = scramble_X
        self.scramble_y = scramble_y
        self.test_size = test_size
        self.r2_scores = []
        self.mses = []
        self.maes = []
        self.predictions_df = pd.DataFrame(columns=['Index', 'Fold', 'True', 'Predicted', 'Residual', 'SquaredResidual'])

    def _tensor_data_prep(self, splitted_data):
        input_train, labels_train, input_val, labels_val = splitted_data
        
        dataset = TensorDataset(torch.tensor(input_train, dtype=torch.float32), 
                                torch.tensor(labels_train, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        input_val = torch.tensor(input_val, dtype=torch.float32).to(self.device)
        labels_val = torch.tensor(labels_val, dtype=torch.float32).to(self.device)
        
        return dataloader, input_val, labels_val
    
    def execute_loop(self, target_cols, output_size):
        for cv in tqdm(range(self.n_cv)):
            print(f"Cross-validation fold: {cv+1}/{self.n_cv}")
            df_train, df_val = train_test_split(self.df, test_size=self.test_size, random_state=self.rand_state + cv)
            target_col = target_cols[:output_size]
            inpt_cols = [col for col in list(df_train.columns) if col not in target_cols]
            
            input_train = df_train[inpt_cols].values
            if self.scramble_X:
                input_train = scramble_array(input_train, random_seed=self.rand_state + cv)

            labels_train = df_train[target_col].values
            if self.scramble_y:
                labels_train = scramble_array(labels_train, random_seed=self.rand_state + cv)

            input_val = df_val[inpt_cols].values
            labels_val = df_val[target_col].values

            scaler = StandardScaler()
            scaler.fit(input_train)
            input_train = scaler.transform(input_train)
            input_val = scaler.transform(input_val)

            splitted_data = [input_train, labels_train, input_val, labels_val]
            dataloader, input_val, labels_val = self._tensor_data_prep(splitted_data)
            
            model = self.MC.train_model(dataloader, input_val, labels_val)
            model.eval()
            val_pred = model(input_val)
            val_pred = val_pred.cpu()
            val_pred = val_pred.detach().numpy()
            labels_val = labels_val.cpu()

            r2 = r2_score(labels_val[:, 0], val_pred[:, 0])
            mse = mean_squared_error(labels_val[:, 0], val_pred[:, 0], squared=False)
            mae = mean_absolute_error(labels_val[:, 0], val_pred[:, 0])

            print(f"Fold {cv+1} R2 Score: {r2:.5f}, MSE: {mse:.5f}, MAE: {mae:.5f}")

            # Calculate residuals and squared residuals for each data point
            residuals = labels_val[:, 0] - val_pred[:, 0]
            squared_residuals = residuals ** 2

            # Append predictions, true values, residuals, and squared residuals with indices to the DataFrame
            fold_df = pd.DataFrame({
                'Index': df_val.index,  # Add index for the validation data
                'Fold': cv + 1,
                'True': labels_val[:, 0],
                'Predicted': val_pred[:, 0],
                'Residual': residuals,
                'SquaredResidual': squared_residuals
            })
            
            self.predictions_df = pd.concat([self.predictions_df, fold_df], ignore_index=True)

            self.r2_scores.append(r2)
            self.mses.append(mse)
            self.maes.append(mae)

    def print_performance(self):
        print(f"R2 Score: {np.mean(self.r2_scores):.5f} ± {np.std(self.r2_scores):.5f}")
        print(f"Cap Avg R2 Score: {np.mean([v if v >= 0 else 0 for v in self.r2_scores]):.5f} ± {np.std([v if v >= 0 else 0 for v in self.r2_scores]):.5f}")
        print(f"MAE: {np.mean(self.maes):.3%} ± {np.std(self.maes):.3%}") 
        print(f"RMSE: {np.mean(self.mses):.3%} ± {np.std(self.mses):.3%}")




class CrossValTest():
    def __init__(self, 
                 df,
                 ext_df,
                 MC, 
                 batch_size: int=100, 
                 rand_state: int=42, 
                 n_cv: int=1, 
                 test_size: float=0.2,):
        self.df = df
        self.ext = ext_df
        self.rand_state = rand_state
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.MC = MC
        self.n_cv = n_cv
        self.test_size = test_size
        self.r2_scores_val = []
        self.mses_val = []
        self.maes_val = []

        self.r2_scores_ext = []
        self.mses_ext = []
        self.maes_ext = []
        self.labels_dict = {} # For predicted labels on ext set


    def _tensor_data_prep(self, splitted_data: list):
        input_train, labels_train, input_val, labels_val, input_ext, labels_ext = splitted_data
        
        dataset = TensorDataset(torch.tensor(input_train, dtype=torch.float32), 
                                torch.tensor(labels_train, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        input_val = torch.tensor(input_val, dtype=torch.float32).to(self.device)
        labels_val = torch.tensor(labels_val, dtype=torch.float32).to(self.device)

        input_ext = torch.tensor(input_ext, dtype=torch.float32).to(self.device)
        labels_ext = torch.tensor(labels_ext, dtype=torch.float32).to(self.device)
        
        return dataloader, input_val, labels_val, input_ext, labels_ext
    
    def execute_loop(self, target_cols, output_size):
        for cv in tqdm(range(self.n_cv)):
            df_ext = self.ext.copy()
            df_train, df_val = train_test_split(self.df,test_size=self.test_size,random_state=cv)
            target_col = target_cols[:output_size]
            inpt_cols = [col for col in list(df_train.columns) if col not in target_cols]
            
            input_train = df_train[inpt_cols].values
            labels_train = df_train[target_col].values

            input_val = df_val[inpt_cols].values
            labels_val = df_val[target_col].values

            input_ext = df_ext[inpt_cols].values
            labels_ext = df_ext[target_col].values

            splitted_data = [input_train, labels_train, input_val, labels_val, input_ext, labels_ext]
            dataloader, input_val, labels_val, input_ext, labels_ext = self._tensor_data_prep(splitted_data)
            
            ### Train and change mode to predict
            model = self.MC.train_model(dataloader, input_val, labels_val)
            model.eval()

            ### CrossVal metrics
            val_pred = model(input_val)
            val_pred = val_pred.cpu()
            val_pred = val_pred.detach().numpy()
            labels_val = labels_val.cpu()

            self.r2_scores_val.append(r2_score(labels_val[:,0], val_pred[:,0]))
            self.mses_val.append(mean_squared_error(labels_val[:,0], val_pred[:,0], squared = False))
            self.maes_val.append(mean_absolute_error(labels_val[:,0], val_pred[:,0]))

            ### External metrics
            ext_pred = model(input_ext)
            ext_pred = ext_pred.cpu()
            ext_pred = ext_pred.detach().numpy()
            labels_ext = labels_ext.cpu()

            self.r2_scores_ext.append(r2_score(labels_ext[:,0], ext_pred[:,0]))
            self.mses_ext.append(mean_squared_error(labels_ext[:,0], ext_pred[:,0], squared = False))
            self.maes_ext.append(mean_absolute_error(labels_ext[:,0], ext_pred[:,0]))
            self.labels_dict[cv] = ext_pred[:,0].tolist()

    def print_performance(self, pred_set: Literal['val','ext']):

        if pred_set == 'val':
            r2_scores = self.r2_scores_val
            maes = self.maes_val
            mses = self.mses_val
            print('Validation')
        elif pred_set == 'ext':
            r2_scores = self.r2_scores_ext
            maes = self.maes_ext
            mses = self.mses_ext
            print('External Validation')
    
        print(f"R2 Score: {np.mean(r2_scores):.5f} ± {np.std(r2_scores):.5f}")
        print(f"Cap Avg R2 Score: {np.mean([v if v >= 0 else 0 for v in r2_scores]):.5f} ± {np.std([v if v >= 0 else 0 for v in r2_scores]):.5f}")
        print(f"MAE: {np.mean(maes):.3%} ± {np.std(maes):.3%}") 
        print(f"RMSE: {np.mean(mses):.3%} ± {np.std(mses):.3%}")
