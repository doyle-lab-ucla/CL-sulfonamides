import os
import pandas as pd
import numpy as np
from rdkit import Chem

def one_hot_encode(df, columns_to_encode):
    """
    Convert specified columns of a pandas DataFrame into one-hot encoded columns.

    Parameters
    ----------
    df : pandas DataFrame
        The DataFrame to one-hot encode.
    columns_to_encode : list of str
        A list of column names to one-hot encode.

    Returns
    -------
    pandas DataFrame
        A new DataFrame with the specified columns one-hot encoded.
    """

    df_encoded = df.copy()
    ohe_columns = []
    for column in columns_to_encode:
        # One-hot encode the column using pandas get_dummies function
        one_hot_encoded = pd.get_dummies(df[column], prefix=column).astype(int)
        # Add the one-hot encoded columns to the new DataFrame
        df_encoded = pd.concat([df_encoded, one_hot_encoded], axis=1)
        ohe_columns.append(one_hot_encoded.columns)
    
    ohe_columns = [col for sublist in ohe_columns for col in sublist]
    df_encoded.drop(columns=columns_to_encode, inplace=True)

    return df_encoded, ohe_columns

def convert_to_canonical_smiles(df_list, col_list, verbose = False):

    for idx, df in enumerate(df_list):
        for col in col_list:
            if col in df.columns:
                count_non_canonicalizable = 0

                def to_canonical(smiles):
                    nonlocal count_non_canonicalizable
                    if pd.isnull(smiles) or smiles == '' or smiles.lower() == 'none':
                        count_non_canonicalizable += 1
                        return None
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        return Chem.MolToSmiles(mol, isomericSmiles=False)
                    else:
                        return None

                df[col] = df[col].apply(to_canonical)

                if count_non_canonicalizable > 0 and verbose:
                    print(f"DataFrame {idx+1}, Column '{col}': {count_non_canonicalizable} values not canonicalized due to being None, empty, or 'none'.")
    
    return df_list

def remove_constant_cols(df):
    
    constant_columns = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]
    df = df.drop(columns=constant_columns)

    return df

def remove_duplicates(df, cols):
    starting_size = len(df)
    df.drop_duplicates(subset = cols, keep='first', inplace = True)
    if starting_size != len(df):
        print("{starting_size-len(df)} duplicated rows found and removed.")
    return df

def expand_list_column(df, col_name):

    col_values = df[col_name].tolist()
    new_col_names = [f"{col_name}_{i}" for i in range(len(col_values[0]))]
    new_df = pd.DataFrame(col_values, columns=new_col_names)
    new_df = pd.concat([df, new_df], axis=1)
    new_df.drop(col_name, axis=1, inplace=True)

    return new_df, new_col_names

def expand_list_column_noise(df, col_name, binary_noise = True):

    col_values = df[col_name].tolist()
    new_col_names = [f"{col_name}_{i}" for i in range(len(col_values[0]))]
    if binary_noise == True:
        new_df = pd.DataFrame(np.random.randint(0,1,size=(len(col_values), len(new_col_names))), columns=new_col_names)
    else:
        new_df = pd.DataFrame(np.random.rand(len(col_values),len(new_col_names)), columns=new_col_names)
    new_df = pd.concat([df, new_df], axis=1)
    new_df.drop(col_name, axis=1, inplace=True)

    return new_df, new_col_names

def standardize_columns(df, columns):
    for col in columns:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

def normalize_columns(df, columns):
    for col in columns:
        df[col] = (df[col] - df[col].min()) / (df[col].max()- df[col].min())
    return df

def load_and_combine_csvs(directory):
    dataframes = []
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    # Load and process each .csv file
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(directory, csv_file))
        prefix = csv_file.split('_')[0]
        df.columns = df.columns[:6].tolist() + [f"{prefix}_{col}" for col in df.columns[6:]]
        dataframes.append(df)

    from functools import reduce
    df_combined = reduce(lambda left,right: pd.merge(left,right,on=list(df.columns[:6])), dataframes)
    
    return df_combined

def scramble_array(input: np.ndarray,
                   random_seed: int=42):
    np.random.seed(random_seed)
    return np.random.permutation(input)