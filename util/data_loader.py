import os
import io
import re
import sys
import time
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

sys.path.insert(0, ROOT_DIR)

def get_id_from_label(label_text, label_to_id_mapping):
    return label_to_id_mapping[label_text]
    

def GetPyTorchDataLoaders(x_train, x_test, y_train, y_test, batch_size = 1000):
    # Pytorch
    X_train  = torch.from_numpy(x_train).float()
    Y_train = torch.from_numpy(y_train)

    X_test = torch.from_numpy(x_test).float()
    Y_test = torch.from_numpy(y_test)

    # Pytorch train and test sets
    train = torch.utils.data.TensorDataset(X_train, Y_train)
    valid = torch.utils.data.TensorDataset(X_test, Y_test)

    # data loader
    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size = batch_size, shuffle = True)

    logger.info('Completed loading data and returning pytorch train and validation data loaders')
    return train_loader, valid_loader


def load_data(file_name):
    IDS_df = pd.read_csv(file_name)
    # IDS_df = IDS_df.drop('timestamp', axis=1)
    
    '''
    # Finding the null values.
    logger.info(IDS_df.isin([np.nan, np.inf, -np.inf]).sum().sum())
    # logger.info shape after dropping NaN rows
    IDS_df = IDS_df.dropna()
    logger.info(IDS_df.shape)
    IDS_df = IDS_df.reset_index(drop=True)
    # Finding the null values.
    logger.info(IDS_df.isin([np.nan, np.inf, -np.inf]).sum().sum())
    '''

    # IDS_df["label"] = IDS_df["label"].apply(get_label)

    # Convert all categorical features into numerical form:
    encodings_dictionary = dict()
    for c in IDS_df.columns:
        if IDS_df[c].dtype == "object":
            encodings_dictionary[c] = LabelEncoder()
            IDS_df[c] = encodings_dictionary[c].fit_transform(IDS_df[c])

    return IDS_df