# %%
# Secret file used for loading data, masking data source from github.
import numpy as np
import pandas as pd

class data_load:
    def __init__(self, data = input('Please insert data file:')):
        self.data = data

    def import_data(self, X: np.ndarray, y=None) -> 'RowStandardScaler':
        """
        Fit scaler on X.
        Transpose X to scale rows instead of columns
        """
        X = X.T
        self.scaler.fit(X)
        return self
# %%
data = input('Please insert data file:')
data
# %%
