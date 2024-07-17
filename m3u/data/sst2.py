import numpy as np
from .base import DeletionData


class DeletionDataForSST2(DeletionData):
    def prepare_df_dr(self):
        all_idx = np.arange(self.train_data.shape[0])
        df_data = self.train_data[all_idx[self.df_mask]]
        dr_data = self.train_data[all_idx[self.dr_mask]]

        return df_data, dr_data
