import torch
from torch.utils.data.dataset import Dataset


class SequenceDataset(Dataset):
    """
    Converts a pandas datafame into a 3d torch-compatible dataset. The dims are: bs x seq_len x feature_len.

    The returned components are:
    * endog – endogenous feature, usually the same as target
    * seas_mask – indexed seasonality
    * exog_cat – exogenous categorical features (int/long encoded)
    * exog_cnt – exnogenous continuous features
    * target – target to forecast

    """
    def __init__(self,
                 df,
                 endog_columns,
                 seas_mask_columns,
                 exog_cat_columns,
                 exog_cnt_columns,
                 target_columns,
                 history_steps,
                 future_steps,
                 convert_to_tensor=True,
                 is_cuda=False,
                 copy=False):
        self.df = df
        self.endog_columns = endog_columns
        self.seas_mask_columns = seas_mask_columns
        self.exog_cat_columns = exog_cat_columns
        self.exog_cnt_columns = exog_cnt_columns
        self.target_columns = target_columns
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.copy = copy

        if is_cuda:
            self.convert = lambda a: torch.from_numpy(a).cuda()
        else:
            self.convert = lambda a: torch.from_numpy(a)

    def __getitem__(self, index):
        endog_slice = slice(index, index + self.history_steps)
        exog_slice = slice(index, index + self.history_steps + self.future_steps)
        target_slice = slice(index + self.history_steps, index + self.history_steps + self.future_steps)

        endog = self.df.iloc[endog_slice][self.endog_columns]
        seas_mask = self.df.iloc[exog_slice][self.seas_mask_columns]
        exog_cat = self.df.iloc[exog_slice][self.exog_cat_columns]
        exog_cnt = self.df.iloc[exog_slice][self.exog_cnt_columns]
        target = self.df.iloc[target_slice][self.target_columns]

        if self.copy:
            endog = endog.copy()
            seas_mask = seas_mask.copy()
            exog_cat = exog_cat.copy()
            exog_cnt = exog_cnt.copy()
            target = target.copy()

        endog = self.convert(endog.values).float()
        seas_mask = self.convert(seas_mask.values).long()
        exog_cat = self.convert(exog_cat.values).long()
        exog_cnt = self.convert(exog_cnt.values).float()
        target = self.convert(target.values).float()

        return endog, seas_mask, exog_cat, exog_cnt, target

    def __len__(self):
        return self.df.shape[0] - self.history_steps - self.future_steps
