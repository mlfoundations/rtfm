import os

import pandas as pd
from tableshift.core.data_source import OfflineDataSource


def preprocess_nhis_diabetes(df) -> pd.DataFrame:
    df["diabetes"] = (df["DIABETICAGE"] <= 85).astype(int)
    df.drop(columns=["DIABETICAGE"], inplace=True)
    return df


def preprocess_dhs_diabetes(df) -> pd.DataFrame:
    df["ALFREQ"].fillna(998.0, inplace=True)
    df["HIGHBP"].fillna(9.0, inplace=True)
    df["RELIGION"].fillna(9998.0, inplace=True)
    return df[df["DIABETES"].isin([0.0, 1.0])]


class NHISDataSource(OfflineDataSource):
    def __init__(self, **kwargs):
        super().__init__(preprocess_fn=preprocess_nhis_diabetes, **kwargs)

    def _load_data(self) -> pd.DataFrame:
        fp = os.path.join(self.cache_dir, "nhis_00003.csv")
        return pd.read_csv(fp)


class DHSDataSource(OfflineDataSource):
    def __init__(self, **kwargs):
        super().__init__(preprocess_fn=preprocess_dhs_diabetes, **kwargs)

    def _load_data(self) -> pd.DataFrame:
        fp = os.path.join(self.cache_dir, "dhs_00002.csv")
        return pd.read_csv(fp)
