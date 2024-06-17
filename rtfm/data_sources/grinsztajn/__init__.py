import os
import zipfile
from typing import Dict

import pandas as pd
from scipy.io.arff import loadarff
from tableshift.core.data_source import (
    DataSource,
    KaggleCompetitionDataSource,
    OfflineDataSource,
)
from tableshift.core.utils import download_file
from ucimlrepo import fetch_ucirepo
from rtfm.datasets.features.grinsztajn import HIGGS_FEATURES

SLUG_TO_OPENML_DATASET_ID: Dict[str, int] = {
    "bank-marketing": 44126,
    "california": 44090,
    "credit": 44089,
    "default-of-credit-card-clients": 45036,
    "house_16H": 44123,
    "electricity": 44120,
    "covertype": 44121,
    "eye_movements": 44130,
    "MagicTelescope": 44125,
    "Higgs": 44129,
    "MiniBooNE": 44128,
    "road-safety": 42803,
    "road_safety_full": 42803,
    "road_safety_full_severity": 42803,
    "jannis": 44131,
    "pol": 44122,
}


def preprocess_bank_marketing(df: pd.DataFrame) -> pd.DataFrame:
    df["y"] = (df["y"] == "yes").astype(int)
    df[["job", "education", "contact", "poutcome"]].fillna("unknown", inplace=True)
    return df


def preprocess_covertype(df: pd.DataFrame) -> pd.DataFrame:
    # binarize the target variable
    df["Cover_Type"] = df["Cover_Type"] == 2
    return df


def preprocess_magic_telescope(df: pd.DataFrame) -> pd.DataFrame:
    df["is_signal"] = (df["class"] == "g").astype(int)
    df.drop(columns=["class"], inplace=True)
    # Add trailing colons on the predictors to match the format in Grisztajn data.
    df.columns = [c + ":" if c != "is_signal" else c for c in df.columns]
    return df


class UCIDataSource(DataSource):
    """UCI data source. Note that this will only work for datasets available through the UCI data API."""

    def __init__(self, id: int, **kwargs):
        self.id = id
        super().__init__(**kwargs)

    def _download_if_not_cached(self):
        """No-op, as data is downloaded/cached in _load_data() due to use of ucirepo Python API."""
        pass

    def _load_data(self) -> pd.DataFrame:
        data = fetch_ucirepo(id=self.id)
        return data.data.original


class BankMarketingDataSource(UCIDataSource):
    def __init__(self, **kwargs):
        super().__init__(id=222, preprocess_fn=preprocess_bank_marketing, **kwargs)


class CoverTypeDataSource(UCIDataSource):
    def __init__(self, **kwargs):
        super().__init__(id=31, preprocess_fn=preprocess_covertype, **kwargs)


class HiggsDataSource(DataSource):
    def __init__(self, **kwargs):
        super().__init__(
            resources=["https://archive.ics.uci.edu/static/public/280/higgs.zip"],
            preprocess_fn=lambda x: x,
            **kwargs,
        )

    def _download_if_not_cached(self):
        """Download files if they are not already cached."""
        for url in self.resources:
            if not os.path.exists(
                os.path.join(self.cache_dir, "HIGGS.csv.gz")
            ) and not os.path.exists(os.path.join(self.cache_dir, "higgs.zip")):
                download_file(url, self.cache_dir)

    def _load_data(self) -> pd.DataFrame:
        csv_gzip_fp = os.path.join(self.cache_dir, "HIGGS.csv.gz")
        if not os.path.exists(csv_gzip_fp):
            zip_fp = os.path.join(self.cache_dir, "higgs.zip")
            assert os.path.exists(zip_fp)
            with zipfile.ZipFile(zip_fp, "r") as zf:
                zf.extractall(self.cache_dir)
        print(f"reading Higgs CSV from {csv_gzip_fp}; this could take a few minutes...")
        df = pd.read_csv(csv_gzip_fp, compression="gzip")
        df.columns = HIGGS_FEATURES.names
        return df


class MagicTelescopeDataSource(UCIDataSource):
    def __init__(self, **kwargs):
        super().__init__(id=159, preprocess_fn=preprocess_magic_telescope, **kwargs)


class CreditDataSource(KaggleCompetitionDataSource):
    def __init__(self, preprocess_fn=lambda x: x, **kwargs):
        super().__init__(
            preprocess_fn=preprocess_fn,
            kaggle_dataset_name="GiveMeSomeCredit",
            **kwargs,
        )

    def _load_data(self) -> pd.DataFrame:
        # only use the training data, since Kaggle set sets are unlabeled.
        train_fp = os.path.join(
            self.cache_dir, self.kaggle_dataset_name, "cs-training.csv"
        )
        return pd.read_csv(train_fp)


class DefaultOfCreditCardClientsDataSource(DataSource):
    def __init__(self, **kwargs):
        super().__init__(
            preprocess_fn=lambda x: x,
            resources=[
                "https://www.openml.org/data/download/21854402/default%20of%20credit%20card%20clients.arff"
            ],
            **kwargs,
        )

    def _load_data(self) -> pd.DataFrame:
        data_file = os.path.join(
            self.cache_dir, "default%20of%20credit%20card%20clients.arff"
        )
        data = loadarff(data_file)
        return pd.DataFrame(data[0])


def preprocess_road_safety(df: pd.DataFrame) -> pd.DataFrame:
    """Drop 'unknown' sex driver."""
    df = df[df["Sex_of_Driver"].isin(["1.0", "2.0"])]
    return df


class RoadSafetyDataSource(OfflineDataSource):
    def __init__(self, **kwargs):
        super().__init__(preprocess_fn=preprocess_road_safety, **kwargs)

    def _load_data(self) -> pd.DataFrame:
        data_file = os.path.join(self.cache_dir, "road-safety-dataset.arff")
        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f"file {data_file} does not exist; if you need to, download the file from "
                "https://www.openml.org/search?type=data&sort=runs&id=42803&status=active"
            )
        try:
            import arff
        except ImportError:
            raise ImportError(
                "liac-arff not found; try running `pip install git+https://github.com/renatopp/liac-arff.git`"
            )
        # scipy does not support string attributes, so we need to use another package to load arff here
        data = arff.load(open(data_file, "r"))
        df = pd.DataFrame(data["data"])
        df.columns = [x[0] for x in data["attributes"]]
        return df


class RoadSafetySeverityDataSource(OfflineDataSource):
    def __init__(self, **kwargs):
        super().__init__(preprocess_fn=lambda x: x, **kwargs)

    def _load_data(self) -> pd.DataFrame:
        data_file = os.path.join(self.cache_dir, "road-safety-dataset.arff")
        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f"file {data_file} does not exist; if you need to, download the file from "
                "https://www.openml.org/search?type=data&sort=runs&id=42803&status=active"
            )
        try:
            import arff
        except ImportError:
            raise ImportError(
                "liac-arff not found; try running `pip install git+https://github.com/renatopp/liac-arff.git`"
            )
        # scipy does not support string attributes, so we need to use another package to load arff here
        data = arff.load(open(data_file, "r"))
        df = pd.DataFrame(data["data"])
        df.columns = [x[0] for x in data["attributes"]]
        return df
