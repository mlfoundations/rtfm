from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn import datasets
from tableshift.core.data_source import OfflineDataSource
from tableshift.core.features import FeatureList, Feature

n_samples = 100_000
seed = 30

# Generic FeatureList to represent all synthetic datasets.
DEFAULT_SYNTHETIC_FEATURES = FeatureList(
    features=[
        Feature("x_0", float),
        Feature("x_1", float),
        Feature("y", int, is_target=True),
    ]
)


@dataclass
class DatasetGenerator:
    """Class that generates synthetic data from a distribution."""

    n_samples: int = n_samples
    seed: int = seed

    def __init__(self):
        X, y = self._generate_data()
        self.X = X
        self.y = y.astype(int)

    def get_data(self) -> pd.DataFrame:
        return pd.concat(
            (
                pd.DataFrame(
                    self.X, columns=[f"x_{i}" for i in range(self.X.shape[1])]
                ),
                pd.Series(self.y).to_frame(name="y"),
            ),
            axis=1,
        )

    def _generate_data(self):
        raise


@dataclass
class NoisyCirclesDataGenerator(DatasetGenerator):
    factor = 0.5
    noise = 0.05

    def __post_init__(self):
        super().__init__()

    def _generate_data(self) -> pd.DataFrame:
        return datasets.make_circles(
            n_samples=self.n_samples,
            factor=self.factor,
            noise=self.noise,
            random_state=self.seed,
        )


@dataclass
class NoisyMoonsDataGenerator(DatasetGenerator):
    noise = 0.05

    def __post_init__(self):
        super().__init__()

    def _generate_data(self):
        return datasets.make_moons(
            n_samples=self.n_samples, noise=self.noise, random_state=self.seed
        )


class BlobsDataGenerator(DatasetGenerator):
    def _generate_data(self):
        return datasets.make_blobs(n_samples=self.n_samples, random_state=self.seed)


@dataclass
class NoisyLinearlySeparableDataGenerator(DatasetGenerator):
    noise_scale = 0.1  # higher values decrease sharpness of boundary

    def __post_init__(self):
        super().__init__()

    def _generate_data(self):
        # linearly separable data
        rng = np.random.RandomState(self.seed)
        x_rand = rng.rand(self.n_samples, 2)
        y_rand = (
            x_rand[:, 1]
            - x_rand[:, 0]
            + rng.normal(scale=self.noise_scale, size=self.n_samples)
            > 0
        )
        return x_rand, y_rand


@dataclass
class AnisotropicDataGenerator(DatasetGenerator):
    transformation = [[0.6, -0.6], [-0.4, 0.8]]

    def __post_init__(self):
        super().__init__()

    def _generate_data(self):
        # Anisotropically distributed data
        X, y = datasets.make_blobs(n_samples=self.n_samples, random_state=self.seed)
        X_aniso = np.dot(X, self.transformation)
        return X_aniso, y


@dataclass
class VariedBlobsDatasetGenerator(DatasetGenerator):
    # Blobs with varied variances
    cluster_std = [1.0, 2.5, 0.5]

    def __post_init__(self):
        super().__init__()

    # blobs with varied variances
    def _generate_data(self):
        return datasets.make_blobs(
            n_samples=self.n_samples,
            cluster_std=self.cluster_std,
            random_state=self.seed,
        )


synthetic_datasets: Dict[str, pd.DataFrame] = {
    "noisy_circles": NoisyCirclesDataGenerator().get_data(),
    "noisy_moons": NoisyMoonsDataGenerator().get_data(),
    "varied_clusters": VariedBlobsDatasetGenerator().get_data(),
    "anisotropic_clusters": AnisotropicDataGenerator().get_data(),
    "blobs": BlobsDataGenerator().get_data(),
    "linearly_separable": NoisyLinearlySeparableDataGenerator().get_data(),
}

SYNTHETIC_DATASET_NAMES = list(synthetic_datasets.keys())


class SyntheticDataSource(OfflineDataSource):
    """Generic data source to represent any synthetic dataset."""

    def __init__(self, name: str, **kwargs):
        assert (
            name in synthetic_datasets.keys()
        ), f"provided name {name} is not in {synthetic_datasets.keys()}"
        self.name = name
        super().__init__(preprocess_fn=lambda x: x, **kwargs)

    def _load_data(self) -> pd.DataFrame:
        return synthetic_datasets[self.name]
