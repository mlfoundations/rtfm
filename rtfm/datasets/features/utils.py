import copy
import logging
from typing import Dict, Sequence

from tableshift.core.features import FeatureList, Feature


def overwrite_feature_list_descriptions(
    fl: FeatureList, descriptions: Dict[str, str]
) -> FeatureList:
    out_features = []
    for feature_name, new_description in descriptions.items():
        # We cannot modify Features, since they are write-protected.
        # Instead, we create a new feature and replace the existing one.
        new_feat_kwargs = copy.deepcopy(fl[feature_name].__dict__)
        new_feat_kwargs.update({"name_extended": new_description})
        out_features.append(Feature(**new_feat_kwargs))
    # From the original dataset, copy any features not in descriptions
    for feature in fl.features:
        if feature.name not in descriptions:
            logging.info(f"copying feature with no updated description: {feature.name}")
            out_features.append(copy.deepcopy(feature))
    return FeatureList(features=out_features, documentation=fl.documentation)


def drop_features_by_name(fl: FeatureList, names_to_drop: Sequence[str]) -> FeatureList:
    """Create a new FeatureList, from fl, which drops the features in names_to_drop."""
    out_features = []
    assert all(
        name in fl.names for name in names_to_drop
    ), f"some of features to drop {names_to_drop} not in FeatureList with names {fl.names}"
    for feature in fl.features:
        if feature.name not in names_to_drop:
            out_features.append(copy.deepcopy(feature))
    return FeatureList(features=out_features, documentation=fl.documentation)


def subset_features_by_name(
    fl: FeatureList, names_to_keep: Sequence[str]
) -> FeatureList:
    """Create a new FeatureList, from fl, which only retains the features in names_to_keep."""
    out_features = []
    assert all(
        name in fl.names for name in names_to_keep
    ), f"some of features to drop {names_to_keep} not in FeatureList with names {fl.names}"
    for feature in names_to_keep:
        out_features.append(copy.deepcopy(fl[feature]))
    return FeatureList(features=out_features, documentation=fl.documentation)
