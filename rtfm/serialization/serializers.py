import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Union, Any, Optional, Sequence

import numpy as np
import pandas as pd

from rtfm import special_tokens as tok
from rtfm.configs import SerializerConfig
from rtfm.serialization.serialization_utils import (
    shuffle_example_features,
    apply_feature_dropout,
    find_all_idxs,
    extract_metafeatures,
    strip_html_whitespace,
)

_SPECIAL_TOKENS_MAP = {
    "eoc_token": tok.EOC_TOKEN,
    "qa_sep_token": tok.QA_SEP_TOKEN,
    "ans_choices_sep_token": tok.ANS_CHOICES_SEP_TOKEN,
}


def basic_serialize_choices(choices: List[str]) -> str:
    if not choices:
        return ""
    else:
        return " or ".join(choices)


def v2_serialize_choices(
    choices: List[str],
    sep_tok: str,
    add_sep_before: bool = True,
    add_sep_after: bool = True,
) -> str:
    if not choices:
        return ""
    else:
        output = ""
        if add_sep_before:
            output += sep_tok
        output += sep_tok.join(choices)
        if add_sep_after:
            output += sep_tok
        return output


@dataclass
class RowSerializer(ABC):
    """Abstract class to serialize rows of a tabular dataset."""

    config: SerializerConfig
    strict: bool = False

    # TODO(jpgard): meta_features belongs in the SerializerConfig
    #  but moving it there requires more refactoring in data.py
    #  to extricate it from the DataArguments class.
    meta_features: Optional[Sequence[str]] = None

    def _round_to_max_precision(self, val: Any):
        """Optionally round a feature to a maximum number of decimal places."""
        assert np.issubdtype(type(val), np.number)
        if self.config.max_precision is not None:
            return round(val, self.config.max_precision)
        else:
            return val

    def _preprocess_value(self, val: Any):
        if np.issubdtype(type(val), np.number):
            val = self._round_to_max_precision(val)
        elif isinstance(val, str):
            val = str(val).strip()
            if val.endswith("."):
                val = val[:-1]
        return val

    def _check_example(self, x: Union[pd.Series, Dict[Any, Any]]) -> None:
        """Check an example to ensure it conforms to expected restrictions."""
        if isinstance(x, pd.Series):
            x = x.to_dict()
        keys = list(x.keys())
        for i, key in enumerate(keys[:-1]):
            if self.strict and any(key in x for x in keys[i + 1 :]):
                raise ValueError(f"Cannot have one key that contains another: {keys}")
        if "__metafeatures__" in x:
            # Check that every feature entry for each metafeature_corresponds to an actual feature.
            for metafeature_dict in x["__metafeatures__"].values():
                assert all(
                    metafeature_name in x
                    for metafeature_name in metafeature_dict.keys()
                )

    def _prepare_example(self, x: Union[pd.Series, Dict[Any, Any]]) -> dict:
        """Prepare an example for serialization."""
        if isinstance(x, pd.Series):
            x = x.to_dict()

        self._check_example(x)

        if self.config.shuffle_instance_features:
            x = shuffle_example_features(x)
        if self.config.feature_dropout > 0.0:
            x = apply_feature_dropout(x, self.config.feature_dropout)
        return x

    @abstractmethod
    def serialize_example(
        self,
        x: Union[pd.Series, Dict[Any, Any]],
        meta: Dict[str, Dict[str, Any]] = None,
    ) -> str:
        raise

    @abstractmethod
    def serialize_choices(self, choices: List[str] = None):
        raise

    @abstractmethod
    def deserialize_example(
        self, x: str, feature_names: Sequence[str]
    ) -> Dict[str, str]:
        """Deserialize an example.

        Note that this method does not try to guess types of data and instead returns
        only strings; as a result it cannot perfectly recover the input example (and
        only recovers its string representation)."""
        raise

    @property
    @abstractmethod
    def special_tokens(self) -> Union[Dict[str, str], None]:
        raise

    @abstractmethod
    def __call__(
        self,
        x: Union[pd.Series, Dict[Any, Any]],
        prefix_text="",
        suffix_text="",
        choices: List[str] = None,
        task_context_text="",
        meta: Dict[str, Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        raise

    def apply_to_rows(self, df: pd.DataFrame) -> List[str]:
        """Apply the RowSerializer to each row of df, return list of results."""
        return df.apply(lambda x: self(x), axis=1).tolist()


@dataclass
class BasicSerializer(RowSerializer):
    """A basic serializer. Equivalent to 'Text Template' of TabLLM."""

    example_end_char = "."  # The character placed at the end of a serialized example.

    @property
    def special_tokens(self):
        return {
            k: v for k, v in _SPECIAL_TOKENS_MAP.items() if k != "ans_choices_sep_token"
        }

    def serialize_key(self, k) -> str:
        return f"The {k} is"

    def deserialize_key(self, serialized_key) -> str:
        return re.search("^The (.*) is$", serialized_key).group(1)

    def serialize_key_and_value(self, k, v, meta: Dict[str, Any]) -> str:
        """Serialize an individual key-value pair."""
        serialized = f"{self.serialize_key(k)} {self._preprocess_value(v)}"
        if meta:
            # Note that metafeatures will only be present for some features. This is because
            # most metafeatures (quantile, scaled value, etc) are only populated for
            # specific data types.
            meta_serialized = (
                " (" + ", ".join(f"{k}:{v}" for k, v in meta.items()) + ")"
            )
            serialized += meta_serialized

        serialized += self.example_end_char

        return serialized

    def serialize_example(
        self,
        x: Union[pd.Series, Dict[Any, Any]],
        meta: Dict[str, Dict[str, Any]] = None,
    ) -> str:
        x = self._prepare_example(x)
        keys_and_values = [
            self.serialize_key_and_value(k, v, extract_metafeatures(k, meta))
            for k, v in x.items()
        ]
        keys_and_values = " ".join(keys_and_values).strip()
        return keys_and_values

    def deserialize_example(
        self, x: str, feature_names: Sequence[str]
    ) -> Dict[str, str]:
        output = {}

        # maps each serialized key to its index in x, if the serialized key occurs in x.
        serialized_key_indices = {}

        for k in feature_names:
            serialized_key = self.serialize_key(k)
            if serialized_key not in x and (not self.config.feature_dropout > 0.0):
                raise ValueError(f"Expected key {serialized_key} not in example {x}.")
            elif serialized_key in x:
                serialized_key_indices[serialized_key] = x.index(serialized_key)

        # Create a list of tuples from the dict, sorted by order of occurrence
        serialized_key_indices = sorted(
            serialized_key_indices.items(), key=lambda item: item[1]
        )

        # Extract the values for all features except the last
        for i in range(len(serialized_key_indices) - 1):
            serialized_key, key_start_index = serialized_key_indices[i]
            value_start = key_start_index + len(serialized_key) + 1  # +1 for space
            value_end = serialized_key_indices[i + 1][1]
            value = x[value_start:value_end]
            # Remove trailing ". " from value
            value = re.sub(self.example_end_char + " $", "", value)

            deserialized_key = self.deserialize_key(serialized_key)
            output[deserialized_key] = value

        # Handle the last feature.
        serialized_key, key_start_index = serialized_key_indices[-1]
        deserialized_key = self.deserialize_key(serialized_key)

        value_start = key_start_index + len(serialized_key) + 1
        value_end = x.rindex(self.example_end_char)
        value = x[value_start:value_end]
        output[deserialized_key] = value

        return output

    def serialize_choices(self, choices: List[str] = None) -> str:
        return basic_serialize_choices(choices)

    def __call__(
        self,
        x: Union[pd.Series, Dict[Any, Any]],
        prefix_text="",
        suffix_text="",
        choices: List[str] = None,
        task_context_text="",
        strict=False,
        meta: Dict[str, Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        keys_and_values = self.serialize_example(x, meta)
        if self.example_end_char in suffix_text and strict:
            raise ValueError(
                f"example end char {self.example_end_char} not "
                f"permitted in suffix text {suffix_text}."
                " Otherwise, this can cause unintended deserialization behavior."
            )

        choices_text = self.serialize_choices(choices)

        elems_to_serialize = [task_context_text, prefix_text]

        if self.config.choices_position in ("front", "both"):
            elems_to_serialize.append(choices_text)
        elems_to_serialize.extend(
            [
                keys_and_values,
                suffix_text,
            ]
        )

        if self.config.choices_position in ("back", "both"):
            elems_to_serialize.append(choices_text)

        serialized = " ".join(x.strip() for x in elems_to_serialize if x).strip()

        return serialized


@dataclass
class BasicSerializerV2(BasicSerializer):
    """A BasicSerializer that uses the default ans_choices_sep_token ('||'),
    and uses consistent delimiting across both choices lists."""

    @property
    def special_tokens(self):
        return _SPECIAL_TOKENS_MAP

    def serialize_choices(self, choices: List[str] = None) -> str:
        return v2_serialize_choices(
            choices, self.special_tokens["ans_choices_sep_token"]
        )


@dataclass
class StructuredSerializer(RowSerializer):
    """A serializer that uses delimiters for examples, keys, and values."""

    choices_start_token: str = tok.CHOICES_START_TOKEN
    choices_end_token: str = tok.CHOICES_END_TOKEN

    context_start_token: str = tok.CONTEXT_START_TOKEN
    context_end_token: str = tok.CONTEXT_END_TOKEN

    meta_start_token: str = tok.META_START_TOKEN
    meta_end_token: str = tok.META_END_TOKEN

    quantile_start_token: str = tok.QUANTILE_START_TOKEN
    quantile_end_token: str = tok.QUANTILE_END_TOKEN

    scale_start_token: str = tok.SCALE_START_TOKEN
    scale_end_token: str = tok.SCALE_END_TOKEN

    key_start_token: str = tok.KEY_START_TOKEN
    key_end_token: str = tok.KEY_END_TOKEN

    prefix_start_token: str = tok.PREFIX_START_TOKEN
    prefix_end_token: str = tok.PREFIX_END_TOKEN

    suffix_start_token: str = tok.SUFFIX_START_TOKEN
    suffix_end_token: str = tok.SUFFIX_END_TOKEN

    train_example_start_token: str = tok.TRAIN_EXAMPLE_START_TOKEN
    train_example_end_token: str = tok.TRAIN_EXAMPLE_END_TOKEN

    value_start_token: str = tok.VALUE_START_TOKEN
    value_end_token: str = tok.VALUE_END_TOKEN

    @property
    def special_tokens(self) -> Dict[str, str]:
        return {
            **_SPECIAL_TOKENS_MAP,
            **{
                k: getattr(self, k)
                for k in (
                    "choices_start_token",
                    "choices_end_token",
                    "context_start_token",
                    "context_end_token",
                    "key_start_token",
                    "key_end_token",
                    "prefix_start_token",
                    "prefix_end_token",
                    "suffix_start_token",
                    "suffix_end_token",
                    "train_example_start_token",
                    "train_example_end_token",
                    "value_start_token",
                    "value_end_token",
                )
            },
        }

    def serialize_choices(self, choices: List[str] = None) -> str:
        if not choices:
            return ""
        else:
            return (
                self.special_tokens["choices_start_token"]
                + v2_serialize_choices(
                    choices,
                    self.special_tokens["ans_choices_sep_token"],
                    add_sep_before=False,
                    add_sep_after=False,
                )
                + self.special_tokens["choices_end_token"]
            )

    @property
    def meta_tokens(self):
        return {
            "quantile": {
                "start": self.quantile_start_token,
                "end": self.quantile_end_token,
            },
            "scale": {"start": self.scale_start_token, "end": self.scale_end_token},
        }

    def serialize_key_and_value(self, k, v, meta: Dict[str, Any]) -> str:
        """Serialize an individual key-value pair."""
        serialized = f"{self.key_start_token}{k}{self.key_end_token}{self.value_start_token}{v}{self.value_end_token}"
        if (
            meta
        ):  # TODO(jpgard): should we check data_args here to ensure metafeatures should be added?
            # Note that metafeatures will only be present for some features. This is because
            # most metafeatures (quantile, scaled value, etc) are only populated for
            # specific data types.
            meta_serialized = "".join(
                f"{self.meta_tokens[k]['start']}{v}{self.meta_tokens[k]['end']}"
                for k, v in meta.items()
            )
            serialized += self.meta_start_token + meta_serialized + self.meta_end_token
        return serialized

    def serialize_example(
        self,
        x: Union[pd.Series, Dict[Any, Any]],
        meta: Dict[str, Dict[str, Any]] = None,
    ) -> str:
        x = self._prepare_example(x)
        output_elems = [
            self.serialize_key_and_value(k, v, extract_metafeatures(k, meta))
            for k, v in x.items()
        ]
        return "".join(output_elems)

    def deserialize_example(
        self, x: str, feature_names: Sequence[str]
    ) -> Dict[str, str]:
        output = {}

        # Check that there are an equal number of all expected start/end tokens.
        key_start_idxs = find_all_idxs(self.key_start_token, x)
        key_end_idxs = find_all_idxs(self.key_end_token, x)
        value_start_idxs = find_all_idxs(self.value_start_token, x)
        value_end_idxs = find_all_idxs(self.value_end_token, x)
        if not all(
            len(idxs) == len(key_start_idxs)
            for idxs in (key_end_idxs, value_start_idxs, value_end_idxs)
        ):
            raise ValueError(f"Bad example: {x}")

        for key_start, key_end, value_start, value_end in zip(
            key_start_idxs, key_end_idxs, value_start_idxs, value_end_idxs
        ):
            key = x[key_start + len(self.key_start_token) : key_end]
            val = x[value_start + len(self.value_start_token) : value_end]
            output[key] = val

        return output

    def __call__(
        self,
        x: Union[pd.Series, Dict[Any, Any]],
        prefix_text="",
        suffix_text="",
        choices: List[str] = None,
        task_context_text="",
        meta: Dict[str, Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        if prefix_text:
            prefix_text = self.prefix_start_token + prefix_text + self.prefix_end_token
        if suffix_text:
            suffix_text = self.suffix_start_token + suffix_text + self.suffix_end_token
        choices_text = self.serialize_choices(choices)
        example_serialized = (
            self.train_example_start_token
            + self.serialize_example(x, meta)
            + self.train_example_end_token
        )

        elems_to_serialize = [task_context_text, prefix_text]

        if self.config.choices_position in ("front", "both"):
            elems_to_serialize.append(choices_text)
        elems_to_serialize.extend(
            [
                example_serialized,
                suffix_text,
            ]
        )

        if self.config.choices_position in ("back", "both"):
            elems_to_serialize.append(choices_text)

        serialized = "".join(x.strip() for x in elems_to_serialize if x).strip()
        return serialized


class BaseDictBasedSerializer(RowSerializer):
    """Base class for other dictionary-based serializers to inherit from."""

    def serialize_key_and_value(self, k, v, meta: Dict[str, Any]) -> str:
        raise

    def serialize_choices(self, choices: List[str] = None):
        raise

    def serialize_example(
        self,
        x: Union[pd.Series, Dict[Any, Any]],
        meta: Dict[str, Dict[str, Any]] = None,
    ) -> str:
        raise

    def deserialize_example(
        self, x: str, feature_names: Sequence[str]
    ) -> Dict[str, str]:
        raise

    @property
    def special_tokens(self):
        raise

    def prepare_sample_dict(
        self,
        x: Union[pd.Series, Dict[Any, Any]],
        prefix_text="",
        suffix_text="",
        choices: Union[str, Sequence[str]] = "",
        task_context_text="",
        meta: Dict[str, Dict[str, Any]] = None,
    ) -> Dict[Any, Any]:
        """Preprocess a sample by preparing a sample dict for further serializer=specific formatting.

        The sample dicts have the format:
            {
                'features': {'feature_name': {'value': val, 'quantile': 0.01, ...}},
                'prefix_text': 'my prefix text here',
                ...
            }
        """
        # serialize the example
        x = self._prepare_example(x)

        keys_and_values = {}
        for k, v in x.items():
            feature_dict = {"value": self._preprocess_value(v)}
            k_metafeatures = extract_metafeatures(k, meta)
            if k_metafeatures:
                feature_dict.update(k_metafeatures)
            keys_and_values[k] = feature_dict

        df_dict = {"features": keys_and_values}
        if prefix_text:
            df_dict["prefix"] = prefix_text
        if suffix_text:
            df_dict["suffix"] = suffix_text
        if choices:
            df_dict["choices"] = choices
        if task_context_text:
            df_dict["task_context"] = task_context_text
        return df_dict

    def __call__(
        self,
        x: Union[pd.Series, Dict[Any, Any]],
        prefix_text="",
        suffix_text="",
        choices: List[str] = None,
        task_context_text="",
        meta: Dict[str, Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        raise


class PandasSeriesSerializer(BaseDictBasedSerializer):
    """Serialize as a pd.Series."""

    def serialize_key_and_value(self, k, v, meta: Dict[str, Any]) -> str:
        """This method should never be called; all k/v pairs are
        serialized only once, together, to form JSON that can be
        properly parsed back to the original format."""
        raise

    def serialize_choices(self, choices: List[str] = None):
        return basic_serialize_choices(choices)

    def serialize_example(
        self,
        x: Union[pd.Series, Dict[Any, Any]],
        meta: Dict[str, Dict[str, Any]] = None,
    ) -> str:
        """This method should never be called; all pieces of the example are
        serialized only once, together, to form JSON that can be
        properly parsed back to the original format."""
        raise

    def deserialize_example(
        self, x: str, feature_names: Sequence[str]
    ) -> Dict[str, str]:
        """Deserialize an example.

        Note that this method does not try to guess types of data and instead returns
        only strings; as a result it cannot perfectly recover the input example (and
        only recovers its string representation)."""
        del feature_names
        return eval(x)["features"]

    @property
    def special_tokens(self):
        return _SPECIAL_TOKENS_MAP

    def __call__(
        self,
        x: Union[pd.Series, Dict[Any, Any]],
        prefix_text="",
        suffix_text="",
        choices: List[str] = None,
        task_context_text="",
        meta: Dict[str, Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        df_dict = self.prepare_sample_dict(
            x=x,
            prefix_text=prefix_text,
            suffix_text=suffix_text,
            choices=self.serialize_choices(choices),
            task_context_text=task_context_text,
            meta=meta,
        )
        serialized = f"pd.Series({df_dict})"
        return serialized


class HtmlSerializer(BaseDictBasedSerializer):
    """A serializer that renders an HTML table of an example.

    Example output shown below.
    #### BEGIN EXAMPLE OUTPUT ####
        <table border="1" class="dataframe">
          <thead>
            <tr style="text-align: right;">
              <th></th>
              <th>0</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th>features</th>
              <td>{'float_feature': {'value': -0.004}, 'bool_feature': {'value': True}, 'int_feature': {'value': 5968}, 'str_feature': {'value': 'my_category'}}</td>
            </tr>
            <tr>
              <th>prefix</th>
              <td>This is an observation drawn from a dataset.</td>
            </tr>
            <tr>
              <th>suffix</th>
              <td>What is the label??</td>
            </tr>
            <tr>
              <th>choices</th>
              <td>2 or 1 or 0.</td>
            </tr>
            <tr>
              <th>task_context</th>
              <td>This is the task context, which provides context.</td>
            </tr>
          </tbody>
        </table>
    #### END EXAMPLE OUTPUT ####

    """

    def serialize_key_and_value(self, k, v, meta: Dict[str, Any]) -> str:
        """This method should never be called; all k/v pairs are
        serialized only once, together, to form JSON that can be
        properly parsed back to the original format."""
        raise

    def serialize_choices(self, choices: List[str] = None):
        return basic_serialize_choices(choices)

    def serialize_example(
        self,
        x: Union[pd.Series, Dict[Any, Any]],
        meta: Dict[str, Dict[str, Any]] = None,
    ) -> str:
        """This method should never be called; all pieces of the example are
        serialized only once, together, to form JSON that can be
        properly parsed back to the original format."""
        raise

    def deserialize_example(
        self, x: str, feature_names: Sequence[str]
    ) -> Dict[str, str]:
        """Deserialize an example.

        Note that this method does not try to guess types of data and instead returns
        only strings; as a result it cannot perfectly recover the input example (and
        only recovers its string representation)."""
        del feature_names
        return eval(x)["features"]

    @property
    def special_tokens(self):
        return _SPECIAL_TOKENS_MAP

    def __call__(
        self,
        x: Union[pd.Series, Dict[Any, Any]],
        prefix_text="",
        suffix_text="",
        choices: List[str] = None,
        task_context_text="",
        meta: Dict[str, Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        example_dict = self.prepare_sample_dict(
            x=x,
            prefix_text=prefix_text,
            suffix_text=suffix_text,
            choices=self.serialize_choices(choices),
            task_context_text=task_context_text,
            meta=meta,
        )
        return pd.Series(example_dict).to_frame().to_html()


class HtmlNoWhitespaceSerializer(HtmlSerializer):
    def __call__(
        self,
        x: Union[pd.Series, Dict[Any, Any]],
        prefix_text="",
        suffix_text="",
        choices: List[str] = None,
        task_context_text="",
        meta: Dict[str, Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        serialized_to_html = super().__call__(
            x=x,
            prefix_text=prefix_text,
            suffix_text=suffix_text,
            choices=choices,
            task_context_text=task_context_text,
            meta=meta,
            *args,
            **kwargs,
        )
        return strip_html_whitespace(serialized_to_html)


class JsonSerializer(BaseDictBasedSerializer):
    """Serialize the inputs to a nested JSON structure.

    Example output:

    #### BEGIN EXAMPLE OUTPUT ####
        {'choices': '2 or 1 or 0.',
         'features': {'bool_feature': {'value': True},
                      'float_feature': {'quantile': 0.0099,
                                        'scale': -0.2,
                                        'value': -0.004},
                      'int_feature': {'quantile': 0.01, 'scale': -0.99, 'value': 5968},
                      'str_feature': {'value': 'my_category'}},
         'prefix': 'This is an observation drawn from a dataset.',
         'suffix': 'What is the label??',
         'task_context': 'This is the task context, which provides context.'}
    #### END EXAMPLE OUTPUT ####

    """

    def serialize_key_and_value(self, k, v, meta: Dict[str, Any]) -> str:
        """This method should never be called; all k/v pairs are
        serialized only once, together, to form JSON that can be
        properly parsed back to the original format."""
        raise

    def serialize_choices(self, choices: List[str] = None):
        raise

    def serialize_example(
        self,
        x: Union[pd.Series, Dict[Any, Any]],
        meta: Dict[str, Dict[str, Any]] = None,
    ) -> str:
        """This method should never be called; all pieces of the example are
        serialized only once, together, to form JSON that can be
        properly parsed back to the original format."""
        raise

    def deserialize_example(
        self, x: str, feature_names: Sequence[str]
    ) -> Dict[str, str]:
        """Deserialize an example.

        Note that this method does not try to guess types of data and instead returns
        only strings; as a result it cannot perfectly recover the input example (and
        only recovers its string representation)."""
        del feature_names
        return eval(x)["features"]

    @property
    def special_tokens(self):
        return _SPECIAL_TOKENS_MAP

    def __call__(
        self,
        x: Union[pd.Series, Dict[Any, Any]],
        prefix_text="",
        suffix_text="",
        choices: List[str] = None,
        task_context_text="",
        meta: Dict[str, Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        example_dict = self.prepare_sample_dict(
            x=x,
            prefix_text=prefix_text,
            suffix_text=suffix_text,
            choices=choices,
            task_context_text=task_context_text,
            meta=meta,
        )
        return json.dumps(example_dict)


def get_serializer(config: SerializerConfig, **kwargs) -> RowSerializer:
    serializer = eval(config.serializer_cls)(**kwargs, config=config)
    return serializer
