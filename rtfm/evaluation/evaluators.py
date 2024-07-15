from typing import Union, Dict

import numpy as np
import pandas as pd
import torch
import torch.cuda
import torch.utils
from tqdm import tqdm

from rtfm.configs import TrainConfig
from rtfm.inference_utils import InvalidInferenceError, ShotSelector, InferenceModel
from rtfm.utils import timestamp


class OpenVocabularyEvaluator:
    def evaluate(
        self,
        inference_model: InferenceModel,
        shot_selector: ShotSelector,
        num_shots: int,
        train_config: TrainConfig,
        df: pd.DataFrame,
        target_colname: str,
        handle_invalid_predictions: str = "warn",
    ) -> Dict[str, Union[int, float]]:
        target_choices = sorted(df[target_colname].unique().tolist())

        all_preds = []
        all_labels = []

        local_samples_seen = 0
        invalid_predictions = 0

        start_time = timestamp()
        for idx in tqdm(range(len(df)), desc="eval_open_vocab", total=len(df)):
            with torch.no_grad():
                target_example = df.iloc[[idx]]
                shots = shot_selector.select_shots(df, num_shots, idx)
                try:
                    decoded_preds = inference_model.predict(
                        target_example=target_example,
                        target_colname=target_colname,
                        target_choices=target_choices,
                        labeled_examples=shots,
                        handle_invalid_predictions=handle_invalid_predictions,
                    )

                except InvalidInferenceError:
                    decoded_preds = [
                        None,
                    ]
                    invalid_predictions += 1

                all_preds.extend(decoded_preds)
                all_labels.extend(target_example[target_colname].tolist())

            local_samples_seen += 1

            if len(all_preds) >= train_config.eval_max_samples:
                break

        runtime = timestamp() - start_time

        results = {
            "accuracy_open_vocab": np.mean(
                [int(x == y) for x, y in zip(all_preds, all_labels)]
            )
        }

        extra_metrics = {
            "samples_seen_per_gpu_open_vocab": local_samples_seen,
            "samples_seen_total_open_vocab": local_samples_seen,
            "invalid_prediction_rate_open_vocab": invalid_predictions
            / float(local_samples_seen)
            if local_samples_seen > 0
            else np.nan,
            "runtime_secs_open_vocab": runtime,
            "secs_per_sample_open_vocab": runtime / float(local_samples_seen)
            if local_samples_seen > 0
            else np.nan,
        }

        if results is not None:
            results.update(extra_metrics)
        else:
            results = extra_metrics

        if "exact_match" in results:  # this will only be triggered on the main process
            results["accuracy_open_vocab"] = results.pop("exact_match")

        return results
