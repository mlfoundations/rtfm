import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Union, Sequence, Optional, Dict, Any, List

import numpy as np
import pandas as pd
import torch
import torch.cuda
import torch.utils
import torchmetrics
import transformers
import wandb
import webdataset as wds
from accelerate import Accelerator
from datasets import IterableDataset, Dataset
from einops import repeat
from tqdm import tqdm

import rtfm.data
from rtfm.configs import TrainConfig
from rtfm.generation_utils import (
    make_eoc_stopping_criterion,
    prepare_input_ids_and_attention_mask_for_generation,
    parse_generated_text,
)
from rtfm.special_tokens import EOC_TOKEN
from rtfm.torch_utils import batch_to_xpu
from rtfm.utils import timestamp


@dataclass
class PostprocessedPredictions:
    decoded_preds: List[str]
    decoded_labels: List[str]
    invalid_predictions_count: int


class Evaluator(ABC):
    """Object to do model evaluation."""

    @abstractmethod
    def evaluate(
        self,
        model,
        tokenizer,
        train_config: TrainConfig,
        dataset: Union[IterableDataset, Dataset],
        accelerator: Accelerator,
        labels: Sequence[str],
        wandb_logging_prefix: Optional[str] = None,
        step: Optional[int] = None,
        normalize_length=False,
    ) -> Dict[str, Any]:
        raise


def build_evaluators(train_config: TrainConfig) -> List[Evaluator]:
    """Build a list of Evaluators based on the specifications in training_args."""
    evaluators = []

    if train_config.eval_open_vocabulary:
        evaluators.append(OpenVocabularyEvaluator())
    if train_config.eval_closed_vocabulary:
        evaluators.append(ClosedVocabularyEvaluator())
    return evaluators


class OpenVocabularyEvaluator(Evaluator):
    def evaluate(
        self,
        model,
        tokenizer: transformers.PreTrainedTokenizer,
        train_config: TrainConfig,
        dataset: Union[IterableDataset, Dataset],
        labels: Sequence[str],
        max_new_tokens=128,
        wandb_logging_prefix: Optional[str] = None,
        step: Optional[int] = None,
        normalize_length: bool = None,
    ) -> Dict[str, Union[int, float]]:
        del labels
        del normalize_length

        data_collator = rtfm.data.DataCollatorForSupervisedDataset(tokenizer)

        all_preds = []
        all_labels = []

        def postprocess(predictions, labels) -> PostprocessedPredictions:
            predictions = predictions.cpu().numpy()
            labels = labels.cpu().numpy()

            decoded_preds = tokenizer.batch_decode(predictions)

            # Replace -100 in the labels as we can't decode them.
            labels = [
                [tok for tok in label if tok not in (tokenizer.pad_token_id, -100)]
                for label in labels.tolist()
            ]

            decoded_labels = tokenizer.batch_decode(labels)

            # IMPORTANT!!! You must strip the text of the labels and predictions
            # before checking. The SentencePiece tokenizer adds spaces between words
            # (i.e. '###THIS@@@' --> '### THIS@@@' after tokenize + re-encoding). We
            # need to remove these to ensure we are checking correctly.

            # Postprocess the predictions
            decoded_preds = [pred.strip() for pred in decoded_preds]
            parsed_preds_and_is_valid = [parse_generated_text(x) for x in decoded_preds]
            invalid_preds = sum(not x[1] for x in parsed_preds_and_is_valid)
            # Handle case where EOC token is not in text.
            parsed_preds = [
                p if p is not None else "" for p, _ in parsed_preds_and_is_valid
            ]

            # Postprocess the labels
            parsed_labels = [
                label.split(EOC_TOKEN)[0].strip() for label in decoded_labels
            ]
            return PostprocessedPredictions(parsed_preds, parsed_labels, invalid_preds)

        loader = torch.utils.data.DataLoader(
            dataset
            if (
                isinstance(dataset, wds.WebDataset)
                or isinstance(dataset, wds.DataPipeline)
            )
            else dataset.with_format("torch"),
            collate_fn=data_collator,
            batch_size=1,
        )

        log_preds = train_config.eval_upload_predictions != "no"
        preds_for_logging = defaultdict(list)

        model.eval()
        local_samples_seen = 0
        oom_count = 0
        invalid_predictions = 0

        start_time = timestamp()
        for batch in tqdm(
            loader, desc="eval_open_vocab", total=train_config.eval_max_samples
        ):
            with torch.no_grad():
                batch = batch_to_xpu(batch)
                assert (
                    len(batch["input_ids"]) == 1
                ), "only batch size of 1 is supported for evals."

                # Cut off the input_ids before the labels begin to prevent leakage in generate().
                (
                    input_ids,
                    attention_mask,
                ) = prepare_input_ids_and_attention_mask_for_generation(batch)

                stopping_criterion = make_eoc_stopping_criterion(input_ids, tokenizer)

                available_context_window_tokens = tokenizer.model_max_length - len(
                    input_ids
                )
                if available_context_window_tokens < 2:
                    logging.warning(
                        f"skipping call to .generate() with input of length {len(input_ids)}"
                    )
                    continue

                generated_tokens = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=min(max_new_tokens, available_context_window_tokens),
                    stopping_criteria=[stopping_criterion],
                )

                labels = batch["labels"]

                assert labels.numel() > 0, "got empty labels tensor."

                postprocessed = postprocess(generated_tokens, labels)

                all_preds.extend(postprocessed.decoded_preds)
                all_labels.extend(postprocessed.decoded_labels)

                invalid_predictions += postprocessed.invalid_predictions_count

                if log_preds:  # accumulate predictions, labels, and inputs for logging.
                    preds_for_logging["predictions"].append(postprocessed.decoded_preds)
                    preds_for_logging["labels"].append(postprocessed.decoded_labels)
                    preds_for_logging["input_text"].append(
                        tokenizer.batch_decode(input_ids)
                    )

            local_samples_seen += len(input_ids)

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
            "oom_count_open_vocab": oom_count,
            "secs_per_sample_open_vocab": runtime / float(local_samples_seen)
            if local_samples_seen > 0
            else np.nan,
        }

        if results is not None:
            results.update(extra_metrics)
        else:
            results = extra_metrics

        if log_preds:  # log the predictions, either to wandb or to the console.
            df = pd.DataFrame(preds_for_logging)
            if "wandb" in train_config.report_to:
                table = wandb.Table(dataframe=df)
                wandb.log(
                    {f"{wandb_logging_prefix}_predictions_table": table}, step=step
                )
            else:
                logging.info(
                    f"predictions and results for {wandb_logging_prefix}:\n{df}"
                )
        if "exact_match" in results:  # this will only be triggered on the main process
            results["accuracy_open_vocab"] = results.pop("exact_match")

        return results


class ClosedVocabularyEvaluator(Evaluator):
    def evaluate(
        self,
        model,
        tokenizer,
        train_config: TrainConfig,
        dataset: Union[IterableDataset, Dataset],
        accelerator: Accelerator,
        labels: Sequence[str],
        wandb_logging_prefix: Optional[str] = None,
        step: Optional[int] = None,
        normalize_length=False,
    ) -> Dict[str, Union[int, float]]:
        """Evaluate with log-likelihoods."""
        raise NotImplementedError(
            "this class needs to be updated similar to OpenVocabularyEvaluator"
            " before it can be used, to ensure there is no labels leakage from "
            "handling of the data."
        )
        accelerator.wait_for_everyone()
        accelerator.print(f"starting evaluation on process {accelerator.process_index}")
        # Compare to https://github.com/huggingface/accelerate/blob/main/examples/by_feature/multi_process_metrics.py

        data_collator = rtfm.data.DataCollatorForSupervisedDataset(tokenizer)

        loader = torch.utils.data.DataLoader(
            dataset
            if (
                isinstance(dataset, wds.WebDataset)
                or isinstance(dataset, wds.DataPipeline)
            )
            else dataset.with_format("torch"),
            collate_fn=data_collator,
            batch_size=train_config.per_device_eval_batch_size,
        )

        labels_to_tokens: Dict[str, torch.Tensor] = {
            k: tokenizer(k, return_tensors="pt").input_ids.to(accelerator.device)
            for k in labels
        }
        # Drop BOS token from classname, most tokenizers add it by default
        for k in labels_to_tokens.keys():
            if labels_to_tokens[k].squeeze()[0] == tokenizer.bos_token_id:
                labels_to_tokens[k] = labels_to_tokens[k][:, 1:]

        metric = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=len(labels)
        )
        metric.to(accelerator.device)
        # Metric must be assigned as model attribute to work; see multi-GPU example at:
        # https://github.com/Lightning-AI/torchmetrics#module-metrics
        model.metric = metric

        try:
            model, loader = accelerator.prepare(model, loader)
        except ValueError:  # model is already wrapped
            loader = accelerator.prepare(loader)

        log_preds = train_config.eval_upload_predictions != "no"
        if log_preds:
            preds_for_logging = defaultdict(list)

        model.eval()
        samples_seen = 0
        start_time = timestamp()

        for batch in tqdm(loader, total=train_config.eval_max_samples):
            class_logprobs = get_class_logprobs(
                batch["input_ids"],
                batch["attention_mask"],
                labels_to_tokens,
                model,
                normalize_length=normalize_length,
            )
            batch_preds = torch.argmax(class_logprobs, dim=1)

            batch_labels_text = [
                x.replace(EOC_TOKEN, "")
                for x in tokenizer.batch_decode(batch["labels"])
            ]
            assert all(
                y in labels for y in batch_labels_text
            ), f"got batch labels {batch_labels_text} but expected labels in {labels}"
            batch_targets = torch.Tensor(
                [labels.index(y) for y in batch_labels_text]
            ).to(accelerator.device)

            batch_preds = accelerator.gather(batch_preds)
            batch_targets = accelerator.gather(batch_targets)

            # Compute value of metric on current batch. Note that we explicitly
            # take torch.argmax because otherwise torchmetrics does normalization
            # which will mostly fail with the very small numbers in class_logprobs,
            # resulting in classes appearing to have equal probability.
            acc = metric(batch_preds.to(metric.device), batch_targets.to(metric.device))
            manual_acc = (batch_preds.cpu() == batch_targets.cpu()).float().mean()
            if acc != manual_acc:
                logging.warning(
                    f"Metric acc: {acc} != Manually-computed acc: {manual_acc}; this could be a bug."
                )

            samples_seen += len(batch["input_ids"])
            if log_preds:
                preds_for_logging["predictions"].extend(batch_labels_text)
                logprobs_dict = [
                    {labels[i]: v for i, v in enumerate(elem)}
                    for elem in class_logprobs.cpu().numpy().tolist()
                ]
                preds_for_logging["logprobs"].extend(logprobs_dict)
                preds_for_logging["labels"].extend(batch_labels_text)

            if samples_seen >= train_config.eval_max_samples:
                accelerator.print(f"terminating eval after {samples_seen} samples")
                break

        runtime = timestamp() - start_time
        acc = metric.compute()
        samples = (sum(metric._final_state()) / metric.num_classes).item()

        metric.reset()

        if log_preds:
            _log_preds(
                preds_for_logging=preds_for_logging,
                train_config=train_config,
                accelerator=accelerator,
                step=step,
                wandb_logging_prefix=wandb_logging_prefix,
            )
        return {
            "accuracy_closed_vocab": acc.item(),
            "samples_seen_closed_vocab": samples_seen,
            "sec_per_sample_closed_vocab": runtime / samples_seen,
            "runtime_closed_vocab": runtime,
            "samples_closed_vocab": samples,
        }


def _log_preds(
    preds_for_logging: Dict[str, Any],
    train_config: TrainConfig,
    accelerator: Accelerator,
    step: int,
    wandb_logging_prefix: str,
):
    """Log the predictions, either to wandb or to the console."""
    df = pd.DataFrame(preds_for_logging)
    if "wandb" in train_config.report_to and accelerator.is_main_process:
        table = wandb.Table(dataframe=df)
        wandb.log({f"{wandb_logging_prefix}_predictions_table": table}, step=step)
    else:
        logging.info(f"predictions and results for {wandb_logging_prefix}:\n{df}")
    return


def get_class_logprobs(
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor,
    labels_to_tokens: Dict[str, torch.Tensor],
    model,
    normalize_length: bool = False,
) -> torch.FloatTensor:
    overall_probs = []
    with torch.no_grad():
        for label, classname_tokens in labels_to_tokens.items():
            # TODO(jpgard): make sure num_tokens_in_classname is correct w/multi-token labels
            num_tokens_in_classname = classname_tokens.shape[1]
            classname_tokens = repeat(
                classname_tokens, "b s -> (repeat b) s", repeat=len(input_ids)
            )
            _input_ids = torch.cat((input_ids, classname_tokens), dim=1)
            _attention_mask = torch.cat(
                [attention_mask, torch.ones_like(classname_tokens).bool()], dim=1
            )
            logits = model(_input_ids, attention_mask=_attention_mask).logits
            logprobs = torch.log_softmax(logits, dim=-1)

            # Extract the probabilities for only the classname tokens
            gen_probs = logprobs[
                :, -num_tokens_in_classname - 1 : -1, :
            ]  # (B, num_tokens_in_classname, vocab_len)
            gen_probs = torch.gather(
                gen_probs, 2, classname_tokens[:, :, None]
            ).squeeze(-1)

            # Aggregate probabilities over tokens in the classname
            if normalize_length:
                class_prob = torch.mean(gen_probs, dim=1)
            else:
                class_prob = torch.sum(gen_probs, dim=1)
            overall_probs.append(class_prob)  # (B, 1)
    return torch.vstack(overall_probs).T  # [B, num_classes]
