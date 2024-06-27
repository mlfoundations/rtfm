import logging
from typing import Dict, Tuple

import torch
import transformers

from rtfm.special_tokens import EOC_TOKEN, IGNORE_INDEX, QA_SEP_TOKEN


class KeywordsStoppingCriteria(transformers.StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = [tokenizer(keyword).input_ids for keyword in keywords]
        self.keyword_ids = [
            keyword_id[0]
            for keyword_id in self.keyword_ids
            if isinstance(keyword_id, list) and len(keyword_id) == 1
        ]
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(
        self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]

        for keyword_id in self.keyword_ids:
            if output_ids[0, -1] == keyword_id:
                return True
        outputs = self.tokenizer.batch_decode(
            output_ids[:, self.start_len :], skip_special_tokens=True
        )[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False


def make_eoc_stopping_criterion(input_ids, tokenizer):
    return KeywordsStoppingCriteria(
        keywords=[EOC_TOKEN],
        input_ids=input_ids,
        tokenizer=tokenizer,
    )


def prepare_input_ids_and_attention_mask_for_generation(
    batch: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract the inputs and attention mask corresponding *only* to the input text.

    This prevents leakage during generation, as the text corresponding to labels
    will never be provided to .generate() if the input ids and attention mask
    from this function are used.
    """
    labels_start_pos = torch.where(batch["labels"] != IGNORE_INDEX)[1][0]
    input_ids = batch["input_ids"][:, :labels_start_pos]
    attention_mask = batch["attention_mask"][:, :labels_start_pos]
    return input_ids, attention_mask


def parse_generated_text(text: str) -> Tuple[str, bool]:
    """Return the parsed text, and a boolean indicating whether the completion is valid.

    Currently, the only invalid completions are ones where there is no EOC_TOKEN following the final QA_SEP_TOKEN;
     in this case this function returns everything after the final QA_SEP_TOKEN.

    Raises ValueError if the text does not contain QA_SEP_TOKEN.
    """
    if QA_SEP_TOKEN not in text:
        raise ValueError(
            f"Invalid QA text: {text}; must contain QA_SEP_TOKEN {QA_SEP_TOKEN}"
        )

    full_completion = text.rsplit(QA_SEP_TOKEN, maxsplit=1)[1]

    if not EOC_TOKEN in full_completion:
        logging.debug(
            "EOC token %s not detected in generated text %s"
            % (EOC_TOKEN, full_completion)
        )
        return full_completion, False

    parsed_completion = full_completion.split(EOC_TOKEN, maxsplit=1)[0]
    if not parsed_completion:
        logging.warning(f"got empty completion after parsing from text {text}")
    return parsed_completion.strip(), True
