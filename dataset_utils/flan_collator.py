# Adapted from transformers pull request: https://github.com/huggingface/transformers/pull/18904

from dataclasses import dataclass
from typing import List

import torch

from transformers import BatchEncoding


@dataclass
class DataCollatorForFlanLM:
    """
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data
    """

    def __call__(self, examples: List[str]) -> BatchEncoding:
        batch = BatchEncoding(
            {k: torch.LongTensor([examples[i][k] for i in range(len(examples))]) for k, v in examples[0].items()}
        )

        batch["labels"] = batch["input_ids"].clone()
        batch['labels'][batch['labels'] == 0] = -100

        if 'cond_input_ids' in batch:
            batch["cond_attention_mask"] = (batch["cond_input_ids"] != 1).long()

        return batch