# Adapted from transformers pull request: https://github.com/huggingface/transformers/pull/18904

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from transformers import BatchEncoding
from transformers.models.bart.modeling_bart import BartForConditionalGeneration, shift_tokens_right


@dataclass
class DataCollatorForBartDenoisingLM:
    """
    Data collator used for BART denoising language modeling.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data
    """

    decoder_start_token_id: int

    def __call__(self, examples: List[Dict[str, List[int]]]) -> BatchEncoding:
        batch = BatchEncoding(
            {k: torch.LongTensor([examples[i][k] for i in range(len(examples))]) for k, v in examples[0].items()}
        )

        batch["labels"] = batch["input_ids"].clone()

        batch["decoder_input_ids"] = torch.tensor(np.array([shift_tokens_right(batch["labels"][:,i], 1, self.decoder_start_token_id)
                                                            for i in range(batch["labels"].shape[1])]))

        batch['labels'][batch['labels'] == 1] = -100

        batch["attention_mask"] = (batch["input_ids"] != 1).long()
        batch["decoder_attention_mask"] = (batch["decoder_input_ids"] != 1).long()

        if 'cond_input_ids' in batch:
            batch["cond_attention_mask"] = (batch["cond_input_ids"] != 1).long()

        return batch

def main():
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    
    from text_dataset import get_dataset
    dataset = get_dataset('e2e')
    
    # import pdb; pdb.set_trace()
    
    dl = DataLoader(
        dataset['train'],
        collate_fn=DataCollatorForBartDenoisingLM(),
        batch_size=4,
        shuffle=True,
    ) 


if __name__ == "__main__":
    main()