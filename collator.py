from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
tokenized_dataset = Dataset.load_from_disk('/dev/shm/train', keep_in_memory=True)

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
    pad_to_multiple_of=112,
)

def collate_and_flatten(example):
    masked = collator([example]) 
    return {
        "input_ids": masked["input_ids"][0],
        "attention_mask": masked["attention_mask"][0],
        "labels": masked["labels"][0],
    }

masked_dataset = tokenized_dataset.map(collate_and_flatten, remove_columns=tokenized_dataset.column_names, num_proc=26)

masked_dataset.save_to_disk("/dev/shm/train_masked")