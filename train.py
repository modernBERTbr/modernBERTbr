import logging

from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers.models.modernbert import ModernBertConfig, ModernBertForMaskedLM


def train():
    # Disable progress bars and verbose logging to reduce clutter
    disable_progress_bar()
    logging.getLogger("datasets").setLevel(logging.ERROR)
    
    # 1. Load WordPiece tokenizer from BERTimbau
    tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

    # 2. Create a fresh config for ModernBERT, using BERTimbau's 30.000 vocab size with ModernBERT's paper default hyperparameters
    config = ModernBertConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
    )

    # 3. Initialize ModernBERT from scratch
    model = ModernBertForMaskedLM(config)
   
    # 4. Load the dataset for training (will load shards as needed)
    tokenized_dataset = Dataset.load_from_disk('brwac/brwac_tokenized_dataset')
    
    # 5. Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # 6. Training arguments
    training_args = TrainingArguments(
        output_dir="./modernbert-br",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,
        dataloader_num_workers=2,  # Reduce number of workers
        dataloader_pin_memory=False,  # Reduce memory usage
    )

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # 8. Train the model
    trainer.train()
    
if __name__ == "__main__":
    train()