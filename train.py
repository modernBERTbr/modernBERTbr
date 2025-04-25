from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers.models.modernbert import ModernBertConfig, ModernBertForMaskedLM

def train():
    # 1. Load WordPiece tokenizer from BERTimbau
    tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

    # 2. Create a fresh config for ModernBERT, using BERTimbau's 30.000 vocab size with ModernBERT's paper default hyperparameters
    config = ModernBertConfig(
        vocab_size=tokenizer.vocab_size,
    )

    # 3. Initialize ModernBERT from scratch
    model = ModernBertForMaskedLM(config)

    # 4. TODO: Load brwac dataset
    dataset = None

    # 5. Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 6. Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # TODO: Adjust training arguments based on hardware
    # 7. Training arguments
    training_args = TrainingArguments(
        output_dir="./modernbert-br",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,
    )

    # 8. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # 9. Train the model
    trainer.train()



if __name__ == "__main__":
    train()
