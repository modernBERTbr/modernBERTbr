import torch
import time
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers.models.modernbert import ModernBertConfig, ModernBertForMaskedLM
from accelerate import Accelerator

def train():
    accelerator = Accelerator()
    device = accelerator.device
    torch.set_float32_matmul_precision('high')

    tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
       
    config = ModernBertConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        max_position_embeddings=112,
    )

    model = ModernBertForMaskedLM(config)
    model.to(device)

    tokenized_dataset = Dataset.load_from_disk('/dev/shm/brwac_tokenized_dataset_2/train', keep_in_memory=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir="./modernbert-br",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1230,
        gradient_accumulation_steps=1,
        save_steps=5000,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=500,
        learning_rate=1e-3,
        warmup_ratio=0.1,
        dataloader_num_workers=16,
        fp16=False,
        bf16=True,                     
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        dataloader_prefetch_factor=4,   
        do_eval=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("\nStarting training...")
    start_time = time.time()
    trainer.train()
    end_time = time.time()

    total_time = end_time - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds")
    print(f"Average time per epoch: {total_time / training_args.num_train_epochs:.2f} seconds")


if __name__ == "__main__":
    train()
