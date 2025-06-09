import os
from datasets import Dataset
from transformers import AutoTokenizer
import gc
from multiprocessing import cpu_count

chunk_size = 2500000  
seq_length = 112
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

def tokenize_function(examples):
    result = tokenizer(
        examples["text"], 
        truncation=True,
        padding=True,
        max_length=seq_length,
    )
    return {"input_ids": result["input_ids"]}
    
def tokenize_dataset(brwac_file="brwac/brwac_plain.txt", output_dir="brwac/brwac_tokenized_dataset_2"):
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)

    total_processed = 0
    chunk_idx = 0    
    num_processes = max(1, cpu_count())
    
    examples_metrics = {
        "total_examples": 0,    
    }
    with open(brwac_file, "r", encoding="utf-8") as f:
        while True:
            shard_path = os.path.join(output_dir, "train", f"{chunk_idx:05d}_of_{chunk_idx+1:05d}")
            if os.path.exists(shard_path):
                print(f"Chunk {chunk_idx} already processed. Skipping...")
                for _ in range(chunk_size):
                    if not f.readline():
                        break
                chunk_idx += 1
                continue

            print(f"Processing chunk {chunk_idx}...")

            lines = []
            for _ in range(chunk_size):
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if line:
                    lines.append(line)

            if not lines:
                break

            total_processed += len(lines)
                
            chunk_data = Dataset.from_dict({"text": lines})
            tokenized_chunk = chunk_data.map(
                tokenize_function, 
                batched=True, 
                remove_columns=["text"],
                num_proc=num_processes,
            )

            tokenized_chunk.save_to_disk(shard_path)
            
            print(f"Saved chunk {chunk_idx} with {len(lines)} examples. Total processed: {total_processed}")

            examples_metrics["total_examples"] += len(lines)

            if total_processed >= 50000000:
                break

            chunk_data = None
            tokenized_chunk = None
            lines = None
            gc.collect() 

            chunk_idx += 1

    print(f"Finished processing. Total examples: {total_processed}")
    print(f"Dataset saved to {output_dir}")
    print(f"Examples metrics: {examples_metrics}")

if __name__ == "__main__":
    tokenize_dataset()
