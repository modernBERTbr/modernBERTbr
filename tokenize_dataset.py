import os
from datasets import Dataset
from transformers import AutoTokenizer
import gc
import numpy as np

def tokenize_dataset(brwac_file="brwac/brwac_plain.txt", output_dir="brwac/brwac_tokenized_dataset"):
    tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    
    chunk_size = 500000  

    def tokenize_function(examples):
        result = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
        # Convert to int16 to save memory
        for key in result:
            if isinstance(result[key][0], list):  # Ensure it's a list of token IDs
                result[key] = [np.array(ids, dtype=np.int16).tolist() for ids in result[key]]
        return result
    
    # Prepare output directories
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)

    total_processed = 0
    chunk_idx = 0
    
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

            chunk_data = Dataset.from_dict({"text": lines})
            tokenized_chunk = chunk_data.map(
                tokenize_function, 
                batched=True, 
                remove_columns=["text"],
                num_proc=max(1, os.cpu_count() // 2),
            )

            tokenized_chunk.save_to_disk(shard_path)
            total_processed += len(lines)

            print(f"Saved chunk {chunk_idx} with {len(lines)} examples. Total processed: {total_processed}")

            chunk_data = None
            tokenized_chunk = None
            lines = None
            gc.collect() 

            chunk_idx += 1

    print(f"Finished processing. Total examples: {total_processed}")
    print(f"Dataset saved to {output_dir}")

if __name__ == "__main__":
    tokenize_dataset()
