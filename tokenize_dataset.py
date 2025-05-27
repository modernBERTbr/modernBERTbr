import os
from datasets import Dataset
from transformers import AutoTokenizer
import gc
import numpy as np
from multiprocessing import Pool, cpu_count

chunk_size = 5000000  
seq_length = 512
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

def tokenize_function(examples):
    result = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=seq_length)
    # Cast to int16, vocab size is 30000
    for key in result:
        if isinstance(result[key][0], list):  
            result[key] = [np.array(ids, dtype=np.int16).tolist() for ids in result[key]]
    return result

def process_lines_chunk(chunk_data):
    """Process a chunk of lines in parallel
    chunk_data: tuple of (chunk_index, lines_chunk)
    """
    chunk_index, lines_chunk = chunk_data
    examples = []
    acc = ""
    acc_tokens = 0
    for line in lines_chunk:
        # Tokenize the line to get token count
        line_tokens = len(tokenizer.encode(line, add_special_tokens=False))
        
        # Check if adding this line would exceed sequence length (accounting for [CLS] and [SEP] tokens)
        if acc_tokens + line_tokens + 2 <= seq_length:  # +2 for [CLS] and [SEP]
            acc += line
            acc_tokens += line_tokens
        else:
            # If current accumulator is not empty, add it to examples
            if acc:
                examples.append(acc)
            # Start new accumulator with current line
            acc = line
            acc_tokens = line_tokens
    
    # Add the last accumulated text if not empty
    if acc:
        examples.append(acc)
        
    return (chunk_index, examples)
    
def tokenize_dataset(brwac_file="brwac/brwac_plain.txt", output_dir="brwac/brwac_tokenized_dataset"):
    # Prepare output directories
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
                
            # Split lines into sub-chunks for parallel processing
            lines_per_process = len(lines) // num_processes
            line_chunks = [(i, lines[i:i + lines_per_process]) for i in range(0, len(lines), lines_per_process)]
            
            # Process chunks in parallel
            with Pool(processes=num_processes) as pool:
                results = pool.map(process_lines_chunk, line_chunks)
            
            # Add examples ordered by chunk index
            examples = []
            for _, result in sorted(results, key=lambda x: x[0]):
                examples.extend(result)

            lines = None
            results = None
            line_chunks = None
            gc.collect() 

            # In the first iteration, chunk_idx is collected by the garbage collector
            if chunk_idx == None:
                chunk_idx = 0

            chunk_data = Dataset.from_dict({"text": examples})
            tokenized_chunk = chunk_data.map(
                tokenize_function, 
                batched=True, 
                remove_columns=["text"],
                num_proc=num_processes // 2,
            )

            tokenized_chunk.save_to_disk(shard_path)
            
            print(f"Saved chunk {chunk_idx} with {len(examples)} examples. Total processed: {total_processed}")

            examples_metrics["total_examples"] += len(examples)

            chunk_data = None
            tokenized_chunk = None
            examples = None
            gc.collect() 

            chunk_idx += 1

    print(f"Finished processing. Total examples: {total_processed}")
    print(f"Dataset saved to {output_dir}")
    print(f"Examples metrics: {examples_metrics}")

if __name__ == "__main__":
    tokenize_dataset()
