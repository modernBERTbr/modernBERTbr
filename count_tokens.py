import numpy as np
from transformers import AutoTokenizer
import multiprocessing
import os
from tqdm import tqdm
import time
import gc
import psutil

def process_chunk(chunk_id, lines, tokenizer):
    """Process a chunk of lines and count tokens for each line."""
    start_time = time.time()
    chunk_tokens = []
    for line in lines:
        if not line.strip():
            continue
        tokens = tokenizer.tokenize(line)
        chunk_tokens.append(len(tokens))
    
    lines = None
    gc.collect()
    
    processing_time = time.time() - start_time
    return chunk_id, chunk_tokens, processing_time, len(chunk_tokens)

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024

def count_tokens_per_line(file_path, chunk_size=1000000):
    """Count tokens per line using parallel processing and return statistics."""
    print(f"\n{'='*50}")
    print(f"Starting tokenization of file: {file_path}")
    start_time = time.time()
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    print(f"Loading tokenizer: neuralmind/bert-base-portuguese-cased...")
    tokenizer_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    print(f"Tokenizer loaded in {time.time() - tokenizer_start:.2f} seconds")
    print(f"Memory after loading tokenizer: {get_memory_usage():.2f} MB (Δ: {get_memory_usage() - initial_memory:.2f} MB)")
    
    print(f"Counting lines in file: {file_path}")
    line_count = 50000000
    print(f"Total lines in file: {line_count}")
    print(f"Memory after counting lines: {get_memory_usage():.2f} MB")
    
    num_chunks = (line_count + chunk_size - 1) // chunk_size
    print(f"File will be processed in {num_chunks} chunks of size {chunk_size}")
    
    num_cores = int(multiprocessing.cpu_count())
    print(f"Processing using {num_cores} CPU cores")
    
    tokens_per_line = []
    processing_start = time.time()
    
    chunk_times = []
    chunk_sizes = []
    print(f"Starting parallel processing...")
    
    batch_size = min(100, num_chunks)
    for batch_start in range(0, num_chunks, batch_size):
        batch_end = min(batch_start + batch_size, num_chunks)
        print(f"Processing batch {batch_start//batch_size + 1}/{(num_chunks + batch_size - 1)//batch_size}: chunks {batch_start}-{batch_end-1}")
        
        chunks = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for _ in range(batch_start * chunk_size):
                next(file, None)
            
            for chunk_idx in range(batch_start, batch_end):
                chunk_lines = []
                for _ in range(chunk_size):
                    line = next(file, None)
                    if line is None:
                        break
                    chunk_lines.append(line)
                if chunk_lines:
                    chunks.append((chunk_idx, chunk_lines))
        
        print(f"Memory after reading batch: {get_memory_usage():.2f} MB")
        
        with multiprocessing.Pool(processes=num_cores) as pool:
            results = []
            for chunk_id, chunk in chunks:
                results.append(pool.apply_async(process_chunk, args=(chunk_id, chunk, tokenizer)))
            
            for result in tqdm(results, desc=f"Processing batch {batch_start//batch_size + 1}"):
                chunk_id, chunk_tokens, proc_time, tokens_processed = result.get()
                tokens_per_line.extend(chunk_tokens)
                chunk_times.append(proc_time)
                chunk_sizes.append(tokens_processed)
                if chunk_id % 10 == 0:
                    print(f"Chunk {chunk_id}/{num_chunks}: Processed {tokens_processed} lines in {proc_time:.2f}s ({tokens_processed/proc_time:.1f} lines/s)")
        
        chunks = None
        results = None
        gc.collect()
        print(f"Memory after processing batch: {get_memory_usage():.2f} MB")
    
    total_processing_time = time.time() - processing_start
    print(f"Parallel processing completed in {total_processing_time:.2f} seconds")
    print(f"Average chunk processing time: {np.mean(chunk_times):.2f} seconds")
    print(f"Total tokens counted: {len(tokens_per_line)}")
    print(f"Current memory usage: {get_memory_usage():.2f} MB")
    
    stats_start = time.time()
    if tokens_per_line:
        stats = {
            'total_lines': len(tokens_per_line),
            'min': min(tokens_per_line),
            'max': max(tokens_per_line),
            'mean': np.mean(tokens_per_line),
            'p95': np.percentile(tokens_per_line, 95),
            'p99': np.percentile(tokens_per_line, 99)
        }
    else:
        stats = {
            'total_lines': 0,
            'min': 0,
            'max': 0,
            'mean': 0,
            'p95': 0,
            'p99': 0
        }
    
    print(f"Statistics calculated in {time.time() - stats_start:.2f} seconds")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    print(f"Final memory usage: {get_memory_usage():.2f} MB (Δ: {get_memory_usage() - initial_memory:.2f} MB)")
    print(f"{'='*50}\n")
    
    gc.collect()
    
    return tokens_per_line, stats

if __name__ == "__main__":
    file_path = "brwac/brwac_plain.txt"
    tokens_per_line, stats = count_tokens_per_line(file_path)
    
    print(f"\nTokenization Results Summary")
    print(f"---------------------------")
    print(f"File: {file_path}")
    print(f"Total lines analyzed: {stats['total_lines']:,}")
    print(f"Total tokens found: {sum(tokens_per_line):,}")
    
    print(f"\nToken Distribution Statistics")
    print(f"---------------------------")
    print(f"Mean tokens per line: {stats['mean']:.2f}")
    print(f"Median tokens per line: {np.median(tokens_per_line):.2f}")
    print(f"Min tokens per line: {stats['min']}")
    print(f"Max tokens per line: {stats['max']}")
    print(f"Standard deviation: {np.std(tokens_per_line):.2f}")
    print(f"90th percentile: {np.percentile(tokens_per_line, 90):.2f}")
    print(f"95th percentile: {stats['p95']:.2f}")
    print(f"99th percentile: {stats['p99']:.2f}")
    
    if stats['max'] - stats['min'] > 10:
        bins = min(20, stats['max'] - stats['min'])
        hist, bin_edges = np.histogram(tokens_per_line, bins=bins)
        print(f"\nToken Distribution Histogram")
        print(f"---------------------------")
        for i in range(len(hist)):
            print(f"{int(bin_edges[i]):4d}-{int(bin_edges[i+1]):4d}: {hist[i]:8,} lines | {'#' * min(50, int(hist[i] * 50 / max(hist)))}")