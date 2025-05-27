import re
import numpy as np

def count_tokens_per_line(file_path):
    """Count tokens per line and return statistics about their distribution."""
    tokens_per_line = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if not line.strip():
                continue
                
            # Split into tokens (words and punctuation)
            tokens = re.findall(r'\w+|[^\w\s]', line)
            tokens_per_line.append(len(tokens))
    
    # Calculate statistics
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
    
    return tokens_per_line, stats

if __name__ == "__main__":
    file_path = "brwac/brwac_plain.txt"
    tokens_per_line, stats = count_tokens_per_line(file_path)
    
    print(f"File: {file_path}")
    print(f"Total lines analyzed: {stats['total_lines']}")
    print(f"Mean tokens per line: {stats['mean']:.2f}")
    print(f"Min tokens per line: {stats['min']}")
    print(f"Max tokens per line: {stats['max']}")
    print(f"95th percentile: {stats['p95']:.2f}")
    print(f"99th percentile: {stats['p99']:.2f}")