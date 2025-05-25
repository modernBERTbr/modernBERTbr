from multiprocessing import Pool, cpu_count
from bs4 import BeautifulSoup
from ftfy import fix_text

# Function to process one sentence
def process_sentences(sentences):
    text = ' '.join(sentences)
    text = BeautifulSoup(text, "html.parser").get_text()
    return fix_text(text)

# Generator that yields one sentence at a time
def sentence_generator(filepath):
    sentence = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            columns = line.split('	')
            if not columns or len(columns) != 10:
                continue

            if columns[0] == '1' and len(sentence) > 0:
                yield sentence
                sentence = [columns[1]]
            else:
                sentence.append(columns[1])

        if sentence:
            yield sentence 

def parallel_process_dataset(input_path, output_path, batch_size=10000000):
    num_workers = cpu_count()
    print(f"Using {num_workers} worker processes.")

    with open(output_path, 'w', encoding='utf-8') as outfile, \
         Pool(processes=num_workers) as pool:

        batch = []
        for sentence in sentence_generator(input_path):
            batch.append(sentence)
            if len(batch) == batch_size:
                results = pool.map(process_sentences, batch)
                outfile.write('\n'.join(results) + '\n')
                batch = []

        if batch:
            results = pool.map(process_sentences, batch)
            outfile.write('\n'.join(results) + '\n')
        
if __name__ == '__main__':
    parallel_process_dataset('brwac/brwac.conll', 'brwac/brwac_plain.txt')
