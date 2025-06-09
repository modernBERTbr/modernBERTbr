from transformers import AutoTokenizer, pipeline
from transformers.models.modernbert import ModernBertForMaskedLM
from transformers.models.bert import BertForMaskedLM
from datasets import load_dataset
import torch
import numpy as np
from tqdm import tqdm

def evaluate_mlm_accuracy(model, tokenizer, samples=1200):
    """Evaluate masked language modeling accuracy"""
    
    sentences = load_dataset('Helsinki-NLP/news_commentary','en-pt', split='train')['translation']
    sentences = [item['pt'] for item in sentences]
    correct = 0
    total = 0
    
    cont=0
    for sentence in tqdm(sentences[:samples], desc="Evaluating MLM"):
        tokens = tokenizer.tokenize(sentence)
        if len(tokens) <= 2:
            continue
            
        if len(tokens) > 110:
            continue

        cont+=1
        mask_pos = np.random.randint(1, len(tokens) - 1)
        original_word = tokens[mask_pos]
        
        tokens[mask_pos] = "[MASK]"
        masked_sentence = tokenizer.convert_tokens_to_string(tokens)
        
        with torch.no_grad():
            result = pipeline('fill-mask', model=model, tokenizer=tokenizer, top_k=1)(masked_sentence)
            if isinstance(result, list):
                result = result[0]
            predicted_token = result['token_str']
        
        if predicted_token.lower().strip() in original_word.lower():
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, cont

def test():
    tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')

    modern_bert = ModernBertForMaskedLM.from_pretrained('./modernbert-br/checkpoint-60000')
    bert = BertForMaskedLM.from_pretrained('./bert-br/checkpoint-75000')
    bertimbau = BertForMaskedLM.from_pretrained('neuralmind/bert-base-portuguese-cased')

    print("\nEvaluating model performance...")
    mb_mlm_accuracy, mb_mlm_samples = evaluate_mlm_accuracy(modern_bert, tokenizer)
    bert_mlm_accuracy, bert_mlm_samples = evaluate_mlm_accuracy(bert, tokenizer)
    bertimbau_mlm_accuracy, bertimbau_mlm_samples = evaluate_mlm_accuracy(bertimbau, tokenizer)

    print(f"Modern BERT MLM Accuracy: {mb_mlm_accuracy:.4f} with {mb_mlm_samples} samples")
    print(f"BERT MLM Accuracy: {bert_mlm_accuracy:.4f} with {bert_mlm_samples} samples")
    print(f"BERTimbau MLM Accuracy: {bertimbau_mlm_accuracy:.4f} with {bertimbau_mlm_samples} samples")

if __name__ == "__main__":
    test()
