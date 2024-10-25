from transformers import AutoModelForMaskedLM, AutoTokenizer, FillMaskPipeline
import pandas as pd 
from tqdm import tqdm
import argparse 
import torch 
import torch.nn.functional as F
from utils.data import CustomDataset
import time 
import os 
cache_dir=os.getenv('CACHE_DIR')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str)
    parser.add_argument('--corpus-age-min', type=int, default=25)
    parser.add_argument('--procedural-prob', type=int, default=0.5)
    parser.add_argument('--tokenizer', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--word-list', type=str)
    parser.add_argument('--word', type=str)
    # parser.add_argument('--control-word-list', type=str)
    parser.add_argument('--device', type=str)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--window', type=int, default=50)
    parser.add_argument('--n-substitutes', type=int, default=10)
    parser.add_argument('--stopwords-path', type=str, default='assets/stopwords.txt')
    parser.add_argument('--output-path', type=str)
    args = parser.parse_args()
    generate_substitution_data(args)

def get_tokenizer(tokenizer_string):
    return AutoTokenizer.from_pretrained(tokenizer_string)

def get_model(model_string):
    return AutoModelForMaskedLM.from_pretrained(model_string)

def get_stopwords(stopwords_filepath):
    stopwords = []
    with open(stopwords_filepath, 'r') as file:
        for line in file:
            stopwords.append(line.strip())
    return stopwords

def is_word_in_tokenized_speech(word_tokenized, speech_tokens):
    if len(speech_tokens)<len(word_tokenized):
        value = False 
    else:
        if len(word_tokenized)==1:
            value = True if word_tokenized[0] in speech_tokens else False 
        else:
            speech_tokens_as_str = str(speech_tokens)
            word_tokens_as_str = str(word_tokenized)
            start_poss = f"{word_tokens_as_str[:-1]} "
            middle_poss = f" {word_tokens_as_str[1:-1]},"
            end_poss = f" {word_tokens_as_str[1:]}"
            value = False
            for poss_string in [start_poss, middle_poss, end_poss]:
                if poss_string in speech_tokens_as_str:
                    value = True
    return value     

def filter_dl(target_word, speech_ids, speeches, tokenizer):
    target_tokenized = tokenizer.encode(target_word, add_special_tokens=False)
    tokenized_speeches = [tokenizer.encode(x) for x in speeches]
    bool_filter = [is_word_in_tokenized_speech(target_tokenized, x) for x in tokenized_speeches]
    if sum(bool_filter)>0:
        speech_ids_filtered = [speech_id for speech_id, boolean in zip(speech_ids, bool_filter) if boolean]
        speeches_filtered = [speech for speech, boolean in zip(speeches, bool_filter) if boolean]
        return speech_ids_filtered, speeches_filtered
    else:
        return

def get_masked_input(target_word, tokenized_speech, tokenizer, window=50):
    if window >= tokenizer.model_max_length / 2:
        print("Context window too large. Make sure the context window is compatible with the model's context window.")
        return
    else:
        tokenized_input = tokenized_speech.copy()
        input_ids = tokenized_input["input_ids"][0]
        token_type_ids = tokenized_input["token_type_ids"][0]
        attention_mask = tokenized_input["attention_mask"][0]
        # Tokenize the target word and get the token IDs, excluding special tokens
        target_token_ids = tokenizer.encode(target_word, add_special_tokens=False)
        # Initialize modified tokenized inputs
        modified_tokenized_inputs = []
        modified_token_type_ids = []
        modified_attention_masks = []
        # Find start indices of all subsequences matching the target token IDs
        target_indices = [i for i in range(len(input_ids) - len(target_token_ids) + 1) if (input_ids[i:i + len(target_token_ids)] == torch.tensor(target_token_ids)).all()]
        for index in target_indices:
            # Replace the sequence with a single mask token
            new_input_ids = torch.cat((input_ids[:index], torch.tensor([tokenizer.mask_token_id]), input_ids[index + len(target_token_ids):]))
            # Calculate the new windowed range after the replacement
            start_index = max(index - window, 0)
            end_index = min(index + window + 1, len(new_input_ids))
            # Extract segments within the new windowed range
            segment_input_ids = new_input_ids[start_index:end_index]
            segment_token_type_ids = token_type_ids[start_index:end_index]
            segment_attention_mask = attention_mask[start_index:end_index]
            # Append segments to the lists
            modified_tokenized_inputs.append(segment_input_ids)
            modified_token_type_ids.append(segment_token_type_ids)
            modified_attention_masks.append(segment_attention_mask)
        # Padding
        max_length = max(tensor.size(0) for tensor in modified_tokenized_inputs)
        padded_inputs = [F.pad(tensor, (0, max_length - tensor.size(0)), "constant", 0) for tensor in modified_tokenized_inputs]
        padded_types = [F.pad(tensor, (0, max_length - tensor.size(0)), "constant", 0) for tensor in modified_token_type_ids]
        padded_attention = [F.pad(tensor, (0, max_length - tensor.size(0)), "constant", 0) for tensor in modified_attention_masks]
        # Replace dict values in tokenized input
        tokenized_input['input_ids'] = torch.stack(padded_inputs)
        tokenized_input['token_type_ids'] = torch.stack(padded_types)
        tokenized_input['attention_mask'] = torch.stack(padded_attention)
        return tokenized_input

def get_invalid_token_ids(tokenizer, stopwords_filepath):
    stopwords = get_stopwords(stopwords_filepath)
    invalid_tok_ids = []
    for token, token_id in tokenizer.vocab.items():
        if len(token) <= 1 or token in stopwords or '##' in token or '...' in token or '[' in token:
            invalid_tok_ids.append(token_id)
    
    return invalid_tok_ids

def get_substitutes(masked_input, model, tokenizer, invalid_tok_ids=[], buffer=100, n=3):
    invalid_tok_ids = torch.tensor(invalid_tok_ids)
    if model.device.type == 'cuda':
        masked_input = {k: v.to(model.device) for k, v in masked_input.items()}
        invalid_tok_ids = invalid_tok_ids.to(model.device)
    
    preds = model(**masked_input).logits
    mask_indices = torch.where(masked_input["input_ids"] == tokenizer.mask_token_id)[1]
    batch_indices = torch.arange(preds.size(0), device=model.device)
    mask_logits = preds[batch_indices, mask_indices]
    log_probs = F.log_softmax(mask_logits, dim=1)
    sorted_preds = log_probs.topk(n+buffer)
    
    speech_valid_pred_tok_ids = []
    speech_valid_log_probs = []
    for tok_ids, log_probs in zip(sorted_preds.indices.detach().cpu(), sorted_preds.values.detach().cpu().numpy()):
        instance_valid_pred_tok_ids = []
        instance_valid_pred_log_probs = [] 
        for tok_id, log_prob in zip(tok_ids, log_probs):
            if len(instance_valid_pred_tok_ids)<n:
                if tok_id not in invalid_tok_ids:
                    instance_valid_pred_tok_ids.append(tok_id)
                    instance_valid_pred_log_probs.append(log_prob)
        speech_valid_pred_tok_ids.append(instance_valid_pred_tok_ids)
        speech_valid_log_probs.append(instance_valid_pred_log_probs)
    
    words = [tokenizer.batch_decode(x) for x in speech_valid_pred_tok_ids]
    return words, speech_valid_log_probs

def get_word_substitution_data(target_word, dataloader, tokenizer, model, window=50, invalid_tok_ids=[], buffer=100, n=10):
    speech_ids_master = []
    # word_filtered_speeches_master = []
    substitutions_master = []
    log_probs_master = []
    for speech_ids, speeches in tqdm(dataloader):
        filtered = filter_dl(target_word, speech_ids, speeches, tokenizer)
        if filtered:
            word_filtered_speech_ids = [int(tensor) for tensor in filtered[0]]
            speech_ids_master+=word_filtered_speech_ids
            word_filtered_speeches = filtered[1]
            # word_filtered_speeches_master += word_filtered_speeches
            tokenized = [tokenizer(x, return_tensors="pt") for x in word_filtered_speeches]
            masked = [get_masked_input(target_word, x, tokenizer, window=window) for x in tokenized]
            results = [get_substitutes(x, model=model, tokenizer=tokenizer, invalid_tok_ids=invalid_tok_ids, buffer=buffer, n=n) for x in masked]
            del(tokenized, masked)
            torch.cuda.empty_cache()
            substitutions = [result[0] for result in results]
            log_probs = [result[1] for result in results]
            substitutions_master += substitutions
            log_probs_master += log_probs
    
    output_df = pd.DataFrame()
    output_df['speech_id'] = speech_ids_master
    # output_df['speech'] = word_filtered_speeches_master
    output_df['substitutions'] = substitutions_master
    output_df['log_probs'] = log_probs_master
    return output_df

def generate_substitution_data(args):
    # Setting up arguments:
    corpus_path = args.corpus 
    tokenizer_str = args.tokenizer 
    model_str = args.model 
    words_path = args.word_list 
    word = args.word
    device = args.device 
    corpus_age_min = args.corpus_age_min 
    procedural_prob = args.procedural_prob
    batch_size = args.batch_size
    window = args.window 
    n_substitutes = args.n_substitutes
    stopwords_path = args.stopwords_path
    output_path = args.output_path
    if words_path and word:
        print("ERROR: Submit either one word, or a list of words, not both!")
    else:
        if words_path:
            # Read in Words:
            words = []
            with open(words_path, 'r') as file:
                for line in file:
                    words.append(line.strip())
        elif word:
            words = [word]
        
        # Import Tokenizer & Model:
        tokenizer = get_tokenizer(tokenizer_str)
        model = get_model(model_str)
        torch.no_grad()
        model.to(device)
        print("Model, tokenizer imported!")
        # Get Token Blacklist Based on Tokenizer and Stopwords:
        invalid_tok_ids = get_invalid_token_ids(tokenizer, stopwords_path)
        # Import Corpus, Filter:
        corpus = pd.read_csv(corpus_path)
        corpus = corpus[corpus['age']>=corpus_age_min] # Any age entries below 25 are going to be errors -- remove them
        corpus = corpus[corpus['procedural_prob']<procedural_prob]
        print("Corpus imported, filtered!")
        # Generate Data Loader:
        corpus_custom = CustomDataset(corpus['speech_id'].to_list(), corpus['speech'].to_list())
        loader = corpus_custom.get_dataloader(batch_size=batch_size)
        # Start on words:
        for word in words:
            print(f"Starting on {word}!")
            output = get_word_substitution_data(word, loader, tokenizer, model, window=window, invalid_tok_ids=invalid_tok_ids, n=n_substitutes)
            output.to_json(os.path.join(output_path, f'{word}.jsonl'), orient='records', lines=True)

if __name__=='__main__':
    main()

