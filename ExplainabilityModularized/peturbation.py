import copy
import random
import torch
import os
from nltk.corpus import wordnet as wn
from nltk.corpus import words
from sentence_transformers import SentenceTransformer, util
from model_setup import take_down_model, setup_model_tokenizer
from utils import find_gpu_with_memory_threshold
import logging


# Load the set of English words from NLTK
english_words = set(words.words())


def get_non_synonyms(word):
    """Fetch non-synonyms for a given word."""
    synonyms = {lemma.name() for syn in wn.synsets(word) for lemma in syn.lemmas()}
    non_synonyms = list(english_words - synonyms - {word})
    return non_synonyms if non_synonyms else [word]


def check_similarity(model_s, text1, text2, threshold=0.7):
    """Check if semantic similarity between two texts is below a threshold."""
    embedding1 = model_s.encode(text1, convert_to_tensor=True)
    embedding2 = model_s.encode(text2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
    return similarity < threshold


def modify_tokens(tokens, model_s, is_top, threshold=0.7, max_attempts=5, peturbation_level=0.4):
    original_text = ' '.join([t['token'] for t in tokens])
    tokens_sorted_by_value = sorted(tokens, key=lambda x: x['value'], reverse=True)

    # Calculate the slice index for top or bottom 40%
    num_tokens = len(tokens)
    slice_index = int(num_tokens * peturbation_level)

    # Determine tokens to be modified based on is_top flag
    if is_top:
        tokens_to_modify = tokens_sorted_by_value[:slice_index]
    else:
        tokens_to_modify = tokens_sorted_by_value[-slice_index:]

    # Keep the untouched segment as is
    untouched_tokens = tokens_sorted_by_value[slice_index:] if is_top else tokens_sorted_by_value[:-slice_index]

    for attempt in range(max_attempts):
        # Randomly replace tokens in the selected segment
        for token_info in tokens_to_modify:
            non_synonyms = get_non_synonyms(token_info['token'])
            token_info['token'] = random.choice(non_synonyms) if non_synonyms else token_info['token']

        # Reconstruct the text to check similarity
        modified_tokens = (tokens_to_modify + untouched_tokens) if is_top else (untouched_tokens + tokens_to_modify)
        modified_tokens = sorted(modified_tokens, key=lambda x: x['position'])  # Sort by original position
        modified_text = ' '.join([t['token'] for t in modified_tokens])

        # Check if modified text meets similarity threshold
        if check_similarity(model_s, original_text, modified_text, threshold):
            return modified_tokens  # Accept modifications if similarity is below threshold

    # If unable to meet threshold after attempts, enforce random changes
    for token_info in tokens_to_modify:
        token_info['token'] = random.choice(list(english_words))  # Force change with a random word

    return sorted(tokens_to_modify + untouched_tokens, key=lambda x: x['position'])

def take_down_sentence_and_put_LLMmodel_back_up(model_s):
    del model_s
    torch.cuda.empty_cache()
    _, llm_model = setup_model_tokenizer()
    return llm_model

def run_peturbation(df, llm_model):

    # NOTE: this loads on GPU by taking ~360-450MB space (which may be occupied) 
    available_gpu = find_gpu_with_memory_threshold(0.4) # 400MB
    logging.info(f"I found that GPU {available_gpu=} has 400MB to load the model")
    print(f'{available_gpu=}')

    if available_gpu == -1: # none available then have to take down model on current GPU
        take_down_model(llm_model)
        torch.cuda.set_device(torch.device(os.environ.get('CUDA_CORE', 'cpu')))
    else:
        torch.cuda.set_device(torch.device(available_gpu))

    # Load the sentence model on GPU
    model_s = SentenceTransformer('all-MiniLM-L6-v2')

    # Make a deep copy of required columns to avoid modifying the original DataFrame.
    _df = copy.deepcopy(df[['instructions_tokens', 'query_tokens']])

    for pct in [0.2]: #[0.1, 0.2, 0.3, 0.4]:
        df[f'instruction_token_top_{pct}_peturbed'] = _df['instructions_tokens'].apply(
            lambda lst: modify_tokens(copy.deepcopy(lst), model_s, is_top=True, peturbation_level=pct)
        )
        df[f'instruction_token_bottom_{pct}_peturbed'] = _df['instructions_tokens'].apply(
            lambda lst: modify_tokens(copy.deepcopy(lst), model_s, is_top=False, peturbation_level=pct)
        )
        df[f'query_token_top_{pct}_peturbed'] = _df['query_tokens'].apply(
            lambda lst: modify_tokens(copy.deepcopy(lst), model_s, is_top=True, peturbation_level=pct)
        )
        df[f'query_token_bottom_{pct}_peturbed'] = _df['query_tokens'].apply(
            lambda lst: modify_tokens(copy.deepcopy(lst), model_s, is_top=False, peturbation_level=pct)
        )

    # put the llm model back up if it had been taken down earlier
    if available_gpu == -1:
        llm_model = take_down_sentence_and_put_LLMmodel_back_up(model_s)
    
    return df, llm_model


