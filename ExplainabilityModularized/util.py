import os
import re

import numpy as np
import torch
import yaml
from torch.nn import functional as F


def check_beam_search(score, index):
    # find the top value in beam search method.
    softmax_array = F.softmax(score, dim=1)
    best = torch.argmax(softmax_array[:, index])
    value = score[best, index].item()
    return value


def expand_array_with_indices(initial_array, insert_indices):
    """
    reconvert the contribution back to its original contribution.
    """

    # Calculate the number of columns in the final array
    insert_indices = [item for sublist in insert_indices for item in sublist]
    final_col_num = initial_array.shape[1] + len(insert_indices)

    # Create a new array filled with 0s
    final_array = np.zeros((initial_array.shape[0], final_col_num))

    # Initiate the indices for the original and final arrays
    initial_idx = 0
    final_idx = 0

    for idx in range(final_col_num):
        if idx in insert_indices:
            # If the index is in the insert_indices list, skip it (leave as 0)
            final_idx += 1
        else:
            # Else, copy the column from the initial array to the final array
            final_array[:, final_idx] = initial_array[:, initial_idx]
            final_idx += 1
            initial_idx += 1

    return final_array


def flatten_list(list_of_lists):
    """
    flattens a list of lists
    e.g., [[1,2,3][4,5,6]] => [1,2,3,4,5,6]
    """
    return [item for sublist in list_of_lists for item in sublist]


def remove_output_index(total_array, romve_list):
    """
    Gets contribution except model output
    """
    cols_to_remove = flatten_list(romve_list)
    mask = np.ones(total_array.shape[1], dtype=bool)
    mask[cols_to_remove] = False
    new_array = total_array[:, mask]
    return new_array


def softmax_greedy(array):
    softmax_array = F.softmax(array, dim=1)
    max_value = torch.max(softmax_array, dim=1).values.item()
    return max_value


def softmax_2d_sample(array, index):
    softmax_array = F.softmax(array, dim=1)
    sample_value = (softmax_array[0][index]).item()
    return sample_value


def normal(be_process_nom, tracking_list):
    new_cont = []
    count = 0
    while count < len(be_process_nom):
        if find_element(tracking_list, count) > 0:
            end = find_element(tracking_list, count)
            new_cont.append(float(sum(be_process_nom[count:count + end])))
            count += end
        else:
            new_cont.append(float(be_process_nom[count]))
            count += 1
    return new_cont


def normalize_array(arr):
    total_sum = np.sum(arr)
    normalized_arr = arr / total_sum
    return normalized_arr


def find_element(my_list, check_value):
    # the first method with greed_method
    for i, element in enumerate(my_list):
        if isinstance(element, list):
            if check_value in element:
                return len(element)
        elif element == check_value:
            return 0


# CHeck if running from inside jupyter
# From https://stackoverflow.com/questions/47211324/check-if-module-is-running-in-jupyter-or-not
def is_punctuation_or_digit(character):
    # Check if the character is a punctuation or a digit using regular expressions
    return bool(re.match(r'\d|[^\w\s]', character))


def is_digit(character):
    # Check if the character is a digit using regular expressions
    return bool(re.match(r'\d', character))


def is_punctuation(character):
    # Check if the character is a punctuation mark using regular expressions
    return bool(re.match(r'[^\w\s]', character))


def load_config(model_name):
    path = os.path.dirname(__file__)
    configs = yaml.safe_load(open(os.path.join(path, "model-config.yaml"),
                                  encoding="utf8"))
    try:
        model_config = configs[model_name]
        model_config = pack_tokenizer_config(model_config)
    except KeyError:
        raise ValueError(
            f"The model '{model_name}' is not defined in Ecco's 'model-config.yaml' file and"
            f" so is not explicitly supported yet. Supported models are:",
            list(configs.keys())) from KeyError()
    return model_config


def pack_tokenizer_config(model_config):
    """
    Convenience method to package tokenizer configs into one element to more easily pass it to
    JavaScript rendering code.
    Args:
        model_config: dict of model configuration options used for model-config or in __init__.py

    Returns:
        model_config dict with 'tokenizer_config' elements
    """
    tokenizer_config = {'token_prefix': model_config['token_prefix'],
                        'partial_token_prefix': model_config['partial_token_prefix']}
    model_config['tokenizer_config'] = tokenizer_config
    return model_config


def strip_tokenizer_prefix(token):
    token = token.lstrip('▁')
    token = token.lstrip('')
    token = token.lstrip(' ')
    return token


def is_partial_token(model_config,
                     token):
    if (token[0: len('')] == '') and \
            ((len('▁') == 0) or \
             token[0:len('▁')] != '▁'):
        return True
    else:
        return False
