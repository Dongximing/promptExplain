import logging
import os
import traceback

import pandas as pd
import torch
from languageModel import LanguageModelExplation
import numpy as np

from util import strip_tokenizer_prefix
from nltk.tokenize import word_tokenize
from captum.attr import (
    FeatureAblation,
    LLMAttribution,
    TextTokenInput,
)


def calculate_word_scores( model_input, data):
    """
    Calculate scores for each word in the input sentence.

    Parameters:
    - model_input: A string containing the sentence to be processed.
    - data: A dictionary with a 'tokens' key, containing the contributions data.

    Returns:
    - A list of dictionaries with scored tokens.
    """
    words = word_tokenize(model_input)
    contributions = data.get('tokens')
    index = 0
    total_value = 0
    real_token = ''
    combined_contributions = []

    for id, word in enumerate(words, start=0):
        while index < len(contributions):
            token = contributions[index].get('token')
            total_value += float(contributions[index].get('value'))
            real_token += token
            index += 1

            if len(real_token) == len(word):
                combined_contributions.append({
                    'token': real_token,
                    'type': 'input',
                    'value': total_value,
                    'position': id
                })
                total_value = 0
                real_token = ''
                break

    return {'tokens': combined_contributions}
def perturbation(model,tokenizer,prompt,real_output):
    fa = FeatureAblation(model)
    llm_attr = LLMAttribution(fa, tokenizer)
    inp = TextTokenInput(
        prompt,
        tokenizer,
        skip_tokens=[1],  # skip the special token for the start of the text <s>
    )
    logging.info(f"real output ===========>{real_output[0]}")
    attr_res = llm_attr.attribute(inp, target=real_output[0])
    attr_value = attr_res.token_attr.cpu().detach().numpy()
    real_attr_value = np.absolute(attr_value)
    input_tokens = attr_res.input_tokens
    real_attr_value_per_token = np.sum(real_attr_value, axis=0)
    newer_sum_normalized_array = real_attr_value_per_token / np.sum(real_attr_value_per_token)
    tokens_list = []
    for idx, token in enumerate(input_tokens):
        _clean_token = strip_tokenizer_prefix(token)
        #_type = 'output' if idx in generated_list else 'input'
        tokens_list.append({'token': _clean_token,
                            'type': 'input',
                            'value': str(newer_sum_normalized_array[idx]),
                            'position': idx
                            })
    #logging.info(f"tokens_list--------------------->{tokens_list}")
    return {
        'tokens': tokens_list,
    }


def calculate_component_scores(scored_tokens, component_positions_dict):
    # Convert token positions to a dictionary for faster lookup
    position_value_map = {token['position']: token for token in scored_tokens}

    # Initialize dictionaries for token and score storage
    components_tokens_dict = {}
    combined_scores_by_component = {}
    combined_scores_by_word = {}

    # Compute scores and tokens for each component
    for component, positions in component_positions_dict.items():
        component_tokens = [position_value_map.get(position) for position in positions if
                            position in position_value_map]
        components_tokens_dict[component] = component_tokens
        combined_scores_by_component[component] = sum(
            float(token['value']) for token in component_tokens if token is not None)
        combined_scores_by_word[component] = sum(
            float(position_value_map.get(position, {}).get('value', 0)) for position in positions)

    return combined_scores_by_component, combined_scores_by_word, components_tokens_dict
def calculate_component(component_sentences,word_scores):
    component_positions_dict =component_sentences

    combined_scores_by_component, combined_scores_by_word, components_tokens_dict = calculate_component_scores(
    word_scores,
    component_positions_dict)

    return_data = {'combined_scores_by_component': combined_scores_by_component,
               'combined_scores_by_word': combined_scores_by_word,
               'component_tokens_dict': components_tokens_dict}
    return return_data




def infer(prompt, model, tokenizer, component_sentences, is_perturbation,logging_ind=None,):

    # skips the instance if GPU memory exceeds.

    try:
        input_tokenized_info = tokenizer(prompt, return_tensors="pt")
        input_ids, attention_mask = input_tokenized_info['input_ids'], input_tokenized_info['attention_mask']
        input_ids = input_ids.to(os.environ.get("CUDA_CORE"))
        original_prompt = len(input_ids[0])
        attention_mask = attention_mask.to(os.environ.get("CUDA_CORE"))
        outputs = model.generate(
            input_ids,
            max_new_tokens=10,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
        real_output = tokenizer.batch_decode(outputs[0][:, original_prompt:].detach().cpu().numpy(),
                                            skip_special_tokens=True)

        torch.cuda.max_memory_allocated()

        if not is_perturbation:


            explain = LanguageModelExplation(model=model, tokenizer=tokenizer,
                                            model_name=os.environ.get("MODEL_NAME"),
                                            config=None,
                                            collect_activations_flag=False,
                                            collect_activations_layer_nums=None,  # None --> collect for all layers
                                            verbose=False,
                                            gpu=True)
            explain_result = explain.analysis(input_id=input_ids, output=outputs, attention_mask=attention_mask,
                                            attribution=['integrated_gradients'])

            token = explain_result.primary_attributions(attr_method='integrated_gradients', style="detailed",
                                                        display_level="char",
                                                        decode_method="greedy",
                                                        generated_list=[],
                                                        scores=outputs.scores,
                                                        model_input=prompt,
                                                        component_sentences=component_sentences)

            # logging.info(f"token level is {token}")

            word = explain_result.primary_attributions(attr_method='integrated_gradients', style="detailed",
                                                    display_level="word",
                                                    decode_method="greedy",
                                                    generated_list=[],
                                                    scores=outputs.scores,
                                                    model_input=prompt,
                                                    component_sentences=component_sentences)

            # logging.info("\n")
            # logging.info(f"word level is {word}")

            component = explain_result.primary_attributions(attr_method='integrated_gradients', style="detailed",
                                                            display_level="component",
                                                            decode_method="greedy",
                                                            generated_list=[],
                                                            scores=outputs.scores,
                                                            model_input=prompt,
                                                          component_sentences=component_sentences)

        else:
            logging.info("--------------------------perturbation-------------------------------")

            perturbation_result = perturbation(model,tokenizer,prompt,real_output)
            word_perturbation_result = calculate_word_scores(prompt,perturbation_result)
            word_perturbation = word_perturbation_result.get('tokens')
            component_level_perturbation = calculate_component(component_sentences,word_perturbation)
            max_memory_used = torch.cuda.max_memory_allocated() / 1024 ** 2  # Convert to MB
            logging.info(f"Maximum GPU memory used during this batch: {max_memory_used:.2f} MB")
            # logging.info("\n")
            # logging.info(f"token level perturbation_result is {perturbation_result}")
            # logging.info("\n")
            # logging.info(f"word level perturbation_result is {word_perturbation_result}")
            # logging.info("\n")
            # logging.info(f"component level perturbation_result is {component_level_perturbation}")


        # logging.info(f"perturbation_level_tokens  is {perturbation_level_tokens}")
        # logging.info(f"perturbation_level is {perturbation_level}")





        if logging_ind:
            logging.info(f"Inference completed for the index: {logging_ind}")
        torch.cuda.empty_cache()

        if is_perturbation:

            return perturbation_result, word_perturbation_result, component_level_perturbation, real_output
        else:

            return token, word, component, real_output
    
    except:
        logging.error(traceback.format_exc())
        return None, None, None, None


def run_initial_inference(df, model, tokenizer):
    logging.info(f"Inferencing Initial samples -------------------------")
    is_perturbation = True

    data = []
    for ind, example in enumerate(df.select(range(len(df)))):
        token, word, component, real_output = infer(example['sentence'],
                                                    model,
                                                    tokenizer,
                                                    example['component_range'],
                                                    is_perturbation,
                                                    logging_ind=ind)
        
        if token is not None:
            data.append(
                {'prompt': example['sentence'], "real_output": real_output, "token_level": token, "word_level": word,
                "label": example['label'],
                "component_level": component,
                'instruction': example['instruction'],
                'query': example['query'],
                "component_range": example['component_range']  # TODO: not a list, its a dict
                }
            )
        else:
            logging.info(f"Skipping index {ind} due to inference error.")
    result = pd.DataFrame(data)
    return result

def run_peturbed_inference(df, model, tokenizer, results_path, column_names=None):
    """depreciated since this func does not handle failure
    demnarkated by the return None from infer func. 
    """
    logging.info(f"Inferencing Peturbed samples -------------------------")

    if column_names is None:
        # getting the columns demarkated by `reconstructed`
        column_names = []
        for col in df.columns:
            if '_reconstructed_' in col:
                # TODO: asper Gopi's latest reuquirement, we run only for 0,2 run
                if "0.2" in col:
                    column_names.append(col)

    print("running inference on petrubation columns:", column_names)

    for col_name in column_names:
        results = df.apply(lambda row: infer(row[col_name], model, tokenizer,
                                             component_sentences=row['component_range'],
                                             is_perturbation = True,
                                             logging_ind=f"{row.name=},{col_name=}")
                                             , axis=1)
        df[f'{col_name}_token_level'], \
            df[f'{col_name}_word_level'], \
            df[f'{col_name}_component_level'], \
            df[f'{col_name}_output'] = zip(*results)

        # save after specific cols,
        # if fails then remove the cols in the `column_names` arg
        df.to_pickle(results_path + "_intermediate-run_peturbed_inference.pkl")
        torch.cuda.empty_cache()

    return df



def erroneous_run_peturbed_inference(df, model, tokenizer, results_path, column_names=None, save_every_n=10):
    logging.info(f"Inferencing Perturbed samples -------------------------")

    if column_names is None:
        column_names = [col for col in df.columns if '_reconstructed_' in col]

    for col_name in column_names:
        successful_results = []
        successful_indices = []
        
        for index, row in df.iterrows():  
            result = df.apply(infer(row[col_name], model, tokenizer,
                            component_sentences=row['component_range'],
                            logging_ind=f"{row.name=},{col_name=}")
                            , axis=1)

            if None not in result:
                successful_results.append(result)
                successful_indices.append(row.Index)
            else:
                logging.warning(f"Unsuccessful inference for row {row.Index} in column '{col_name}'. Skipping this entry.")
            
            # Save the DataFrame every Nth row or at the end of the DataFrame
            if index % save_every_n == 0 or index == len(df):
                current_results = pd.DataFrame(successful_results, index=successful_indices)
                for c, res_col in enumerate(['token_level', 'word_level', 'component_level', 'output'], start=1):
                    df.loc[successful_indices, f'{col_name}_{res_col}'] = current_results[c]

                pickle_path = os.path.join(results_path, f'partial_run_peturbed_inference.pkl')
                df.to_pickle(pickle_path)
                logging.info(f"Saved partial results to {pickle_path}")

                successful_results.clear()  # Clearing the list for the next batch
                successful_indices.clear()

            torch.cuda.empty_cache()

    final_save_path = os.path.join(results_path, 'final_result.pkl')
    df.to_pickle(final_save_path)  # Saving the final complete DataFrame
    logging.info(f"Final results saved to {final_save_path}")

    return df


