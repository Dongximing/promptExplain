import logging
import os
import traceback

import pandas as pd
import torch
from languageModel import LanguageModelExplation
import numpy as np
from util import strip_tokenizer_prefix
from captum.attr import (
    FeatureAblation,
    LLMAttribution,
    TextTokenInput,
)
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
    logging.info(f"tokens_list--------------------->{tokens_list}")
    return  input_tokens,real_attr_value_per_token








def infer(prompt, model, tokenizer, component_sentences, logging_ind=None):

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

        logging.info(f"token level is {token}")

        word = explain_result.primary_attributions(attr_method='integrated_gradients', style="detailed",
                                                display_level="word",
                                                decode_method="greedy",
                                                generated_list=[],
                                                scores=outputs.scores,
                                                model_input=prompt,
                                                component_sentences=component_sentences)
        logging.info(f"word level is {word}")

        component = explain_result.primary_attributions(attr_method='integrated_gradients', style="detailed",
                                                        display_level="component",
                                                        decode_method="greedy",
                                                        generated_list=[],
                                                        scores=outputs.scores,
                                                        model_input=prompt,
                                                      component_sentences=component_sentences)
        logging.info(f"component level is {component}")

        perturbation_level_tokens , perturbation_level = perturbation(model,tokenizer,prompt,real_output)

        # logging.info(f"perturbation_level_tokens  is {perturbation_level_tokens}")
        # logging.info(f"perturbation_level is {perturbation_level}")





        if logging_ind:
            logging.info(f"Inference completed for the index: {logging_ind}")
        torch.cuda.empty_cache()

        return token, word, component, real_output
    
    except:
        logging.error(traceback.format_exc())
        return None, None, None, None


def run_initial_inference(df, model, tokenizer):
    logging.info(f"Inferencing Initial samples -------------------------")

    data = []
    for ind, example in enumerate(df.select(range(len(df)))):
        token, word, component, real_output = infer(example['sentence'],
                                                    model,
                                                    tokenizer,
                                                    example['component_range'],
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


