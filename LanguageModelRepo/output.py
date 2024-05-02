import json
import os
import random
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython import display as d
from nltk.tokenize import word_tokenize

from util import strip_tokenizer_prefix, expand_array_with_indices, flatten_list, remove_output_index, \
    softmax_greedy, \
    softmax_2d_sample, normalize_array, check_beam_search

mapping = {
    "S": "##Safety",
    "Person": "##Persona",
    "T": "##Tone",
    "Inst": "##Instructions",
    "Response": "##Response Grounding",
    "Ex": "##Examples",
    "K": "##Knowledge",
    "J": "##Jailbreaks",
    "Custom": "##Custom Context",
    "Output": "##Output Format"
}


def computing(values, real_contribution):
    """
    computes the contribution
    """
    real_contribution = torch.from_numpy(np.array(real_contribution))
    values = torch.from_numpy(np.array(values))
    values = torch.unsqueeze(values, 0)
    result = torch.mm(values, real_contribution)
    return result


def greedy_method(scores, real_contribution):
    values = []
    for i in range(len(scores)):
        value = softmax_greedy(scores[i])
        values.append(value)
    result = computing(values, real_contribution)
    return result


def sampling_method(scores, real_contribution, indexs):
    values = []
    for i in range(len(scores)):
        value = softmax_2d_sample(scores[i], indexs[i])  # find the weight of generated tokens
        values.append(value)
    result = computing(values, real_contribution)
    return result


def beamSearch_method(scores, real_contribution, output_index):
    values = []
    assert len(scores) == len(output_index)
    for i in range(len(output_index)):
        score = scores[i]
        result = check_beam_search(score, output_index[i])  # find the weight of generated tokens
        values.append(result)
    result = computing(values, real_contribution)
    return result


def contribution_preprocess(attribution, n_input_tokens, generated_list, scores, token_ids):
    """remove the output contribution, only consider input text contribution"""

    real_contribution = [att.tolist()[:n_input_tokens] for att in attribution]
    real_contribution = np.array(real_contribution)
    index = token_ids[0][n_input_tokens:]  # check the last one is nan,if it is nan, it will move nan value
    scores = scores[:len(index)]
    if np.any(np.isnan(real_contribution[-1][:])) == True:
        real_contribution = real_contribution[:-1][:]
        scores = scores[:-1]
        index = token_ids[0][n_input_tokens:-1]

    assert len(scores) == len(index)
    """check if that has any output, if have remove it, such as p1 p2 p3 p4 o1
        o2 o3 p5 p6 in the next step, it will remove o1 , o2 , o3 """
    if len(generated_list) > 0:
        real_contribution = remove_output_index(real_contribution, generated_list)

    real_contribution = normalize_array(real_contribution)
    return real_contribution, scores, index


def final_contribution(normalization, generated_list):
    """
    After find the contribution, then it will recover to original display such as
    con_1 con_2 con_3 con_4 con_5 con_6 to # con_1 con_2 con_3 o1(0) o2(0) o3(0) con_4 con_5 con_6
    """
    normalization = normalization.numpy()
    normalization = normalize_array(normalization)
    reconvert_contribution = expand_array_with_indices(normalization, generated_list)
    return reconvert_contribution


def hg_contribution_method(config, tokens, tokenizer, token_ids, attribution, n_input_tokens,
                           scores, decode_method, generated_list):
    """
    This generates the contribution of each token based on huggingface output result

    Args:
        decode_method: one of `beam`, `greedy` or `sampling`
        n_input_tokens: the number of orginal input tokens in the prompt
        config: model config
        tokenizer: model tokenizer
        token_ids: id associated with each token
        tokens: generated tokens (different than the input tokens)
        attribution: Attribution for each token
        generated_list: index of model real output
        scores: model output score
    """

    real_contribution, scores, index = contribution_preprocess(attribution, n_input_tokens, generated_list, scores,
                                                               token_ids)

    if decode_method == 'greedy':
        normalization = greedy_method(scores, real_contribution)
    elif decode_method == 'beam':
        normalization = beamSearch_method(scores, real_contribution, index)
    elif decode_method == 'sampling':
        normalization = sampling_method(scores, real_contribution, index)
    else:
        raise ValueError(f'Invalid {decode_method=}')

    reconvert_contribution = final_contribution(normalization, generated_list)

    tokens_list = []
    generated_list = flatten_list(generated_list)
    for idx, token in enumerate(tokens[0][:n_input_tokens]):
        token_id = token_ids[0][idx]
        _clean_token = strip_tokenizer_prefix(tokenizer.decode(token_id))
        _type = 'output' if idx in generated_list else 'input'
        tokens_list.append({'token': _clean_token,
                            'type': _type,
                            'value': str(reconvert_contribution.tolist()[0][idx]),
                            'position': idx
                            })
    return {
        'tokens': tokens_list,
        'attributions': reconvert_contribution.tolist()
    }


class OutputSeq:
    """An OutputSeq object is the result of running a language model on some input data. It contains not only the output
    sequence of words generated by the model, but also other data collecting during the generation process
    that is useful to analyze the model.

    In addition to the data, the object has methods to create plots
    and visualizations of that collected data. These include:

    - [layer_predictions()](./#ecco.output.OutputSeq.layer_predictions) <br/>
    Which tokens did the model consider as the best outputs for a specific position in the sequence?
    - [rankings()](./#ecco.output.OutputSeq.rankings) <br/>
    After the model chooses an output token for a specific position, this visual looks back at the ranking
    of this token at each layer of the model when it was generated (layers assign scores to candidate output tokens,
    the higher the "probability" score, the higher the ranking of the token).
    - [rankings_watch()](./#ecco.output.OutputSeq.rankings_watch) <br />
    Shows the rankings of multiple tokens as the model scored them for a single position. For example, if the input is
    "The cat \_\_\_", we use this method to observe how the model ranked the words "is", "are", "was" as candidates
    to fill in the blank.
    - [primary_attributions()](./#ecco.output.OutputSeq.primary_attributions) <br />
    How important was each input token in the selection of calculating the output token?


    To process neuron activations, OutputSeq has methods to reduce the dimensionality and reveal underlying patterns in
    neuron firings. These are:

    - [run_nmf()](./#ecco.output.OutputSeq.run_nmf)


    """

    def __init__(self,
                 token_ids=None,
                 n_input_tokens=None,
                 tokenizer=None,
                 output_text=None,
                 tokens=None,
                 encoder_hidden_states=None,
                 decoder_hidden_states=None,
                 embedding_states=None,
                 attribution=None,
                 activations=None,
                 collect_activations_layer_nums=None,
                 attention=None,
                 model_type: str = 'mlm',
                 lm_head=None,
                 device='cpu',
                 config=None):
        """

        Args:
            token_ids: The input token ids. Dimensions: (batch, position)
            n_input_tokens: Int. The number of input tokens in the sequence.
            tokenizer: huggingface tokenizer associated with the model generating this output
            output_text: The output text generated by the model (if processed with generate())
            tokens: A list of token text. Shorthand to passing the token ids by the tokenizer.
                dimensions are (batch, position)
            hidden_states: A tensor of  dimensions (layer, position, hidden_dimension).
                In layer, index 0 is for embedding hidden_state.
            attribution: A list of attributions. One element per generated token.
                Each element is a list giving a value for tokens from 0 to right before the generated token.
            activations: The activations collected from model processing.
                Shape is (batch, layer, neurons, position)
            collect_activations_layer_nums:
            attention: The attention tensor retrieved from the language model
            model_outputs: Raw return object returned by the model
            lm_head: The trained language model head from a language model projecting a
                hidden state to an output vocabulary associated with teh tokenizer.
            device: "cuda" or "cpu"
            config: The configuration dict of the language model
        """
        self.token_ids = token_ids
        self.tokenizer = tokenizer
        self.n_input_tokens = n_input_tokens
        self.output_text = output_text
        self.tokens = tokens
        self.encoder_hidden_states = encoder_hidden_states
        self.decoder_hidden_states = decoder_hidden_states
        self.embedding_states = embedding_states
        self.attribution = attribution
        self.activations = activations
        self.collect_activations_layer_nums = collect_activations_layer_nums
        self.attention_values = attention
        self.lm_head = lm_head
        self.device = device
        self.config = config
        self.model_type = model_type
        self._path = os.path.dirname(os.path.abspath(__file__))

    def _get_encoder_hidden_states(self):
        return self.encoder_hidden_states if self.encoder_hidden_states is not None else self.decoder_hidden_states

    def _get_hidden_states(self) -> Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        return (self.encoder_hidden_states, self.decoder_hidden_states)

    def __str__(self):
        return "<LMOutput '{}' # of lm outputs: {}>".format(self.output_text, len(self._get_hidden_states()[1][-1]))

    def to(self, tensor: torch.Tensor):
        if self.device == 'cuda':
            return tensor.to('cuda')
        return tensor

    def explorable(self, printJson: Optional[bool] = False):
        """
        Explorable showing primary attributions of each token generation step.
        Hovering-over or tapping an output token imposes a saliency map on other tokens
        showing their importance as features to that prediction.
        """
        tokens = []
        for idx, token in enumerate(self.tokens[0]):
            type = "input" if idx < self.n_input_tokens else 'output'

            tokens.append({'token': token,
                           'token_id': int(self.token_ids[0][idx]),
                           'type': type
                           })

        data = {
            'tokens': tokens
        }

        d.display(d.HTML(filename=os.path.join(self._path, "html", "setup.html")))

        js = f"""
         requirejs(['basic', 'ecco'], function(basic, ecco){{
            const viz_id = basic.init()

            ecco.renderOutputSequence({{
                parentDiv: viz_id,
                data: {data},
                tokenization_config: {json.dumps(self.config['tokenizer_config'])}
            }})
         }}, function (err) {{
            console.log(err);
        }})"""

        if printJson:
            print(data)

    def __call__(self, position=None, **kwargs):

        if position is not None:
            self.position(position, **kwargs)
        else:
            self.primary_attributions(**kwargs)

    def position(self, position, attr_method='grad_x_input'):
        """
        Method is only for visualization
        """
        if (position < self.n_input_tokens) or (position > len(self.tokens) - 1):
            raise ValueError("'position' should indicate a position of a generated token. "
                             "Accepted values for this sequence are between {} and {}."
                             .format(self.n_input_tokens, len(self.tokens) - 1))

        importance_id = position - self.n_input_tokens
        tokens = []

        assert attr_method in self.attribution, \
            f"attr_method={attr_method} not found. Choose one of the following: {list(self.attribution.keys())}"
        attribution = self.attribution[attr_method]
        for idx, token in enumerate(self.tokens):
            type = "input" if idx < self.n_input_tokens else 'output'
            if idx < len(attribution[importance_id]):
                imp = attribution[importance_id][idx]
            else:
                imp = -1

            tokens.append({'token': token,
                           'token_id': int(self.token_ids[idx]),
                           'type': type,
                           'value': str(imp)  # because json complains of floats
                           })

        data = {
            'tokens': tokens
        }

        d.display(d.HTML(filename=os.path.join(self._path, "html", "setup.html")))
        viz_id = 'viz_{}'.format(round(random.random() * 1000000))
        js = """
         requirejs(['basic', 'ecco'], function(basic, ecco){{
            const viz_id = basic.init()
            ecco.renderSeqHighlightPosition(viz_id, {}, {})
         }}, function (err) {{
            console.log(err);
        }})""".format(position, data)
        d.display(d.Javascript(js))

    def calculate_word_scores(self, model_input, data):
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

        return combined_contributions

    def calculate_component_scores(self, scored_tokens, component_positions_dict):
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


        """
            The view shows the Gradient * Inputs method of input saliency. The attribution values are calculated across the
            embedding dimensions, then we use the L2 norm to calculate a score for each token (from the values of its embeddings dimension)
            To get a percentage value, we normalize the scores by dividing by the sum of the attribution scores for all
            the tokens in the sequence.
        """


    def primary_attributions(self,
                             attr_method: Optional[str] = 'grad_x_input',
                             style="no_display",
                             **kwargs):
       

        # Check if all required variables are present in kwargs
        self.validate_kwargs_prevalence(kwargs, required_variables=[
            'model_input', 'decode_method', 'scores'])  # 'generated_list'

        assert attr_method in self.attribution, \
            f"attr_method={attr_method} not found. Choose one of the following: {list(self.attribution.keys())}"

        attribution = self.attribution[attr_method]

        display_level = kwargs.get('display_level', 'char')
        scores = kwargs.get('scores')
        decode_method = kwargs.get('decode_method')
        generated_list = kwargs.get('generated_list', [])
        model_input = kwargs.get("model_input")

        contribution_data = hg_contribution_method(self.config, self.tokens, self.tokenizer, self.token_ids,
                                                   attribution,
                                                   self.n_input_tokens, scores,
                                                   decode_method, generated_list)

        word_scores = self.calculate_word_scores(model_input, contribution_data)

        if display_level == 'char':
            return_data = contribution_data

        elif display_level == "word":
            return_data = {'tokens': word_scores}

        elif display_level == "component":
            self.validate_kwargs_prevalence(kwargs, required_variables=[
                'component_sentences'])
            component_positions_dict = kwargs.get("component_sentences")

            combined_scores_by_component, combined_scores_by_word, components_tokens_dict = self.calculate_component_scores(
                word_scores,
                component_positions_dict)

            return_data = {'combined_scores_by_component': combined_scores_by_component,
                           'combined_scores_by_word': combined_scores_by_word,
                           'component_tokens_dict': components_tokens_dict}

        self.handle_style(return_data, style)
        return return_data

    def validate_kwargs_prevalence(self, kwargs, required_variables):
        for var in required_variables:
            if var not in kwargs:
                raise ValueError(f"Variable '{var}' is required but not provided in kwargs.")

    def handle_style(self, data, style):
        """
        handles either `minimal` or `detailed`
        default `no_display` doesnt define a js
        """
        if (style == "minimal"):
            js = f"""
             requirejs(['basic', 'ecco'], function(basic, ecco){{
                const viz_id = basic.init()
                console.log(viz_id)
                // ecco.interactiveTokens(viz_id, {{}})
                window.ecco[viz_id] = new ecco.MinimalHighlighter({{
                    parentDiv: viz_id,
                    data: {json.dumps(data)},
                    preset: 'viridis',
                    tokenization_config: {json.dumps({'token_prefix': 'Ġ', 'partial_token_prefix': ''})}

             }})

             window.ecco[viz_id].init();
             window.ecco[viz_id].selectFirstToken();

             }}, function (err) {{
                console.log(err);
            }})"""

        elif (style == "detailed"):

            js = f"""
             requirejs(['basic', 'ecco'], function(basic, ecco){{
                const viz_id = basic.init()
                console.log(viz_id)
                window.ecco[viz_id] = ecco.interactiveTokens({{
                    parentDiv: viz_id,
                    data: {json.dumps(data)},
                    tokenization_config: {json.dumps({'token_prefix': 'Ġ', 'partial_token_prefix': ''})}
             }})

             }}, function (err) {{
                console.log(err);
            }})"""

    def _repr_html_(self, **kwargs):
        self.explorable(**kwargs)
        return '<OutputSeq>'

    def plot(self, n_components=3):

        for idx, comp in enumerate(self.components):
            comp = comp[:n_components, :].T

            fig, ax1 = plt.subplots(1)
            plt.subplots_adjust(wspace=.4)
            fig.set_figheight(2)
            fig.set_figwidth(17)
            ax1.plot(comp)
            ax1.set_xticks(range(len(self.tokens)))
            ax1.set_xticklabels(self.tokens, rotation=-90)
            ax1.legend(['Component {}'.format(i + 1) for i in range(n_components)], loc='center left',
                       bbox_to_anchor=(1.01, 0.5))
