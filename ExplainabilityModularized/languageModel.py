import inspect
import json
import os
import random
import re
from collections import defaultdict
from operator import attrgetter
from typing import Optional, Any, List, Tuple, Dict, Union

import numpy as np
import torch
import transformers
from IPython import display as d
from packaging import version
from torch.nn import functional as F
from transformers import BatchEncoding

from computingAttribution import compute_primary_attributions_scores
from output import OutputSeq
from util import is_partial_token, strip_tokenizer_prefix


class LanguageModelExplation(object):
    def __init__(self,
                 model: transformers.PreTrainedModel,
                 tokenizer: transformers.PreTrainedTokenizerFast,
                 model_name: str,
                 config: Dict[str, Any],
                 collect_activations_flag: Optional[bool] = False,
                 collect_activations_layer_nums: Optional[List[int]] = None,  # None --> collect for all layers
                 verbose: Optional[bool] = False,
                 gpu: Optional[bool] = False
                 ):
        self.model_name = model_name
        self.model = model

        # TODO: @Aadi: we are using other cores, e.g., BMC5, should this be hardcoded like this?
        self.device = 'cuda:1' if torch.cuda.is_available() \
                                  and self.model.device.type == 'cuda' \
            else 'cpu'
        self.tokenizer = tokenizer
        self.verbose = False
        self._path = os.path.dirname(os.path.abspath(__file__))
        # Neuron Activation
        self.collect_activations_flag = collect_activations_flag
        self.collect_activations_layer_nums = collect_activations_layer_nums
        self.model_config = config
        try:
            self.model_type = 'causal'
            embeddings_layer_name = "model.embed_tokens.weight"
            embed_retriever = attrgetter(embeddings_layer_name)
            self.model_embeddings = embed_retriever(self.model)
            self.collect_activations_layer_name_sig = 'down_proj'
        except KeyError:
            raise ValueError(
                f"The model '{self.model_name}' is not correctly configured in Ecco's 'model-config.yaml' file"
            ) from KeyError()

        assert self.model_type in ['causal', 'mlm', 'enc-dec'], f"model type {self.model_type} not found"

        self._reset()

    def _reset(self):
        self._all_activations_dict = defaultdict(dict)
        self.activations = defaultdict(dict)
        self.all_activations = []
        self.generation_activations = []
        self.neurons_to_inhibit = {}
        self.neurons_to_induce = {}
        self._hooks = {}

    def to(self, tensor: Union[torch.Tensor, BatchEncoding]):
        if self.device == 'cuda':
            device = torch.device("cuda")
            return tensor.to(dtype=torch.float16, device=device)
        return tensor

    def _analyze_token(self,
                       encoder_input_embeds: torch.Tensor,
                       encoder_attention_mask,  # TODO: use encoder mask and also decoder mask
                       decoder_input_embeds: Optional[torch.Tensor],
                       prediction_id: torch.Tensor,
                       attribution_flags: Optional[List[str]] = []) -> None:
        """
        This methods computes the primary attribution explainability scores for each given token by analyzing the
        prediction token
        """
        for attr_method in attribution_flags:
            # deactivate hooks: attr method can perform multiple forward steps
            self._remove_hooks()

            # Add attribution scores to self.attributions
            self.attributions[attr_method].append(
                compute_primary_attributions_scores(
                    attr_method=attr_method,
                    model=self.model,
                    forward_kwargs={
                        'inputs_embeds': encoder_input_embeds,
                        'decoder_inputs_embeds': decoder_input_embeds
                    },
                    prediction_id=prediction_id
                ).cpu().detach().numpy()
            )

    def analysis(self, output, input_id, attention_mask, attribution: Optional[List[str]] = []):
        """
        Adopted from: https://github.com/jalammar/ecco/blob/main/src/ecco/lm.py
        Look at `generate` function for more details

        Arguments:
              output {torch.Tensor}: model output
              input_id {torch.Tensor}: input prompt
              attention_mask {torch.Tensor}: attention mask for the input prompt
              attribution {[List[str]]}: List of contribution methods
        """
        n_input_tokens = len(input_id[0])
        cur_len = n_input_tokens
        pad_token_id = self.model.config.pad_token_id
        eos_token_id = self.model.config.eos_token_id
        decoder_input_ids = None

        # Print output
        n_printed_tokens = n_input_tokens
        if self.verbose:
            viz_id = self.display_input_sequence(input_id[0])
        if output.__class__.__name__.endswith("EncoderDecoderOutput"):
            prediction_ids, prediction_scores = output.sequences[0][1:], output.scores

        elif output.__class__.__name__.endswith("DecoderOnlyOutput"):
            prediction_ids, prediction_scores = output.sequences[0][n_input_tokens:], output.scores
        else:
            raise NotImplementedError(f"Unexpected output type: {type(output)}")

        assert prediction_ids != []

        self.attributions = defaultdict(list)  # reset attributions dict
        for pred_index, prediction_id in enumerate(prediction_ids):
            # First get encoder/decoder input embeddings
            encoder_input_embeds, _ = self._get_embeddings(input_id)
            # Should make separate ones for more flexibility
            if decoder_input_ids is not None:
                decoder_input_embeds, _ = self._get_embeddings(decoder_input_ids)
            else:
                decoder_input_embeds = None

            if pred_index == len(
                    prediction_ids) - 1:  # -1 because we want to catch the inputs for the last generated token
                # attach hooks and run last forward step
                # TODO: collect activation for more than 1 step
                self._attach_hooks(self.model)
                extra_forward_kwargs = {'attention_mask': attention_mask, 'decoder_inputs_embeds': decoder_input_embeds}
                forward_kwargs = {
                    'inputs_embeds': encoder_input_embeds,
                    'use_cache': False,
                    'return_dict': True,
                    **{k: v for k, v in extra_forward_kwargs.items() if
                       k in inspect.signature(self.model.forward).parameters}
                }
                _ = self.model(**forward_kwargs)

            # Get primary attributions for produced token
            self._analyze_token(
                encoder_input_embeds=encoder_input_embeds,
                encoder_attention_mask=attention_mask,
                decoder_input_embeds=decoder_input_embeds,
                attribution_flags=attribution,
                prediction_id=prediction_id
            )

            # Recomputing inputs ids, attention mask and decoder input ids
            if decoder_input_ids is not None:
                assert len(decoder_input_ids.size()) == 2  # will break otherwise
                decoder_input_ids = torch.cat(
                    [decoder_input_ids, torch.tensor([[prediction_id]], device=decoder_input_ids.device)],
                    dim=-1
                )
            else:
                input_id = torch.cat(
                    [input_id, torch.tensor([[prediction_id]], device=input_id.device)],
                    dim=-1
                )

                # Recomputing Attention Mask
                if getattr(self.model, '_prepare_attention_mask_for_generation'):
                    assert len(input_id.size()) == 2  # will break otherwise
                    attention_mask = self.model._prepare_attention_mask_for_generation(input_id, pad_token_id,
                                                                                       eos_token_id)
                    attention_mask = self.to(attention_mask)

            offset = n_input_tokens if decoder_input_ids is not None else 0
            generated_token_ids = decoder_input_ids if decoder_input_ids is not None else input_id

            # More than one token can be generated at once (e.g., automatic split/pad tokens)
            while len(generated_token_ids[0]) + offset != n_printed_tokens:

                # Display token
                if self.verbose:
                    self.display_token(
                        viz_id,
                        generated_token_ids[0][n_printed_tokens - offset].cpu().numpy(),
                        cur_len
                    )

                n_printed_tokens += 1

                # Add a zero vector to the attributions vector, if we did not reach the last predicted token
                if len(generated_token_ids[0]) + offset != n_printed_tokens:
                    for k in self.attributions:
                        self.attributions[k].insert(-1, np.zeros_like(self.attributions[k][-1]))

            cur_len += 1

        embedding_states = None
        for attributes in ["hidden_states", "encoder_hidden_states", "decoder_hidden_states"]:
            out_attr = getattr(output, attributes, None)
            if out_attr is not None:

                tokens_hs_list = []
                for token_out_attr in out_attr:

                    hs_list = []
                    for idx, layer_hs in enumerate(token_out_attr):
                        # in Hugging Face Transformers v4, there's an extra index for batch
                        if len(layer_hs.shape) == 3:  # If there's a batch dimension, pick the first oen
                            hs = layer_hs.cpu().detach()[0].unsqueeze(0)  # Adding a dimension to concat to later
                        # Earlier versions are only 2 dimentional
                        # But also, in v4, for GPT2, all except the last one would have 3 dims, the last layer
                        # would only have two dims
                        else:
                            hs = layer_hs.cpu().detach().unsqueeze(0)

                        hs_list.append(hs)

                    # First hidden state is the embedding layer, skip it
                    # FIXME: do this in a cleaner way
                    hs_list = torch.cat(hs_list, dim=0)
                    embedding_states = hs_list[0]
                    hidden_states = hs_list[1:]
                    tokens_hs_list.append(hidden_states)

                setattr(output, attributes, tokens_hs_list)

        # Pass 'hidden_states' to 'decoder_hidden_states'
        if getattr(output, "hidden_states", None) is not None:
            assert getattr(output, "encoder_hidden_states", None) is None \
                   and getattr(output, "decoder_hidden_states", None) is None, \
                "Not expected to have encoder_hidden_states/decoder_hidden_states with 'hidden_states'"
            setattr(output, "decoder_hidden_states", output.hidden_states)

        encoder_hidden_states = getattr(output, "encoder_hidden_states", None)
        decoder_hidden_states = getattr(output, "hidden_states", getattr(output, "decoder_hidden_states", None))

        # Turn activations from dict to a proper array
        activations_dict = self._all_activations_dict
        for layer_type, activations in activations_dict.items():
            self.activations[layer_type] = activations_dict_to_array(activations)

        if decoder_input_ids is not None:
            assert len(decoder_input_ids.size()) == 2
            all_token_ids = torch.cat([input_id, decoder_input_ids], dim=-1)[0]
        else:
            all_token_ids = input_id[0]

        tokens = self.tokenizer.convert_ids_to_tokens(all_token_ids)
        attributions = self.attributions
        attn = getattr(output, "attentions", None)

        return OutputSeq(**{'tokenizer': self.tokenizer,
                            'token_ids': all_token_ids.unsqueeze(0),  # Add a batch dimension
                            'n_input_tokens': n_input_tokens,
                            'output_text': self.tokenizer.decode(all_token_ids),
                            'tokens': [tokens],  # Add a batch dimension
                            'encoder_hidden_states': encoder_hidden_states,
                            'decoder_hidden_states': decoder_hidden_states,
                            'embedding_states': embedding_states,
                            'attention': attn,
                            'attribution': attributions,
                            'activations': self.activations,
                            'collect_activations_layer_nums': self.collect_activations_layer_nums,
                            'lm_head': self.model.lm_head,
                            'model_type': self.model_type,
                            'device': self.device,
                            'config': self.model_config})

    def __call__(self, input_tokens: torch.Tensor):
        """
        Run a forward pass through the model. For when we don't care about output tokens.
        Currently only support activations collection. No attribution/saliency.

        Usage:

        ```python
        inputs = lm.tokenizer("Hello computer", return_tensors="pt")
        output = lm(inputs)
        ```

        Args:
            input_tokens: tuple returned by tokenizer( TEXT, return_tensors="pt").
                contains key 'input_ids', its value tensor with input token ids.
                Shape is (batch_size, sequence_length).
                Also a key for masked tokens
        """

        if 'input_ids' not in input_tokens:
            raise ValueError("Parameter 'input_tokens' needs to have the attribute 'input_ids'."
                             "Verify it was produced by the appropriate tokenizer with the "
                             "parameter return_tensors=\"pt\".")

        # Move inputs to GPU if the model is on GPU
        if self.model.device.type == "cuda" and input_tokens['input_ids'].device.type == "cpu":
            input_tokens = self.to(input_tokens)

        # Remove downstream. For now setting to batch length
        n_input_tokens = len(input_tokens['input_ids'][0])

        # attach hooks
        self._attach_hooks(self.model)

        # model
        if self.model_type == 'mlm':
            output = self.model(**input_tokens, return_dict=True)
            lm_head = None
        elif self.model_type == 'causal':
            output = self.model(**input_tokens, return_dict=True, use_cache=False)
            lm_head = self.model.lm_head
        elif self.model_type == 'enc-dec':
            decoder_input_ids = self.model._prepare_decoder_input_ids_for_generation(input_tokens['input_ids'], None,
                                                                                     None)
            output = self.model(**input_tokens, decoder_input_ids=decoder_input_ids, return_dict=True, use_cache=False)
            lm_head = self.model.lm_head
        else:
            raise NotImplemented(f"model type {self.model_type} not found")

        # Turn activations from dict to a proper array
        activations_dict = self._all_activations_dict
        for layer_type, activations in activations_dict.items():
            self.activations[layer_type] = activations_dict_to_array(activations)

        encoder_hidden_states = getattr(output, "encoder_hidden_states", None)
        decoder_hidden_states = getattr(output, "hidden_states", getattr(output, "decoder_hidden_states", None))

        if self.model_type in ['causal', 'mlm']:
            # First hidden state of the causal model is the embedding layer, skip it
            # FIXME: do this in a cleaner way
            embedding_states = decoder_hidden_states[0]
            decoder_hidden_states = decoder_hidden_states[1:]

        elif self.model_type == 'enc-dec':
            embedding_states = encoder_hidden_states[0]
            encoder_hidden_states = encoder_hidden_states[1:]

        else:
            raise NotImplemented(f"model type {self.model_type} not found")

        tokens = []
        for i in input_tokens['input_ids']:
            token = self.tokenizer.convert_ids_to_tokens(i)
            tokens.append(token)

        attn = getattr(output, "attentions", None)
        return OutputSeq(**{'tokenizer': self.tokenizer,
                            'token_ids': input_tokens['input_ids'],
                            'n_input_tokens': n_input_tokens,
                            'tokens': tokens,
                            'encoder_hidden_states': encoder_hidden_states,
                            'decoder_hidden_states': decoder_hidden_states,
                            'embedding_states': embedding_states,
                            'attention': attn,
                            'activations': self.activations,
                            'collect_activations_layer_nums': self.collect_activations_layer_nums,
                            'lm_head': lm_head,
                            'model_type': self.model_type,
                            'device': self.device,
                            'config': self.model_config})

    def _get_embeddings(self, input_ids) -> Tuple[torch.HalfTensor, torch.HalfTensor]:
        """
        Get token embeddings and one-hot vector into vocab. It's done via matrix multiplication
        so that 
        attribution is available when needed.
        Args:
            input_ids: Int tensor containing token ids. Of length (sequence length).
            Generally returned from the tokenizer such as
            lm.tokenizer(text, return_tensors="pt")['input_ids'][0]
        Returns:
            inputs_embeds: Embeddings of the tokens. Dimensions are (sequence_len, d_embed)
            token_ids_tensor_one_hot: Dimensions are (sequence_len, vocab_size)
        """

        embedding_matrix = self.model_embeddings

        vocab_size = embedding_matrix.shape[0]

        one_hot_tensor = self.to(_one_hot_batched(input_ids, vocab_size))
        token_ids_tensor_one_hot = one_hot_tensor.clone().requires_grad_(True)

        inputs_embeds = torch.matmul(token_ids_tensor_one_hot.half() , embedding_matrix.half())
        return inputs_embeds, token_ids_tensor_one_hot

    def _attach_hooks(self, model):
        """
        Adopted from `_attach_hooks` at https://github.com/jalammar/ecco/blob/main/src/ecco/lm.py
        """
        # TODO: Collect activations for more than 1 step

        if self._hooks:
            # skip if hooks are already attached
            return

        for name, module in model.named_modules():
            # Add hooks to capture activations in every FFNN

            if re.search(self.collect_activations_layer_name_sig, name):
                if self.collect_activations_flag:
                    self._hooks[name] = module.register_forward_hook(
                        lambda self_, input_, output,
                               name=name: self._get_activations_hook(name, input_))

                # Register neuron inhibition hook
                self._hooks[name + '_inhibit'] = module.register_forward_pre_hook(
                    lambda self_, input_, name=name: \
                        self._inhibit_neurons_hook(name, input_)
                )

    def _remove_hooks(self):
        for handle in self._hooks.values():
            handle.remove()
        self._hooks = {}

    def _get_activations_hook(self, name: str, input_):
        """
        Collects the activation for all tokens (input and output).
        The default activations collection method.

        Args:
            input_: activation tuple with dimensions (batch_size, sequence_length, neurons)
            name: either encoder or decoder
        """
        layer_number = re.search("(?<=\.)\d+(?=\.)", name).group(0)
        layer_type = 'encoder' if name.startswith('encoder.') else 'decoder'

        collecting_this_layer = (self.collect_activations_layer_nums is None) or (
                layer_number in self.collect_activations_layer_nums)

        if collecting_this_layer:
            # Initialize the layer's key the first time we encounter it
            if layer_number not in self._all_activations_dict:
                self._all_activations_dict[layer_type][layer_number] = [0]

            # For MLM, we only run one inference step. We save it.
            # For Causal LM, we could be running multiple inference steps with generate(). In that case,
            # overwrite the previous step activations. This collects all activations in the last step
            # Assuming all input tokens are presented as input, no "past"
            # The inputs to c_proj already pass through the gelu activation function
            self._all_activations_dict[layer_type][layer_number] = input_[0].detach().cpu().numpy()

    def _inhibit_neurons_hook(self, name: str, input_tensor):
        """
        After being attached as a pre-forward hook, it sets to zero the activation value
        of the neurons indicated in self.neurons_to_inhibit
        """

        layer_number = re.search("(?<=\.)\d+(?=\.)", name).group(0)
        if layer_number in self.neurons_to_inhibit.keys():
            for n in self.neurons_to_inhibit[layer_number]:
                input_tensor[0][0][-1][n] = 0  # tuple, batch, position

        if layer_number in self.neurons_to_induce.keys():
            for n in self.neurons_to_induce[layer_number]:
                input_tensor[0][0][-1][n] = input_tensor[0][0][-1][n] * 10  # tuple, batch, position

        return input_tensor

    def display_input_sequence(self, input_ids):
        tokens = []
        for idx, token_id in enumerate(input_ids):
            type = "input"
            raw_token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
            clean_token = self.tokenizer.decode(token_id)
            # Strip prefixes because bert decode still has ## for partials even after decode()
            clean_token = strip_tokenizer_prefix(clean_token)
            tokens.append({
                'token': clean_token,
                'is_partial': is_partial_token(self.model_config, raw_token),
                'position': idx,
                'token_id': int(token_id),
                'type': type})
        data = {'tokens': tokens}

        d.display(d.HTML(filename=os.path.join(self._path, "setup.html")))

        viz_id = f'viz_{round(random.random() * 1000000)}'

        # TODO: Stop passing tokenization_config to JS now that
        # it's handled with the is_partial parameter
        js = f"""
             requirejs( ['basic', 'ecco'], function(basic, ecco){{
                basic.init('{viz_id}') // Python needs to know the viz id. Used for each output token.
                window.ecco['{viz_id}'] = new ecco.renderOutputSequence({{
                        parentDiv: '{viz_id}',
                        data: {json.dumps(data)},
                        tokenization_config: {json.dumps(self.model_config['tokenizer_config'])}
                }})
             }}, function (err) {{
                console.log(err);
            }})
            """

        d.display(d.Javascript(js))
        return viz_id

    def display_token(self, viz_id, token_id, position):
        raw_token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        clean_token = self.tokenizer.decode(token_id)
        # Strip prefixes because bert decode still has ## for partials even after decode()
        clean_token = strip_tokenizer_prefix(clean_token)

        token = {
            'token': clean_token,
            'is_partial': is_partial_token(self.model_config, raw_token),
            'token_id': int(token_id),
            'position': position,
            'type': 'output'
        }
        js = f"""
            // We don't really need these require scripts. But this is to avert
            //this code from running before display_input_sequence which DOES require external files
            requirejs(['basic', 'ecco'], function(basic, ecco){{
                    console.log('addToken viz_id', '{viz_id}');
                    window.ecco['{viz_id}'].addToken({json.dumps(token)})
                    window.ecco['{viz_id}'].redraw()
            }})
            """
        d.display(d.Javascript(js))

    def predict_token(self, inputs, topk=50, temperature=1.0):
        output = self.model(**inputs)
        scores = output[0][0][-1] / temperature
        s = scores.detach().numpy()
        sorted_predictions = s.argsort()[::-1]
        sm = F.softmax(scores, dim=-1).detach().numpy()

        tokens = [self.tokenizer.decode([t]) for t in sorted_predictions[:topk]]
        probs = sm[sorted_predictions[:topk]]

        prediction_data = []
        for idx, (token, prob) in enumerate(zip(tokens, probs)):
            prediction_data.append({'token': token,
                                    'prob': str(prob),
                                    'ranking': idx + 1,
                                    'token_id': str(sorted_predictions[idx])
                                    })

        params = prediction_data

        viz_id = 'viz_{}'.format(round(random.random() * 1000000))

        d.display(d.HTML(filename=os.path.join(self._path, "html", "predict_token.html")))
        js = """
            requirejs(['predict_token'], function(predict_token){{
            if (window.predict === undefined)
                window.predict = {{}}
            window.predict["{}"] = new predict_token.predictToken("{}", {})
            }}
            )
            """.format(viz_id, viz_id, json.dumps(params))
        d.display(d.Javascript(js))

    def sample_output_token(scores, do_sample, temperature, top_k, top_p):
        # TODO: Add beam search in here
        if do_sample:
            # Temperature (higher temperature => more likely to sample low probability tokens)
            if temperature != 1.0:
                scores = scores / temperature
            # Top-p/top-k filtering
            next_token_logscores = transformers.generation_utils. \
                top_k_top_p_filtering(scores,
                                      top_k=top_k,
                                      top_p=top_p)
            # Sample
            probs = F.softmax(next_token_logscores, dim=-1)

            prediction_id = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy decoding
            prediction_id = torch.argmax(scores, dim=-1)
        prediction_id = prediction_id.squeeze()
        return prediction_id


def _one_hot(token_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    return torch.zeros(len(token_ids), vocab_size, device=token_ids.device).scatter_(1, token_ids.unsqueeze(1), 1.)


def _one_hot_batched(token_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    batch_size, num_tokens = token_ids.shape
    return torch.zeros(batch_size, num_tokens, vocab_size, device=token_ids.device).scatter_(-1,
                                                                                             token_ids.unsqueeze(-1),
                                                                                             1.)


def activations_dict_to_array(activations_dict):
    """
    Converts the dict used to collect activations into an array of the
    shape (batch, layers, neurons, token position).
    Args:
        activations_dict: python dictionary. Contains a key/value for each layer
        in the model whose activations were collected. Key is the layer id ('0', '1').
        Value is a tensor of shape (batch, position, neurons).
    """

    activations = []
    for i in sorted(activations_dict.keys()):
        activations.append(activations_dict[i])

    activations = np.array(activations)
    # 'activations' now is in the shape (layer, batch, position, neurons)
    activations = np.swapaxes(activations, 2, 3)
    activations = np.swapaxes(activations, 0, 1)
    return activations
