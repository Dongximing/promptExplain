import logging
import os
import time

start_time = time.time()  # note at start of script

# %% Environmentf Setup
from utils import setup_bmc5_env, setup_logging_results

setup_bmc5_env()

# PARAMS SPECIFIC TO  THE CURRENT RUN
os.environ['CUDA_CORE'] = "cuda:0"  # NOTE: belongs to setup func above, but keep here for running diff scripts.
dataset_range = range(0, 1)

results_path = setup_logging_results(dataset_range)
logging.info(f"Script started")

# %% Data read and preprocess
from data_read_preprocess import load_and_preprocess

preprocessed_dataset = load_and_preprocess(dataset_range)
logging.info(f"Data read and preprocessing completed for dataset range: {dataset_range}")

# %%  model setup file.

from model_setup import setup_model_tokenizer

tokenizer, model = setup_model_tokenizer()
logging.info("Model setup completed")

# %%

from inference import run_initial_inference

inferenced_df = run_initial_inference(preprocessed_dataset, model, tokenizer)
inferenced_df.to_pickle(f'{results_path}inferenced_df.pkl')
logging.info("Initial inference completed")

# %%

from postprocessing import postproces_inferenced

postprocessed_df = postproces_inferenced(inferenced_df.copy())
postprocessed_df.to_pickle(f'{results_path}postprocessed_df.pkl')
logging.info("Postprocessing completed")

# %% Running peturbation
from peturbation import run_peturbation

peturbed_df, model = run_peturbation(postprocessed_df.copy(),model)
peturbed_df.to_pickle(f'{results_path}peturbed_df.pkl')
logging.info("Perturbation process completed")

# %%

from postprocessing import do_peturbed_reconstruct

reconstructed_df = do_peturbed_reconstruct(peturbed_df.copy(),model)
reconstructed_df.to_pickle(f'{results_path}reconstructed_df.pkl')
logging.info("Reconstruction process completed")

# %%
from inference import run_peturbed_inference

peturbed_inferenced_df = run_peturbed_inference(reconstructed_df, model, tokenizer, results_path)
peturbed_inferenced_df.to_pickle(f'{results_path}peturbed_inferenced_df.pkl')
logging.info("Inference on petrubed data completed")

# %% save the final execution time!
execution_minutes = round((start_time - time.time()) / 60, 2)
logging.info(f"Time taken to execute the entire script, {execution_minutes=}.")
