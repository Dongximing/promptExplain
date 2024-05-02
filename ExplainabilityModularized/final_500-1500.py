import logging
import os
import time

start_time = time.time()  # note at start of script

# %% Environment Setup
from utils import setup_bmc5_env, setup_logging_results

setup_bmc5_env()

# PARAMS SPECIFIC TO  THE CURRENT RUN
os.environ['CUDA_CORE'] = "cuda:0"  # NOTE: belongs to setup func above, but keep here for running diff scripts.
dataset_range = range(500, 1500)

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

import pandas as pd 

reconstructed_df = pd.read_pickle('/home/s4user/PromptExplainabilityProject/ExplainabilityModularized/Results/2024-03-27_16-24-03_range(500, 1500)/reconstructed_df.pkl')
logging.info("Reconstruction read from old file")

# %%
from inference import run_peturbed_inference

peturbed_inferenced_df = run_peturbed_inference(reconstructed_df, model, tokenizer, results_path)
peturbed_inferenced_df.to_pickle(f'{results_path}peturbed_inferenced_df.pkl')
logging.info("Inference on petrubed data completed")

# %% save the final execution time!
execution_minutes = round((start_time - time.time()) / 60, 2)
logging.info(f"Time taken to execute the entire script, {execution_minutes=}.")
