def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def download_nltk():
    import nltk

    # Ensure necessary NLTK data is available
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('words')
    nltk.download('punkt')


def setup_logging_results(current_range):
    import os
    import time, logging

    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")

    # making results dir
    current_range = str(current_range)
    results_path = f"{os.environ.get("RESULTS_PATH")}{current_time}_{current_range}/"
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    # init logging at results_dir
    logging.basicConfig(filename=f'{results_path}log.txt', level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')

    return results_path


def setup_bmc5_env():
    import os
    import sys
    import nltk

    # setup ssl for huggingface download
    os.environ["CURL_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"

    # adding path for the LanguageModelRepo 
    sys.path.append(
        "/home/s4user/PromptExplainabilityProject/LanguageModelRepo")  # Adding wherever Ximings language model is

    os.environ['MODEL_NAME'] = '/home/s4user/model_download/from_kingston1/llama-2-7b-chat-hf'

    # TODO: uncomment
    # os.environ['CUDA_CORE'] = "cuda:0" # update whichever wanna use

    # Loading dataset
    # can load dataset if the SSL doesnt work
    # dataset_file_path = "/home/s4user/model_download/sst2/train-00000-of-00001.parquet"  # Adjust if necessary
    # os.environ['dataset'] = load_dataset('parquet', data_files={'train': dataset_file_path}, split='train')

    # nltk load
    nltk.data.path.append('/home/s4user/PromptExplainabilityProject/nltk_data')
    nltk.data.path.append('/home/s4user/PromptExplainabilityProject/nltk_data/corpora')
    nltk.data.path.append('/home/s4user/PromptExplainabilityProject/nltk_data/tokenizers')

    # results path
    os.environ["RESULTS_PATH"] = "/home/s4user/PromptExplainabilityProject/ExplainabilityModularized/Results/"

    return True


def find_gpu_with_memory_threshold(threshold_gb):
    """
    Finds the first GPU with available memory greater than the specified threshold and returns its name.
    Uses nvidia-smi to get accurate free memory information.
    """
    import torch
    import subprocess
    import re
    import logging


    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            # Running nvidia-smi command to get memory usage
            smi_output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader', '-i', str(i)], encoding='utf-8')
            free_memory_gb = float(smi_output.strip()) / 1024  # Convert MiB to GiB

            if free_memory_gb > threshold_gb:
                gpu_name = torch.cuda.get_device_name(i)
                logging.info(f"For cuda:{i}, found {free_memory_gb=}, {threshold_gb=}")
                return f'cuda:{i}'  # Return the name of the GPU meeting the threshold

    return -1
