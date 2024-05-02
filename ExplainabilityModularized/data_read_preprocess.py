from datasets import load_dataset
from nltk.tokenize import word_tokenize

# global var
instruction = """Analyze the sentiment of the previous sentence and respond only with ^POSITIVE^ or ^NEGATIVE^. Your answer is ^"""


def get_query_instruction_positions(query):
    # Concatenate the sentences
    combined_text = query + ' ' + instruction

    # Tokenize the combined text
    total_range = len(word_tokenize(combined_text))

    # find count of first 
    query_end = len(word_tokenize(query))

    return {"query": list(range(0, query_end)),
            "instruction": list(range(query_end, total_range))
            }


def preprocess_data(examples):
    updated_sentences = []
    ranges = []
    queries = []
    instructions = []
    for query in examples['sentence']:
        combined = query + ' ' + instruction
        updated_sentences.append(combined)
        range_info = get_query_instruction_positions(query)
        ranges.append(range_info)
        queries.append(query)
        instructions.append(instruction)
    examples['sentence'] = updated_sentences
    examples['component_range'] = ranges
    examples['instruction'] = instructions
    examples['query'] = queries
    return examples


def load_and_preprocess(dataset_range):
    dataset = load_dataset("sst2", split='train')
    preprocessed_dataset = dataset.map(preprocess_data, batched=True)  # , load_from_cache_file=False)
    return preprocessed_dataset.select(list(dataset_range))


if __name__ == "__main__":
    from utils import setup_bmc5_env

    setup_bmc5_env()

    df = load_and_preprocess(10)
    print(type(df[1]))
