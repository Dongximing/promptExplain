import re

from nltk.tokenize import word_tokenize


def postproces_inferenced(df):
    def join_token_texts(component_dict, key):
        tokens_list = component_dict['component_tokens_dict'][key]
        return tokens_list

    # Applying the function to each row for both 'instructions' and 'query'
    df['instructions_tokens'] = df['component_level'].apply(join_token_texts, key='instruction')
    df['query_tokens'] = df['component_level'].apply(join_token_texts, key='query')

    return df


def do_peturbed_reconstruct(df, modification_types=None):
    if modification_types is None:
        modification_types = ['top', 'bottom']

    # Helper function to tokenize and reconstruct prompts
    def reconstruct_prompt(row, modified_column, original_column='prompt'):
        modified_tokens = row[modified_column]
        tokenized_original = word_tokenize(row[original_column])
        updated_tokens = {
            mod_token['position']: mod_token['token']
            for mod_token in modified_tokens
            if 'position' in mod_token and 'token' in mod_token
        }
        return ' '.join([updated_tokens.get(i, token) for i, token in enumerate(tokenized_original)])

    # Extracting the unique percentage levels from column names
    percentage_levels = {float(re.search(r'_(\d+\.\d+)_', col).group(1)) for col in df.columns if
                         re.search(r'_\d+\.\d+_', col)}

    for mod_type in modification_types:
        for token_type in ['instruction', 'query']:
            for pct in percentage_levels:
                perturbed_col_pattern = f'{token_type}_token_{mod_type}_{pct}_peturbed'
                reconstructed_col_name = f'{mod_type}_reconstructed_{token_type}_{pct}'

                # Apply the reconstruct function only if the perturbed column exists
                if perturbed_col_pattern in df.columns:
                    df[reconstructed_col_name] = df.apply(reconstruct_prompt, axis=1,
                                                          modified_column=perturbed_col_pattern)

    return df


def old_do_peturbed_reconstruct(df, modification_types=None):
    def reconstruct_modified_prompt(original_prompt, modified_tokens):
        # Tokenize the original prompt to match positions
        tokenized_prompt = word_tokenize(original_prompt)
        position_to_token = {i: token for i, token in enumerate(tokenized_prompt)}

        # Update the mapping with modified tokens based on their positions
        for mod_token in modified_tokens:
            if 'position' in mod_token and 'token' in mod_token:
                position_to_token[mod_token['position']] = mod_token['token']

        # Reconstruct the prompt from the updated mapping
        reconstructed_prompt = [position_to_token[pos] for pos in sorted(position_to_token)]
        return ' '.join(reconstructed_prompt)

    # Identifying the unique percentage levels in the DataFrame columns
    percentage_levels = set()
    for column in df.columns:
        if 'peturbed' in column:
            try:
                # Extracting the percentage level from column name
                pct_level = column.split('_')[3]
                # Ensuring the extracted percentage is a float
                if pct_level.replace('.', '', 1).isdigit():
                    percentage_levels.add(float(pct_level))
            except IndexError:
                continue  # Skip columns that don't fit the expected naming convention

    for mod_type in modification_types:
        for token_type in ['instruction', 'query']:
            for pct in percentage_levels:
                column_name = f'{token_type}_token_{mod_type}_{pct}_peturbed'
                new_column_name = f'{mod_type}_reconstructed_{token_type}_{pct}'

                for idx, row in df.iterrows():
                    if column_name in df.columns:  # Check if this column exists
                        original_prompt = row['prompt']
                        modified_tokens = row[column_name]
                        modified_prompt = reconstruct_modified_prompt(original_prompt, modified_tokens)
                        df.at[idx, new_column_name] = modified_prompt

    return df
