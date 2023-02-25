import pandas as pd
import spacy
from spacy.tokens import Doc
from utils.constantnames import PreprocessNames
import os

DATA_DIR = "data"


def create_coref_csv(news_vendor: str, object_name: str, input_csv_name: str,
                     output_dir: str = "data/preprocessed_data"):
    nlp = spacy.load("en_coreference_web_trf")
    df = pd.read_csv(f'{DATA_DIR}/{input_csv_name}')
    output_filename = f'{news_vendor}_{object_name}_{PreprocessNames.COREF}.csv'
    full_output_path = f'{output_dir}/{output_filename}'
    if os.path.isfile(full_output_path):
        return pd.read_csv(full_output_path)
    else:
        for index, row in df.iterrows():
            text = row['text']
            doc = nlp(text)
            print(doc.spans)
            coref_text = resolve_references(doc, object_name)
            df.at[index, 'text'] = coref_text
        try:
            df.to_csv(full_output_path)
        except Exception as e:
            df.to_csv(f'{output_filename}')
        return df


def resolve_references(doc: Doc, object: str = None) -> str:
    """Function for resolving references with the coref ouput
    doc (Doc): The Doc object processed by the coref pipeline
    RETURNS (str): The Doc string with resolved references
    """
    # token.idx : token.text
    token_mention_mapper = {}
    output_string = ""
    clusters = [
        val for key, val in doc.spans.items() if key.startswith("coref_cluster")
    ]

    # Iterate through every found cluster
    for cluster in clusters:
        first_mention = cluster[0]
        if object in first_mention.text:
            # Iterate through every other span in the cluster
            for mention_span in list(cluster)[1:]:
                # Set first_mention as value for the first token in mention_span in the token_mention_mapper
                token_mention_mapper[mention_span[0].idx] = object + mention_span[0].whitespace_

                for token in mention_span[1:]:
                    # Set empty string for all the other tokens in mention_span
                    token_mention_mapper[token.idx] = ""

    # Iterate through every token in the Doc
    for token in doc:
        # Check if token exists in token_mention_mapper
        if token.idx in token_mention_mapper:
            output_string += token_mention_mapper[token.idx]
        # Else add original token text
        else:
            output_string += token.text + token.whitespace_

    return output_string


input_csv = 'data/cnn-articles-netanyahu.csv'
