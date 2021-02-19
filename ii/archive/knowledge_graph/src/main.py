import pickle
import numpy as np
from pprint import pprint
import logging

from genre import GENRE
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_fairseq


if __name__ == '__main__':
    # load the model
    model = GENRE.from_pretrained("/models/fairseq_e2e_entity_linking_aidayago.tar.gz").eval()

    # load a document and separate sentences
    # TODO: load a document
    sentences = ["In 1921, Einstein received a Nobel Prize."]

    # End-to-End Entity Linking
    prefix_allowed_tokens_fn = get_end_to_end_prefix_allowed_tokens_fn_fairseq(model, sentences)
    result = model.sample(
        sentences,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    )

    # generate the knowledge graph

    # output csv files
    # TODO: KG.csv
    # TODO: entity.csv
    # TODO: sentence.csv

    pprint(result)