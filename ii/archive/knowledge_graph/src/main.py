import pickle
import numpy as np
from pprint import pprint
import logging
from typing import Dict, List
import itertools
import csv
from nltk.tokenize import sent_tokenize

from genre import GENRE
from genre.trie import Trie
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_fairseq
from genre.utils import get_entity_spans_fairseq


def values_to_csv(values: List, fpath: str, columns: List[str] = ['id', 'values']) -> None:
    with open(fpath, 'w') as f:
        writer = csv.writer(f)
        if columns is not None:
            writer.writerow(columns)
        for i, v in enumerate(values):
            writer.writerow([i, v])

def list_to_csv(table: List[List], fpath: str, columns: List[str] = None) -> None:
    with open(fpath, 'w') as f:
        writer = csv.writer(f)
        if columns is not None:
            writer.writerow(columns)
        writer.writerows(table)


if __name__ == '__main__':
    # init
    knowledge_graph = []
    entity_list = []
    sentence_list = []

    # load the model
    model = GENRE.from_pretrained("/models/fairseq_e2e_entity_linking_aidayago.tar.gz").eval()

    # load a document and separate sentences
    # TODO: load a document
    text = "God is Great! I won a lottery."
    sentences = sent_tokenize(text)

    # register the sentence IDs
    sentence_list += sentences

    # End-to-End Entity Linking
    # URL: https://github.com/facebookresearch/GENRE/blob/main/examples/fairseq.md#end-to-end-entity-linking
    document_result = get_entity_spans_fairseq(
        model,
        sentences
    )

    # register entities
    for sentence_resuslt in document_result:
        entity_list += [e for _, _, e in sentence_resuslt]
    entity_list = list(set(entity_list))    # get unique entities

    # generate the knowledge graph
    for s_i, sentence_resuslt in enumerate(document_result):
        e_ids = []
        for _, _, entity in sentence_resuslt:
            e_ids.append(entity_list.index(entity))
        for e0, e1 in itertools.permutations(e_ids, 2):
            knowledge_graph.append([e0, e1, s_i])

    # output csv files
    list_to_csv(knowledge_graph, fpath='kg.csv', columns=['entity ID1', 'entity ID2', 'sentence ID'])
    values_to_csv(entity_list, fpath='entity.csv')
    values_to_csv(sentence_list, fpath='sentence.csv')
