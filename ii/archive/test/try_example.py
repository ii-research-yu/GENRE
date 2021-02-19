
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_fairseq

from genre.base_model import GENRE

def try_genre():
    model = GENRE.from_pretrained("/Users/iimac/Downloads/Nedo_genre/fairseq_e2e_entity_linking_wiki_abs.tar.gz").eval()

    sentences = ["In 1921, Einstein received a Nobel Prize."]

    prefix_allowed_tokens_fn = get_end_to_end_prefix_allowed_tokens_fn_fairseq(model, sentences)

    rest = model.sample(
        sentences,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    )

    print(rest)

if __name__ == "__main__":
    #1
    try_genre()