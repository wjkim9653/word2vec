import json
import pandas as pd
from Korpora import Korpora
from konlpy.tag import Okt
from pprint import pprint
from tqdm import tqdm
from collections import Counter


def save_to_file(obj, filename):
    with open(filename, 'w') as f:
        json.dump(obj, f)


def load_from_file(filename):
    with open(filename, 'r') as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        if all(isinstance(k, str) and k.isdigit() for k in obj.keys()):
            obj = {int(k):v for k, v in obj.items()}
    return obj

# Building Vocabulary (w/ morphemes)
def build_vocab(corpus, n_vocab, special_tokens):
    """
    :param corpus: (dataframe) corpus
    :param n_vocab: (int) vocabulary size limit
    :param special_tokens: (list) list of special tokens
    :return: vocab: (list) vocabulary (w/ length of n_vocab)
    """
    counter = Counter()  # built in python counter class instance
    for tokens in corpus:
        counter.update(tokens)  # add new lists (within corpus list) and update counter
    vocab = special_tokens  # populate vocab w/ special tokens first ([<unk>, ...])

    # get the top int(n_vocab) # of tokens and populate vocabulary w/ them in DESC order
    for token, count in counter.most_common(n_vocab):
        vocab.append(token)

    return vocab


# Extract Word Pairs : [[center_word,context_word], ...]
def get_word_pairs(tokens, window_size):
    """
    :param tokens: (list) list(sentences) of list(tokens of a sentences)
    :param window_size: (int) # of context_words to extract from each center_word
    :return: (list) extracted word pairs ,list of lists
    """
    pairs = []
    for sentence in tqdm(tokens, desc="sentence -> word pair"):
        sentence_length = len(sentence)
        for idx, center_word in enumerate(sentence):
            window_start = max(0, idx - window_size)  # window_start index can't be smaller than 0
            window_end = min(sentence_length, idx + window_size)  # window_end index can't be bigger than length of list
            center_word = sentence[idx]  # string
            context_words = sentence[window_start:idx] + sentence[idx+1:window_end]  # list of strings
            for context_word in context_words:
                pairs.append([center_word, context_word])
    return pairs  # list of lists


# Convert to Index Pairs [[595, 199], ...]
def get_index_pairs(word_pairs, token_to_id):
    pairs = []
    unk_index = token_to_id['<unk>']  # words not present in vocabulary : classify as <unk> idx
    for word_pair in tqdm(word_pairs, desc="word pair -> index pair"):
        center_word, context_word = word_pair
        center_index = token_to_id.get(center_word, unk_index)  # if not present in vocab, assign unk_index
        context_index = token_to_id.get(context_word, unk_index)  # if not present in vocab, assign unk_index
        pairs.append([center_index, context_index])
    return pairs  # list of lists, each list : [(int)idx, (int)idx]


if __name__ == "__main__":
    # Loading NSMC dataset
    corpus = Korpora.load("nsmc")
    corpus = pd.DataFrame(corpus.test)  # Corpus: Korpora's "nsmc" (movie dataset) Test Set
    # print(corpus)

    # Tokenization (to get Morphemes from Corpus)
    tokenizer = Okt()

    # Okt Tokenizer's morphs() to extract Morphemes from corpus
    tokens = [tokenizer.morphs(review) for review in tqdm(corpus.text, desc='Tokenizing')]
    print(tokens[:5])

    vocab = build_vocab(corpus=tokens, n_vocab=5000, special_tokens=['<unk>'])
    token_to_id = {token: idx for idx, token in enumerate(vocab)}  # id: corresponding idx of each token
    id_to_token = {idx: token for idx, token in enumerate(vocab)}  # token: corresponding token of each idx

    print(f'Vocab: {vocab[:5]} | Size : {len(vocab)}')  # length of vocab should equal 5001 (special token(1) + token(5000))

    word_pairs = get_word_pairs(tokens=tokens, window_size=2)
    print(word_pairs[:5])

    index_pairs = get_index_pairs(word_pairs=word_pairs, token_to_id=token_to_id)
    print(index_pairs[:5])


    save_to_file(obj=tokens, filename='tokens.json')
    save_to_file(obj=vocab, filename='vocab.json')
    save_to_file(obj=token_to_id, filename='token_to_id.json')
    save_to_file(obj=id_to_token, filename='id_to_token.json')
    save_to_file(obj=word_pairs, filename='word_pairs.json')
    save_to_file(obj=index_pairs, filename='index_pairs.json')
    print('done')
