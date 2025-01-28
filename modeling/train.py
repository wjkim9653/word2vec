from dataset.preprocess import load_from_file
from word2vec import VanillaSkipGram
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from tqdm import tqdm
import numpy as np
from numpy.linalg import norm
import os
from pprint import pprint


def cosine_similarity(embedding_token, embedding_matrix):
    """
    :param embedding_token: [1, embedding_dim]
    :param embedding_matrix: [n_vocab, embedding_dim]
    :return:
    """
    cosine_matrix = (np.dot(embedding_matrix, embedding_token)  # dot product between [n_vocab, embedding_dim] * [1, embedding_dim]
                     / (norm(embedding_matrix, axis=1) * norm(embedding_token)))  # `axis=1` ensures each row(w/ shape of [1, embedding_dim]) is used to calculate norm
    return cosine_matrix  # shape: [n_vocab, 1]  (each row representing token, and single column representing cosine similarity score)


def top_n_index(cosine_matrix, n):
    """

    :param cosine_matrix: [n_vocab, 1]
    :param n:
    :return: []
    """
    # get the index before sorting w/ numpy's argsort() method
    closest_indexes = cosine_matrix.argsort()[::-1]  # [n_vocab, embedding_dim] and sort in DESCENDING ORDER (-1)

    top_n = closest_indexes[1:n+1]  # get 5 highest similar vectors, w/o itself (which sits at top, thus removing row 0)
    return top_n  # [(int)idx1, (int)idx2, (int)idx3, ...]


if __name__ == "__main__":
    tokens = load_from_file("../dataset/tokens.json"); print(f'# of Sentences : {len(tokens)} ✅')
    vocab = load_from_file("../dataset/vocab.json"); print(f'# of Vocab : {len(vocab)} ✅')
    token_to_id = load_from_file("../dataset/token_to_id.json")
    id_to_token = load_from_file("../dataset/id_to_token.json")
    word_pairs = load_from_file("../dataset/word_pairs.json"); print(f'# of Word Pairs : {len(word_pairs)} ✅')
    index_pairs = load_from_file("../dataset/index_pairs.json"); print(f'# of Index Pairs : {len(index_pairs)} ✅')
    print(vocab[:50])

    # Turn into (feature & label)Tensors (for training)
    index_pairs = torch.tensor(index_pairs)
    center_indexes = index_pairs[:, 0]  # idx 0 of all tensors within index_pairs
    context_indexes = index_pairs[:, 1]  # idx 1 of all tensors within index_pairs

    # Create TensorDataset : feature tensors + label tensors into dataset
    dataset = TensorDataset(center_indexes, context_indexes)  # shape: [N, 2]
    # Apply DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    word2vec = VanillaSkipGram(vocab_size=len(token_to_id), embedding_dim=128).to(device)
    criterion = nn.CrossEntropyLoss().to(device)  # CrossEntroypyLoss for Multi-class Classification (#: vocab size)
    optimizer = optim.SGD(params=word2vec.parameters(), lr=0.1)

    # Train Phase
    if "word2vec_weights(Vanilla_SkipGram).pth" in os.listdir(os.getcwd()):
        print('❄️Loading pretrained word2vec weights...')
        word2vec.load_state_dict(torch.load(f="word2vec_weights(Vanilla_SkipGram).pth", weights_only=True))
    else:
        print('🔥Training word2vec weights...')
        epochs = 20
        for epoch in range(epochs):
            cost = 0.0  # Accumulate Cost per Each Epoch
            for input_ids, target_ids in tqdm(dataloader, desc=f"Epoch: {epoch+1}/{epochs}"):
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)

                logit = word2vec(input_ids)  # output of VanillaSkipGram.forward() is logit (probabilities)
                loss = criterion(logit, target_ids)  # calculate loss against label

                optimizer.zero_grad()  # Reset Accumulated Gradient in Optimizer from Prev Step
                loss.backward()  # Back-Propagation & Loss Update
                optimizer.step()

                cost += loss
            cost = cost / len(dataloader)  # Avg. Loss as Cost
            print(f"\nEpoch: {epoch+1}, Cost: {cost:.5f}")
        print("Train - Done ✅")
        # Save trained weights
        print("Saving..."); torch.save(word2vec.state_dict(), "word2vec_weights(Vanilla_SkipGram).pth"); print("Saved ✅")

    # Let's look at the Embedding Matrix I just trained
    token_to_embedding = dict()
    embedding_matrix = word2vec.embedding.weight.detach().cpu().numpy()  # embedding_matrix: [vocab_size, embedding_dim]

    for word, embedding in zip(vocab, embedding_matrix):  # word, embedding_matrix both have length of vocab_size (5001)
        token_to_embedding[word] = embedding  # can get embedding vector w/ shape of [1, embedding_dim] for each token

    # Let's check random(idx:30) token & its corresponding embedding vector
    index = 3
    token = vocab[index]
    token_embedding = token_to_embedding[token]  # [1, embedding_dim]
    pprint(f'token: {token}\n embedding: {token_embedding}')

    # Let's run Cosing Similarity Search
    n = 5
    cosine_matrix = cosine_similarity(embedding_token=token_embedding, embedding_matrix=embedding_matrix)
    print(f'cosine_matrix(shape:{cosine_matrix.shape}): {cosine_matrix}')
    top_n = top_n_index(cosine_matrix, n=n)
    print(f'top_n: {top_n} (type: {type(top_n)}) (elem type: {type(top_n[0])})')

    print(f"{token}'s top-{n} similar words are:")
    for idx in top_n:
        print(f"{id_to_token[int(idx)]} - cosine similarity score: {cosine_matrix[int(idx)]:.5f}")

