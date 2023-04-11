from process_dataset import (
    get_word_mapping,
    tokenize,
    sentences_to_tensor,
    split_dataset,
)
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


def load_nonparallel_data(filename):
    train = pd.read_csv(filename)
    print(f"{train.info()}")
    # print(f"{train['comment_text'].isnull().sum()}")
    # for i in range(5):
    #    print(f"{train['comment_text'][i]}")
    text = train["comment_text"].tolist()
    # get all columns other than id and comment_text, i.e. all
    all_labels = train[train.columns.difference(["id", "comment_text"])]
    merged_labels = all_labels.astype(bool).any(axis=1).astype(int).tolist()
    print(f"{text[0]=}, {merged_labels[0]=}")
    print(f"{text[6]=}, {merged_labels[6]=}")
    return text, merged_labels


def print_longer_than(tokenized_data, length):
    num_longer = sum([len(lst) > length for lst in tokenized_data])
    print(f"There are {num_longer} strings of length > {length}")


def generate_nonparallel_dataset(filename, device, discard_threshold=-1):
    """
    discard_threshold is the max number of token a string can contain. Any string
    longer than this will be discarded to reduce the memory requirement.
    """
    texts, labels = load_nonparallel_data(filename)
    print(f"{len(texts)=}, {len(labels)=}")
    tokenized_texts = tokenize(texts)

    for i in range(4):
        print_longer_than(tokenized_texts, 10**i)
        print_longer_than(tokenized_texts, 3 * 10**i)

    # discard all strings that are too long.
    if discard_threshold != -1:
        tokenized_texts, labels = zip(
            *[
                (x, y)
                for x, y in zip(tokenized_texts, labels)
                if len(x) <= discard_threshold
            ]
        )

    w2id, id2w, vocab = get_word_mapping(tokenized_texts)

    toxic_texts = []
    nontoxic_texts = []

    for x, y in zip(tokenized_texts, labels):
        if y == 1:
            toxic_texts.append(x)
        else:
            nontoxic_texts.append(x)

    toxic_text_tensors = sentences_to_tensor(toxic_texts, w2id, vocab, device)
    nontoxic_text_tensors = sentences_to_tensor(nontoxic_texts, w2id, vocab, device)

    print(f"{toxic_text_tensors.size()=}, {nontoxic_text_tensors.size()=}")

    toxic_dataset = TensorDataset(toxic_text_tensors)
    nontoxic_dataset = TensorDataset(nontoxic_text_tensors)
    return toxic_dataset, nontoxic_dataset, w2id, id2w, vocab


if __name__ == "__main__":
    tox, nontox, w2id, id2w, vocab = generate_nonparallel_dataset(
        "data/jigsaw/train.csv", "cpu", discard_threshold=1024
    )
    tox_train_loader, tox_val_loader, tox_test_loader = split_dataset(tox)
    nontox_train_loader, nontox_val_loader, nontox_test_loader = split_dataset(nontox)
    for x in tox_train_loader:
        print(f"{x=}")
        break

    for x in nontox_train_loader:
        print(f"{x=}")
        break
