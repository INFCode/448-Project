from nltk import data
from nltk.tokenize import word_tokenize
from nltk.lm import Vocabulary
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.nn.utils.rnn import pad_sequence

pad_token = "<PAD>"
end_of_sentence_token = "<EOS>"
unknown_token = "<UNK>"


def load_data(filename, keep_single=False):
    texts = []
    labels = []
    with open(filename, "r") as file:
        first_line = True
        for line in file:
            if first_line:
                first_line = False
                continue
            datapoint = line.strip().split("\t")
            if keep_single:
                datapoint = datapoint[:2]
            for label in datapoint[1:]:
                texts.append(datapoint[0])
                labels.append(label)
    return texts, labels


def tokenize(sentences):
    print(f"{sentences[0]=}")
    return [word_tokenize(sentence) + [end_of_sentence_token] for sentence in sentences]


def get_word_mapping(dataset, cutoff=2):
    word_to_id = {}
    id_to_word = {}
    idx = 0

    def add_to_word(word, index):
        word_to_id[word] = index
        id_to_word[index] = word
        index += 1
        return index

    idx = add_to_word(pad_token, idx)

    tokens = [token for sentence in dataset for token in sentence]
    vocab = Vocabulary(tokens, cutoff, unknown_token)

    for token in vocab:
        idx = add_to_word(token, idx)

    if unknown_token not in vocab:
        idx = add_to_word(unknown_token, idx)

    return word_to_id, id_to_word, vocab


def sentences_to_tensor(sentences, word_to_id, vocab, device):
    return pad_sequence(
        [
            torch.tensor(
                [
                    word_to_id[word if word in vocab else unknown_token]
                    for word in sentence
                ],
                device=device,
            )
            for sentence in sentences
        ],
        batch_first=True,
    )


def tensor_to_words(tensor, id_to_word):
    return [[id_to_word[idx.item()] for idx in row] for row in tensor]


def generate_dataset(filename, device):
    texts, labels = load_data(filename)
    print(f"{len(texts)=}, {len(labels)=}")
    tokenized_texts = tokenize(texts)
    tokenized_labels = tokenize(labels)

    w2id, id2w, vocab = get_word_mapping(tokenized_labels + tokenized_texts)

    text_tensors = sentences_to_tensor(tokenized_texts, w2id, vocab, device)
    label_tensors = sentences_to_tensor(tokenized_labels, w2id, vocab, device)

    print(f"{text_tensors.size()=}, {label_tensors.size()=}")

    dataset = TensorDataset(text_tensors, label_tensors)
    return dataset, w2id, id2w, vocab


def split_dataset(
    dataset,
    train_weight=0.7,
    val_weight=0.2,
    train_batch_size=4,
    val_batch_size=4,
    test_batch_size=4,
):
    # Define the sizes of your training, validation, and testing sets
    train_size = int(train_weight * len(dataset))
    val_size = int(val_weight * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Use random_split to split the dataset into training, validation, and testing sets
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Define your data loaders for the training, validation, and testing sets
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    dataset, w2id, id2w, vocab = generate_dataset(
        "./data/paradetox/paradetox.tsv", "cpu"
    )
    # print(f"{dataset=}")
    # print(f"{w2id=}")
    # print(f"{id2w=}")
    # print(f"{vocab=}")
    train_loader, val_loader, test_loader = split_dataset(dataset)
    for k, v in train_loader:
        k_sent = " ".join([id2w[x.item()] for x in k[0] if x != w2id[pad_token]])
        print(f"{k=},\n{v=}")
        print(f"{k_sent=}")
        break
