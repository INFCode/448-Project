import numpy as np
import torch
from network import *
from process_dataset import *

if __name__ == "__main__":
    # load the saved model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_checkpoint = False
    filename = "./data/paradetox/paradetox.tsv"
    dataset, w2id, id2w, vocab = generate_dataset(filename, device)
    train_loader, val_loader, test_loader = split_dataset(dataset)

    max_out_length = 20

    vocab_size = len(w2id)
    model = Autoencoder(
        vocab_size=vocab_size,
        embed_size=64,
        hidden_size=2,
        num_layers=2,
        label_size=1,
        attention_size=16,
        max_output=max_out_length,
        pad_id=w2id[pad_token],
        eos_id=w2id[end_of_sentence_token],
        device=device,
    )



    model = model.load_state_dict(torch.load("./model_saved/supervised.pt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    filename = "./data/paradetox/paradetox.tsv"
    dataset, w2id, id2w, vocab = generate_dataset(filename, device)
    train_loader, val_loader, test_loader = split_dataset(dataset)

    predicted = []
    for i, (toxic, detoxic) in enumerate(test_loader):
        batch_size = toxic.shape[0]
        s_in = torch.zeros([batch_size, 1], device=device)
        s_out = torch.ones([batch_size, 1], device=device)
        predicted.append(model.forward(toxic, s_in, s_out))

    predicted = torch.cat(predicted, dim=0)

    predicted = predicted.argmax(axis=-1)
    with open("./output/supervised_output.txt", "w") as f:
        for i, word_list in enumerate(tensor_to_words(predicted, id2w)):
            sentence = " ".join(word_list)
            f.write(sentence + "\n")

