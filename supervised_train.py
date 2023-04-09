import torch
from torch.optim import Adam
from process_dataset import *
from network import Autoencoder
import torch.nn as nn
from tqdm import tqdm


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    filename = "./data/paradetox/paradetox.tsv"
    dataset, w2id, id2w, vocab = generate_dataset(filename, device)
    train_loader, val_loader, test_loader = split_dataset(dataset)

    max_out_length = 100

    vocab_size = len(w2id)
    net = Autoencoder(vocab_size, 32, 32, 4, 1, 8, max_out_length, device)

    net.to(device)
    # Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=0.001, weight_decay=0.0001)

    total_epoch = 2
    for epoch in range(total_epoch):
        for i, (X, label) in tqdm(enumerate(train_loader), total=len(train_loader)):
            if i % 100 == 0:
                checkpoint = {
                    "epoch": epoch + 1,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                epoch += 1
                f_path = "./checkpoints/supervised/checkpoint.pt"
                torch.save(checkpoint, f_path)
            # if epoch >= 2:
            #    break
            running_loss = 0.0
            running_acc = 0.0
            batch_size = X.shape[0]
            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using data from the training set
            s_in = torch.zeros([batch_size, 1], device=device)
            s_out = torch.ones([batch_size, 1], device=device)
            result = net.forward(X, s_in, s_out)
            outputs = result
            # transfer labels into distribution vectors
            # print(X.shape,label.shape)
            y = torch.zeros(result.shape, device=device)
            # print(y.shape)
            for i, sentence in enumerate(label):
                sentence_length = sentence.shape
                for j, index in enumerate(sentence):
                    # print(i,j,index)
                    y[i][j][index] = 1
            # print(outputs.shape, y.shape)
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, y)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()
    print("finished training")
    torch.save(net, "./model_saved/supervised.pt")
    ### Start Test

    predicted = []
    for i, (toxic, _) in enumerate(test_loader):
        if i >= 2:
            break
        batch_size = toxic.shape[0]
        s_in = torch.zeros([batch_size, 1], device=device)
        s_out = torch.ones([batch_size, 1], device=device)
        predicted.append(net.forward(toxic, s_in, s_out))

    predicted = torch.cat(predicted)

    print("finished prediction")

    predicted = predicted.argmax(dim=-1)
    with open("./output/supervised_output.txt", "w") as f:
        for i, word_list in enumerate(tensor_to_words(predicted, id2w)):
            sentence = " ".join(word_list)
            f.write(sentence + "\n")

    # net = Autoencoder(3, 4, 4, 2, 1, 2, 4)
    # x = torch.tensor([[1, 2, 0, 2, 1], [1, 0, 0, 0, 1]])
    # s_in = torch.tensor([[0], [1]])
    # s_out = torch.tensor([[0], [1]])
    # result = net.forward(x, s_in, s_out)
    # print(f"{result}, {result.size()=}")


if __name__ == "__main__":
    train()
