from enum import _EnumDict
import torch
from torch.optim import Adam
from process_dataset import *
from network import Autoencoder
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def early_stopping(validation_loss, curr_count_to_patience, global_min_loss):
    """Calculate new patience and validation loss.
    Increment curr_count_to_patience by one if new loss is not less than global_min_loss
    Otherwise, update global_min_loss with the current val loss, and reset curr_count_to_patience to 0
    Returns: new values of curr_count_to_patience and global_min_loss
    """
    # TODO implement early stopping
    splits = ["Validation", "Train", "Test"]
    metrics = ["Fluency", "Detoxification", "Synonym"]
    idx = len(metrics[1]) * 0 + 1
    if validation_loss >= global_min_loss:
        curr_count_to_patience += 1
    else:
        global_min_loss = validation_loss
        curr_count_to_patience = 0
    #
    return curr_count_to_patience, global_min_loss


def predict(net, test_loader, device, output_file="./output/supervised_output.txt"):
    ### Start Test

    predicted = []
    for i, (toxic, _) in enumerate(test_loader):
        if i >= 20:
            break
        batch_size = toxic.shape[0]
        predicted.append(net.forward(toxic))

    predicted = torch.cat(predicted)
    original = torch.cat([x for (x, y) in test_loader])
    print("finished prediction")

    predicted = predicted.argmax(dim=-1)
    #predicted = F.softmax(predicted)
    #pred = torch.zeros(predicted.size()[:2])
    #for i, sentence in enumerate(predicted):
    #    for j, word in enumerate(sentence):
    #        pred[i][j] = torch.multinomial(word, 1)
    with open(output_file, "w") as f:
        original_words = tensor_to_words(original, id2w)
        for i, word_list in enumerate(tensor_to_words(predicted, id2w)):
            original_sentence = " ".join(original_words[i])
            sentence = " ".join(word_list)
            f.write(original_sentence + "  " + sentence + "\n")


def train(net, total_epoch, device):
    def _get_metrics(loader):
        fluency = 0
        detoxification = 0
        synonym = 0
        criterion = torch.nn.CrossEntropyLoss()
        running_loss = []
        for X, y in loader:
            with torch.no_grad():
                output = net.forward(X)
                running_loss.append(criterion(output, y).item())
        loss = torch.mean(torch.tensor(running_loss))
        return fluency, detoxification, synonym, loss

    # Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
    loss_fn = nn.CrossEntropyLoss(ignore_index=w2id[pad_token])
    optimizer = Adam(net.parameters(), lr=1e-4)

    # total_epoch = 2
    loss_traj = []
    stats = []
    patience = 5
    curr_count_to_patience = 0
    # initial val loss for early stopping
    global_min_loss = 0

    for epoch in tqdm(range(total_epoch)):
        running_loss = 0.0
        progress = tqdm(enumerate(train_loader), total=len(train_loader))
        if curr_count_to_patience >= patience:
            break
        for i, (X, label) in progress:
            if i % 1000 == 999:
                checkpoint = {
                    "epoch": epoch + 1,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                f_path = "./checkpoints/supervised/checkpoint.pt"
                torch.save(checkpoint, f_path)
                # print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0
            batch_size = X.shape[0]
            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using data from the training set
            result = net(X)
            # outputs = result.transpose(1, 2)
            # transfer labels into distribution vectors
            # print(X.shape,label.shape)
            # y = torch.zeros(result.shape, device=device)
            # print(y.shape)
            # for i, sentence in enumerate(label):
            #    sentence_length = sentence.shape
            #    for j, index in enumerate(sentence):
            #        # print(i,j,index)
            #        y[i][j][index] = 1
            # print(outputs.shape, y.shape)

            # pad as needed
            # len_out = outputs.size(-1)
            # len_label = label.size(1)

            # len_diff = len_label - len_out
            # if len_diff > 0:
            #     # pad the "<PAD>" token at the end
            #     pad_tensor = F.one_hot(
            #         torch.tensor(w2id[pad_token], device=device).long(), vocab_size
            #     )
            #     outputs = torch.cat(
            #         [
            #             outputs,
            #             pad_tensor.repeat(outputs.size(0), len_diff, 1).transpose(1, 2),
            #         ],
            #         dim=2,
            #     )
            # elif len_diff < 0:
            #     label = torch.nn.functional.pad(
            #         label, (0, -len_diff), value=w2id[pad_token]
            #     )
            # print(f"{outputs.size()=}, {label.size()=}")
            # compute the loss based on model output and real labels
            loss = loss_fn(result.view(-1, vocab_size), label.view(-1))
            # print(f"{outputs[:,1,0]=}, {torch.sum(label)=}")
            # backpropagate the loss
            loss.backward()
            running_loss += loss.item()
            progress.set_postfix({"loss": loss.item()})
            if i % 100 == 0:
                loss_traj.append(loss.item())
            # adjust parameters based on the calculated gradients
            nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()
        ### Early stopping
        # train_fluency, train_detoxification, train_synonym, train_loss = _get_metrics(train_loader)
        # val_fluency, val_detoxification, val_synonym, val_loss = _get_metrics(val_loader)
        # curr_count_to_patience, global_min_loss = early_stopping(
        #     val_loss, curr_count_to_patience, global_min_loss
        # )
        # print("epoch ", "validation loss ", val_loss )
        predict(
            net,
            test_loader,
            device,
            output_file=f"./output/supervised_output_mid{epoch}.txt",
        )
    print("finished training")
    torch.save(net.state_dict(), "./model_saved/supervised.pt")
    return net


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_checkpoint = False
    filename = "./data/paradetox/paradetox.tsv"
    dataset, w2id, id2w, vocab = generate_dataset(filename, device)
    train_loader, val_loader, test_loader = split_dataset(dataset)

    # max_out_length = 20

    vocab_size = len(w2id)
    net = Autoencoder(
        embedding_dim = 33,
        hidden_dim = 33,
        num_layers = 2,
        device = device,
        vocab_size = vocab_size,
    )

    if use_checkpoint:
        checkpoint = torch.load("./model_saved/supervised.pt")
        net.load_state_dict(checkpoint)
        print("checkpoint loaded")

    net.to(device)
    torch.set_float32_matmul_precision("high")
    net = torch.compile(net)

    predict(net, test_loader, device, output_file="./output/supervised_output_pre.txt")
    net = train(net, total_epoch=50, device=device)
    predict(net, test_loader, device)
