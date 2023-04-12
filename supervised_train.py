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
        if i >= 2:
            break
        batch_size = toxic.shape[0]
        s_in = torch.zeros([batch_size, 1], device=device)
        s_out = torch.ones([batch_size, 1], device=device)
        predicted.append(net.forward(toxic, s_in, s_out))

    predicted = torch.cat(predicted)

    print("finished prediction")

    predicted = predicted.argmax(dim=-1)
    with open(output_file, "w") as f:
        for i, word_list in enumerate(tensor_to_words(predicted, id2w)):
            sentence = " ".join(word_list)
            f.write(sentence + "\n")


def train(net, device):
    def _get_metrics(loader):
        fluency = 0
        detoxification=0 
        synonym = 0
        criterion = torch.nn.CrossEntropyLoss()
        running_loss = []
        s_in = torch.zeros([batch_size, 1], device=device)
        s_out = torch.ones([batch_size, 1], device=device)
        for X,y in loader:
            with torch.no_grad():
                output = net.forward(X, s_in, s_out)
                running_loss.append(criterion(output,y).item())
        loss = torch.mean(running_loss)
        return fluency, detoxification, synonym, loss
    

    # Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
    loss_fn = nn.CrossEntropyLoss(ignore_index=w2id[pad_token])
    optimizer = Adam(net.parameters(), lr=5e-4, weight_decay=1e-4)

    total_epoch = 2
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
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0
            batch_size = X.shape[0]
            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using data from the training set
            s_in = torch.zeros([batch_size, 1], device=device)
            s_out = torch.ones([batch_size, 1], device=device)
            result = net.forward(X, s_in, s_out)
            outputs = result.transpose(1, 2)
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
            len_out = outputs.size(-1)
            len_label = label.size(1)

            len_diff = len_label - len_out
            if len_diff > 0:
                # pad the "<PAD>" token at the end
                pad_tensor = F.one_hot(w2id[pad_token], vocab_size)
                outputs = torch.cat(
                    [outputs, pad_tensor.repeat(outputs.size(0), 1, len_diff)], dim=1
                )
            elif len_diff < 0:
                label = torch.nn.functional.pad(
                    label, (0, -len_diff), value=w2id[pad_token]
                )
            # print(f"{outputs.size()=}, {label.size()=}")
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, label)
            # print(f"{outputs[:,1,0]=}, {torch.sum(label)=}")
            # backpropagate the loss
            loss.backward()
            running_loss += loss.item()
            progress.set_postfix({"loss": loss.item()})
            if i % 100 == 0:
                loss_traj.append(loss.item())
            # adjust parameters based on the calculated gradients
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

    max_out_length = 100

    vocab_size = len(w2id)
    net = Autoencoder(vocab_size, 32, 1, 1, 1, 8, max_out_length, device)

    if use_checkpoint:
        checkpoint = torch.load("./model_saved/supervised.pt")
        net.load_state_dict(checkpoint)
        print("checkpoint loaded")

    net.to(device)

    predict(net, test_loader, device, output_file="./output/supervised_output_pre.txt")
    net = train(net, device=device)
    predict(net, test_loader, device)
