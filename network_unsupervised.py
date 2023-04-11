import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from process_nonparallel_dataset import generate_nonparallel_dataset
from process_dataset import *
from torch.optim import Adam
import network
from tqdm import tqdm

def predict(net, test_loader, device, output_file="./output/unsupervised_output.txt"):
    ### Start Test

    predicted = []
    for i, [toxic] in enumerate(test_loader):
        if i >= 2:
            break
        batch_size = toxic.shape[0]
        s_in = torch.zeros([batch_size, 1], device=device)
        s_out = torch.ones([batch_size, 1], device=device)
        predicted.append(net.test_forward(toxic, s_in, s_out))

    predicted = torch.cat(predicted)

    print("finished prediction")

    predicted = predicted.argmax(dim=-1)
    with open(output_file, "w") as f:
        for i, word_list in enumerate(tensor_to_words(predicted, id2w)):
            sentence = " ".join(word_list)
            f.write(sentence + "\n")


def train(net, device, id2w, w2id):
    # Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
    classifier = network.Classifier(device)
    for param in classifier.parameters():
        param.requires_grad = False
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=w2id[pad_token])
    optimizer = Adam(net.parameters(), lr=0.001, weight_decay=0.0001)

    total_epoch = 2
    loss_traj = []
    for epoch in tqdm(range(total_epoch)):
        progress = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, [X] in progress:
            if i % 1000 == 0:
                checkpoint = {
                    "epoch": epoch + 1,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                f_path = "./checkpoints/unsupervised/checkpoint.pt"
                torch.save(checkpoint, f_path)
            batch_size = X.shape[0]
            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using data from the training set
            s_in = torch.zeros([batch_size, 1], device=device)
            s_out = torch.ones([batch_size, 1], device=device)
            result1, result2, result3 = net.forward(X, s_in, s_out)
            outputs1 = result1.transpose(1, 2)
            outputs3 = result3.transpose(1, 2)
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
            len_out1 = outputs1.size(-1)
            len_out3 = outputs3.size(-1)
            len_label = X.size(1)
            label_output = X

            len_diff1 = len_label - len_out1
            if len_diff1 > 0:
                # pad the "<PAD>" token at the end
                pad_tensor = F.one_hot(w2id[pad_token], len(w2id))
                outputs1 = torch.cat(
                    [outputs1, pad_tensor.repeat(outputs1.size(0), 1, len_diff1)], dim=1
                )
            elif len_diff1 < 0:
                label_output = torch.nn.functional.pad(
                    label_output, (0, -len_diff1), value=w2id[pad_token]
                )
            # print(f"{outputs.size()=}, {label.size()=}")
            # compute the loss based on model output and real labels
            loss1 = loss_fn(outputs1, label_output)

            label_output = X

            len_diff3 = len_label - len_out3
            if len_diff3 > 0:
                # pad the "<PAD>" token at the end
                pad_tensor = F.one_hot(w2id[pad_token], len(w2id))
                outputs3 = torch.cat(
                    [outputs3, pad_tensor.repeat(outputs3.size(0), 1, len_diff1)], dim=1
                )
            elif len_diff3 < 0:
                label_output = torch.nn.functional.pad(
                    label_output, (0, -len_diff3), value=w2id[pad_token]
                )
            # print(f"{outputs.size()=}, {label.size()=}")
            # compute the loss based on model output and real labels
            loss2 = loss_fn(outputs3, label_output)

            results1 = torch.argmax(result1, dim=2)
            results2 = torch.argmax(result2, dim=2)
            results3 = torch.argmax(result3, dim=2)

            sentences1 = []
            sentences2 = []
            sentences3 = []
            for i in range(results1.shape[0]):
                sentence = ""
                for j in range(results1.shape[1]):
                    if id2w[results1[i][j]] != pad_token or id2w[results1[i][j]] != end_of_sentence_token:
                        sentence += id2w[results1[i][j]]
                        sentence += " "
                sentences1.append(sentence)
            
            for i in range(results2.shape[0]):
                sentence = ""
                for j in range(results2.shape[1]):
                    if id2w[results2[i][j]] != pad_token or id2w[results2[i][j]] != end_of_sentence_token:
                        sentence += id2w[results2[i][j]]
                        sentence += " "
                sentences2.append(sentence)

            for i in range(results3.shape[0]):
                sentence = ""
                for j in range(results3.shape[1]):
                    if id2w[results3[i][j]] != pad_token or id2w[results3[i][j]] != end_of_sentence_token:
                        sentence += id2w[results3[i][j]]
                        sentence += " "
                sentences3.append(sentence)

            labels1 = classifier(sentences1)[:, 0]
            labels2 = classifier(sentences2)[:, 0]
            labels3 = classifier(sentences3)[:, 0]

            loss3 = torch.linalg.norm(labels1 - torch.ones_like(labels1))
            loss4 = torch.linalg.norm(labels2 - torch.zeros_like(labels2))
            loss5 = torch.linalg.norm(labels3 - torch.ones_like(labels3))
            
            loss = loss1 + loss2 + loss3 + loss4 + loss5
            # print(f"{outputs[:,1,0]=}, {torch.sum(label)=}")
            # backpropagate the loss
            loss.backward()
            progress.set_postfix({"loss": loss.item()})
            if i % 100 == 0:
                loss_traj.append(loss.item())
            # adjust parameters based on the calculated gradients
            optimizer.step()
    print("finished training")
    torch.save(net.state_dict(), "./model_saved/unsupervised.pt")
    return net

