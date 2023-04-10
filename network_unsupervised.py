import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from process_dataset import *
from torch.optim import Adam
import network


class UnsupervisedAutoencoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size,
        hidden_size,
        num_layers,
        label_size,
        attention_size,
        max_output,
    ):
        super(UnsupervisedAutoencoder, self).__init__()
        self.max_output = max_output
        self.encoder = network.Encoder(
            vocab_size, embed_size, hidden_size, num_layers, label_size
        )
        self.attention = network.Attention(attention_size)
        self.decoder = network.Decoder(vocab_size, hidden_size, num_layers, label_size)
        self.key_mapping = nn.Linear(hidden_size, attention_size,device=device)
        self.query_mapping = nn.Linear(hidden_size, attention_size,device=device)

    def forward(self, x, s_encode, s_decode):
        """
        x is the input of shape (batch_size, sentence_length), where x[i][j] is the j-th word's
        id in the i-th sentnece in the batch
        s_encode and s_decode are currently not used. Just set s_encode to be (batch_size, 1) of zeros
        and s_decode to be (batch_size,1) of ones would be fine

        output is of shape (batch_size, out_sentence_length, vocab_size), where output[i][j][k] is the
        possibility of using the word whose id is k at position j in the i-th sentence of the batch.
        """

        encoder_outputs, hidden = self.encoder(x, s_encode)  # (N, L, H)
        # print(f"{encoder_outputs.size()=}")
        keys = self.key_mapping(encoder_outputs)
        values = encoder_outputs

        outputs1 = []
        outputs2 = []
        for _ in range(self.max_output):
            query = self.query_mapping(hidden[0][1].unsqueeze(1))
            context, _ = self.attention(query, keys, values)
            output1, hidden = self.decoder(context, hidden, s_encode)
            query2 = self.query_mapping(hidden[0][1].unsqueeze(1))
            context2, _ = self.attention(query2, keys, values)
            output2, hidden = self.decoder(context2, hidden, s_decode)


            # print(f"{output.size()}")
            # if not torch.argmax(output, dim=-1).any():
            #     # all sentences are padding now
            #     break
            outputs1.append(output1)
            outputs2.append(output2)
        outputs1 = torch.cat(outputs1, dim=1)
        outputs2 = torch.cat(outputs2, dim=1)

        x2 = torch.argmax(outputs2, dim=2)
        encoder_outputs, hidden = self.encoder(x2, s_decode)  # (N, L, H)
        # print(f"{encoder_outputs.size()=}")
        keys = self.key_mapping(encoder_outputs)
        values = encoder_outputs

        outputs3 = []
        for _ in range(self.max_output):
            query = self.query_mapping(hidden[0][1].unsqueeze(1))
            context, _ = self.attention(query, keys, values)
            output3, hidden = self.decoder(context, hidden, s_decode)
            # print(f"{output.size()}")
            # if not torch.argmax(output, dim=-1).any():
            #     # all sentences are padding now
            #     break
            outputs3.append(output3)
        outputs3 = torch.cat(outputs3, dim=1)
        return outputs1, outputs2, outputs3


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    filename = './data/jigsaw/train.csv'
    dataset, w2id, id2w, vocab = generate_dataset(filename,device)
    train_loader, val_loader, test_loader = split_dataset(dataset)

    max_out_length = 100

    vocab_size = len(w2id)
    net = UnsupervisedAutoencoder(vocab_size, 32, 32, 4, 1, 8, max_out_length)

    net.to(device)
    # Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=0.001, weight_decay=0.0001)


    epoch = 0

    for i, (X,label) in enumerate(train_loader):
        print(i)
        if i % 100 == 0:
            checkpoint = {
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
            }
            epoch += 1
            f_path ="./checkpoints/unsupervised/'checkpoint.pt"
            torch.save(checkpoint,f_path)
        if epoch >= 2:
            break
        running_loss = 0.0
        running_acc = 0.0
        batch_size = X.shape[0]
        # zero the parameter gradients
        optimizer.zero_grad()
        # predict classes using data from the training set
        s_in = torch.zeros([batch_size,1],device=device)
        s_out = torch.ones([batch_size,1],device=device)


        # ***** outputs1: Upper decoder, outputs2: lower decoder, outputs3: backward transfer decoder **** #
        outputs1, outputs2, outputs3 = net.forward(X, s_in, s_out)

        
        # transfer labels into distribution vectors
        # print(X.shape,label.shape)
        y = torch.zeros(outputs1.shape,device=device)
        # print(y.shape)
        for i, sentence in enumerate(label):
            sentence_length = sentence.shape
            for j,index in enumerate(sentence):
                # print(i,j,index)
                y[i][j][index] = 1
        # print(outputs.shape, y.shape)
        # compute the loss based on model output and real labels
        loss1 = loss_fn(outputs1, y)
        loss2 = loss_fn(outputs3, y)

        # CLASSIFICATION LOSS : YET TO IMPLEMENT

        loss = loss1 + loss2
        # backpropagate the loss
        loss.backward()
        # adjust parameters based on the calculated gradients
        optimizer.step()
    print("finished training")
    torch.save(net,"./model_saved/unsupervised.pt")
    ### Start Test
    
    predicted = []
    for i, (toxic, detoxic) in enumerate(test_loader):
        if i >= 1:
            break
        batch_size = toxic.shape[0]
        s_in = torch.zeros([batch_size,1],device=device)
        s_out = torch.ones([batch_size,1],device=device)
        predicted = net.forward(toxic,s_in,s_out)

    print("finished prediction")
    
    predicted = predicted.argmax(axis = -1)
    with open("./output/unsupervised_output.txt","w") as f:
        for i, word_list in enumerate(tensor_to_words(predicted,id2w)):
            sentence = " ".join(word_list)
            f.write(sentence+"\n")

    # net = Autoencoder(3, 4, 4, 2, 1, 2, 4)
    # x = torch.tensor([[1, 2, 0, 2, 1], [1, 0, 0, 0, 1]])
    # s_in = torch.tensor([[0], [1]])
    # s_out = torch.tensor([[0], [1]])
    # result = net.forward(x, s_in, s_out)
    # print(f"{result}, {result.size()=}")
