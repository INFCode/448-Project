import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from detoxify import Detoxify
from process_dataset import *


class Encoder(nn.Module):
    def __init__(
        self, vocab_size, embed_size, hidden_size, num_layers, label_size, device
    ):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_size, padding_idx=0, device=device
        )
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers, batch_first=True, device=device
        )
        self.fc = nn.Linear(hidden_size + label_size, hidden_size, device=device)

    def forward(self, input_sentence, encode_label):
        embedded = self.embedding(input_sentence)  # (N, L, E)
        output, hidden = self.lstm(embedded)  # (N, L, H)
        label_new_shape = output.size()[:-1] + (1,)
        output = torch.cat(
            (output, encode_label.unsqueeze(-2).expand(*label_new_shape)), dim=-1
        )
        output = self.fc(output)
        return output, hidden


class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value):
        # print(f"{query.size()=}, {key.size()=}")
        score = torch.matmul(query, key.transpose(1, 2)) / self.sqrt_dim

        attn = F.softmax(score, -1)
        # print(f"{attn.size()=}")
        context = torch.matmul(attn, value)
        return context, attn


class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, label_size, device):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers, batch_first=True, device=device
        )
        self.fc = nn.Linear(hidden_size + label_size, vocab_size, device=device)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, context, hidden, decode_label):
        # print(f"{context.size()=}")
        output, hidden = self.lstm(context, hidden)
        label_new_shape = output.size()[:-1] + (1,)
        output = torch.cat(
            (output, decode_label.unsqueeze(-2).expand(*label_new_shape)), dim=-1
        )
        # print(f"1:{output.size()}")
        output = self.fc(output)
        # print(f"2:{output.size()}")
        # output = self.softmax(output)
        return output, hidden


class Autoencoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, device, vocab_size):
        super(Autoencoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, device = device)
        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, 
                               num_layers=num_layers, batch_first=True, device = device)
        self.attention = nn.Linear(hidden_dim, hidden_dim, device = device)
        self.decoder = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim,
                               num_layers=num_layers, batch_first=True, device = device)
        self.output = nn.Linear(embedding_dim, vocab_size, device= device)

    def forward(self, x):
        embedded = self.embedding(x)
        encoder_output, (hidden, cell) = self.encoder(embedded)
        energy = self.attention(encoder_output)
        attention_scores = torch.softmax(energy, dim=1)
        context_vector = torch.bmm(attention_scores.permute(0, 2, 1), encoder_output)
        decoder_output, _ = self.decoder(context_vector, (hidden, cell))
        output = self.output(decoder_output)
        return output


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
        device,
    ):
        super(UnsupervisedAutoencoder, self).__init__()
        self.max_output = max_output
        self.encoder = Encoder(
            vocab_size, embed_size, hidden_size, num_layers, label_size, device
        )
        self.attention = Attention(attention_size)
        self.decoder = Decoder(vocab_size, hidden_size, num_layers, label_size, device)
        self.key_mapping = nn.Linear(hidden_size, attention_size, device=device)
        self.query_mapping = nn.Linear(hidden_size, attention_size, device=device)

    def test_forward(self, x, s_encode, s_decode):
        encoder_outputs, hidden = self.encoder(x, s_encode)  # (N, L, H)
        # print(f"{encoder_outputs.size()=}")
        keys = self.key_mapping(encoder_outputs)
        values = encoder_outputs

        outputs = []
        for _ in range(self.max_output):
            query = self.query_mapping(hidden[0][-1].unsqueeze(1))
            context, _ = self.attention(query, keys, values)
            output, hidden = self.decoder(context, hidden, s_decode)
            # print(f"{output.size()}")
            # if not torch.argmax(output, dim=-1).any():
            #     # all sentences are padding now
            #     break
            outputs.append(output)
        outputs = torch.cat(outputs, dim=1)
        return outputs

    def forward(self, x, s_encode, s_decode):
        """
        x is the input of shape (batch_size, sentence_length), where x[i][j] is the j-th word's
        id in the i-th sentnece in the batch
        s_encode and s_decode are currently not used. Just set s_encode to be (batch_size, 1) of zeros
        and s_decode to be (batch_size,1) of ones would be fine

        output1, output2, output3 are of shape (batch_size, out_sentence_length, vocab_size), where output[i][j][k] is the
        possibility of using the word whose id is k at position j in the i-th sentence of the batch.
        """

        encoder_outputs, hidden = self.encoder(x, s_encode)  # (N, L, H)
        # print(f"{encoder_outputs.size()=}")
        keys = self.key_mapping(encoder_outputs)
        values = encoder_outputs

        outputs1 = []
        outputs2 = []
        for _ in range(self.max_output):
            query = self.query_mapping(hidden[0][-1].unsqueeze(1))
            context, _ = self.attention(query, keys, values)
            output1, hidden = self.decoder(context, hidden, s_encode)
            query2 = self.query_mapping(hidden[0][-1].unsqueeze(1))
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
            query = self.query_mapping(hidden[0][-1].unsqueeze(1))
            context, _ = self.attention(query, keys, values)
            output3, hidden = self.decoder(context, hidden, s_decode)
            # print(f"{output.size()}")
            # if not torch.argmax(output, dim=-1).any():
            #     # all sentences are padding now
            #     break
            outputs3.append(output3)
        outputs3 = torch.cat(outputs3, dim=1)
        return outputs1, outputs2, outputs3


class Classifier(nn.Module):
    def __init__(self, device):
        super(Classifier, self).__init__()
        self.model = Detoxify("original-small", device=device)
        # this model should not be trained
        self.device = device

    def forward(self, text):
        """
        Simply send the forward request to the pre-trained model.
        Args:
        text: A string or a list of B strings that represents the (batch of) string to
              classify
        Returns:
        score: A tensor of shape (B, C) where C is the number of classes.
        """
        self.model.model.eval()
        inputs = self.model.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        ).to(self.model.device)
        out = self.model.model(**inputs)[0]
        scores = torch.sigmoid(out)
        return scores

    def get_class_names(self, index):
        """
        Get the name of the class of toxicity
        """
        return self.model.class_names[index]
