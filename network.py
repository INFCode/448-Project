import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, label_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size + label_size, hidden_size)

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
    def __init__(self, vocab_size, hidden_size, num_layers, label_size):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size + label_size, vocab_size)
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
        output = self.softmax(output)
        return output, hidden


class Autoencoder(nn.Module):
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
        super(Autoencoder, self).__init__()
        self.max_output = max_output
        self.encoder = Encoder(
            vocab_size, embed_size, hidden_size, num_layers, label_size
        )
        self.attention = Attention(attention_size)
        self.decoder = Decoder(vocab_size, hidden_size, num_layers, label_size)
        self.key_mapping = nn.Linear(hidden_size, attention_size)
        self.query_mapping = nn.Linear(hidden_size, attention_size)

    def forward(self, x, s_encode, s_decode):
        encoder_outputs, hidden = self.encoder(x, s_encode)  # (N, L, H)
        # print(f"{encoder_outputs.size()=}")
        keys = self.key_mapping(encoder_outputs)
        values = encoder_outputs

        outputs = []
        for _ in range(self.max_output):
            query = self.query_mapping(hidden[0][1].unsqueeze(1))
            context, _ = self.attention(query, keys, values)
            output, hidden = self.decoder(context, hidden, s_decode)
            print(f"{output.size()}")
            if not torch.argmax(output, dim=-1).any():
                # all sentences are padding now
                break
            outputs.append(output)
        outputs = torch.cat(outputs, dim=1)
        return outputs


if __name__ == "__main__":
    net = Autoencoder(3, 4, 4, 2, 1, 2, 4)
    x = torch.tensor([[1, 2, 0, 2, 1], [1, 0, 0, 0, 1]])
    s_in = torch.tensor([[0], [1]])
    s_out = torch.tensor([[0], [1]])
    result = net.forward(x, s_in, s_out)
    print(f"{result}, {result.size()=}")