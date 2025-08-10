import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence

class SentimentDataset(Dataset):
    def __init__(self, text, labels):
        self.text = text
        self.labels = labels

    def __getitem__(self, idx):
        sentence = self.text.iloc[idx]
        label = self.labels.iloc[idx]
        return {"sentence": sentence,
                "label": label}

    def __len__(self):
        return len(self.labels)


class LSTMClassifier(nn.Module):
    """
    Define multi-classification LSTM neural network.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, bidirectional, dropout, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text, text_lengths):
        # text = [batch size,sent_length]
        embedded = self.embedding(text)
        # embedded = [batch size, sent_len, emb dim]

        packed_embedded = pack_padded_sequence(embedded, text_lengths, batch_first=True)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # hidden = [batch size, num layers * num directions,hid dim]
        # cell = [batch size, num layers * num directions,hid dim]

        # concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        # hidden = [batch size, hid dim * num directions]
        output = self.fc(hidden)

        return output